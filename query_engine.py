import spacy
from spacy.matcher import Matcher
import open_clip
from open_clip import tokenizer
import torch
import numpy as np
import pickle
import random
import re


class QueryParser:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.matcher = Matcher(self.nlp.vocab)
        self._register_patterns()

    def _register_patterns(self):
        # Define spatial relation patterns
        patterns = {
            "NEXT_TO": [{"LOWER": "next"}, {"LOWER": "to"}],
            "NEAR":[{"LOWER": "next"}],
            "IN_FRONT_OF": [{"LOWER": "in"}, {"LOWER": "front"}, {"LOWER": "of"}],
            "ON_TOP_OF": [{"LOWER": "on"}, {"LOWER": "top"}, {"LOWER": "of"}],
            "TO_LEFT_OF": [{"LOWER": "to"}, {"LOWER": "the"}, {"LOWER": "left"}, {"LOWER": "of"}],
            "TO_RIGHT_OF": [{"LOWER": "to"}, {"LOWER": "the"}, {"LOWER": "right"}, {"LOWER": "of"}],
            "BEHIND": [{"LOWER": "behind"}],
            "ON": [{"LOWER": "on"}],
            "UNDER": [{"LOWER": "under"}],
            "ABOVE": [{"LOWER": "above"}]
        }
        for key, pattern in patterns.items():
            self.matcher.add(key, [pattern])

    def parse(self, query: str):
        doc = self.nlp(query.lower())
        matches = self.matcher(doc)

        if not matches:
            return {"target": self._extract_target(doc), "relation": "", "reference": ""}

        match_id, start, end = max(matches, key=lambda x: x[2] - x[1])
        relation = doc[start:end].text
        before = doc[:start].text.strip()
        after = doc[end:].text.strip()

        target = self._extract_noun_phrase(before, side="end")
        reference = self._extract_noun_phrase(after, side="start")

        return {"target": target, "relation": relation, "reference": reference}

    def _extract_noun_phrase(self, text: str, side="end") -> str:
        doc = self.nlp(text)
        noun_chunks = list(doc.noun_chunks)
        if not noun_chunks:
            return text
        return noun_chunks[-1].text if side == "end" else noun_chunks[0].text

    def _extract_target(self, doc):
        for token in doc:
            if token.dep_ == "dobj" and token.head.pos_ == "VERB":
                return self._extract_noun_phrase(token.text, side="end")
        return ""

def normalize(text):
    text = text.lower().strip()
    text = re.sub(r'\b(the|a|an)\b', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.replace(' ', '_')
    return text

def extract_text_feature(description, model):
    with torch.no_grad():
        text_tokens = tokenizer.tokenize([description])
        text_features = model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy()[0]

def find_top_k_similar(object_features, text_feature, k=5):
    object_ids = list(object_features.keys())
    features = np.stack([
        object_features[obj_id][0] if object_features[obj_id].ndim == 2 else object_features[obj_id]
        for obj_id in object_ids
    ])
    sims = features @ text_feature
    topk_idx = sims.argsort()[-k:][::-1]
    return [(object_ids[i], sims[i]) for i in topk_idx]

def check_spatial_relation(G, target_ids, ref_ids, relation_text):
    matches = []
    best_match = None
    best_sim_sum = -float("inf")
    for t_id, t_sim in target_ids:
        for r_id, r_sim in ref_ids:
            if G.has_edge(t_id, r_id) or G.has_edge(r_id, t_id):
                edge_relation = G[t_id][r_id].get("relation", [])
                if any(normalize(rel) == normalize(relation_text) for rel in edge_relation):
                    matches.append((t_id, r_id, relation_text, t_sim, r_sim))
                    sim_sum = t_sim + r_sim
                    if sim_sum > best_sim_sum:
                        best_match = (t_id, r_id, relation_text, t_sim, r_sim)
                        best_sim_sum = sim_sum
    return matches, best_match

class QueryEngine:
    def __init__(self, feature_path, graph_path):
        self.parser = QueryParser()
        self.model, _, _ = open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b_s32b_b79k")
        self.model.to("cpu")
        self.model.eval()

        self.object_features = np.load(feature_path, allow_pickle=True).item()

        with open(graph_path, "rb") as f:
            self.G = pickle.load(f)

    def run_query(self, query_text):
        parsed = self.parser.parse(query_text)
        target_text = parsed["target"]
        ref_text = parsed["reference"]
        relation = parsed["relation"]

        top5_target, top5_ref = [], []
        if target_text:
            t_feat = extract_text_feature(target_text, self.model)
            top5_target = find_top_k_similar(self.object_features, t_feat, k=5)

        if ref_text:
            r_feat = extract_text_feature(ref_text, self.model)
            top5_ref = find_top_k_similar(self.object_features, r_feat, k=5)

        matched_edges, best_match = check_spatial_relation(self.G, top5_target, top5_ref, relation)

        return {
            "parsed": parsed,
            "top_target": top5_target,
            "top_reference": top5_ref,
            "matches": matched_edges,
            "best_match": best_match
        }
    
