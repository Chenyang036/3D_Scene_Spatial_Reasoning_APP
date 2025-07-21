import open_clip
import torch

def preload_openclip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained="laion2b_s32b_b79k"
    )
    model.to(device)
    print("OpenCLIP model loaded and cached.")

if __name__ == "__main__":
    preload_openclip()
