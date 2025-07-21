import streamlit as st
import plotly.graph_objects as go
import open3d as o3d
import numpy as np
import pickle
import torch
from query_engine import QueryEngine  
from sklearn.neighbors import KDTree
import logging

DEFAULT_CAMERA = dict(eye=dict(x=1.5, y=1.5, z=1.5))
logger = st.logger.get_logger(__name__)
@st.cache_resource
def load_mesh(path):
    mesh = o3d.io.read_triangle_mesh(path)
    return mesh

def plot_original_mesh(mesh):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]

    if mesh.has_vertex_colors():
        vertex_colors = np.asarray(mesh.vertex_colors)
    else:
        vertex_colors = np.ones_like(vertices) * 0.8

    face_colors = vertex_colors[triangles].mean(axis=1)
    face_colors_rgb = (face_colors * 255).astype(np.uint8)
    face_colors_hex = [f'rgb({r},{g},{b})' for r, g, b in face_colors_rgb]

    mesh3d = go.Mesh3d(
        x=x, y=y, z=z,
        i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2],
        facecolor=face_colors_hex,
        flatshading=True,
        opacity=1.0
    )

    fig = go.Figure(data=[mesh3d])
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data',
            camera=DEFAULT_CAMERA
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        height=400,
        showlegend=False
    )
    return fig


def plot_mesh_with_highlight(mesh, sampled_coords=None, object_dict=None,
                             target_id=None, reference_id=None):
    import copy
    original_mesh = copy.deepcopy(mesh)

    # === 1. Simplify mesh for background ===
    background_mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=10000)
    background_mesh.compute_vertex_normals()
    vertices_bg = np.asarray(background_mesh.vertices)
    triangles_bg = np.asarray(background_mesh.triangles)

    base_mesh = go.Mesh3d(
        x=vertices_bg[:, 0], y=vertices_bg[:, 1], z=vertices_bg[:, 2],
        i=triangles_bg[:, 0], j=triangles_bg[:, 1], k=triangles_bg[:, 2],
        color='lightgray',
        opacity=0.1,
        flatshading=False,
        lighting=dict(ambient=0.3, diffuse=0.8, specular=0.5, roughness=0.5),
        lightposition=dict(x=100, y=200, z=300),
        name='Scene'
    )

    # === 2. Highlight meshes ===
    highlight_meshes = []
    if object_dict and sampled_coords is not None:
        vertices_full = np.asarray(original_mesh.vertices)
        triangles_full = np.asarray(original_mesh.triangles)
        tree = KDTree(vertices_full)

        highlight_configs = {
            'Reference': (reference_id, 'deepskyblue'),
            'Target': (target_id, 'gold')
        }

        for label, (obj_id, color) in highlight_configs.items():
            if obj_id is None or obj_id not in object_dict:
                continue

            point_ids = object_dict[obj_id]['point_ids']
            coords = sampled_coords[point_ids]
            matched_idx = tree.query_radius(coords, r=0.02)

            highlight_vertex_indices = set()
            for idxs in matched_idx:
                highlight_vertex_indices.update(idxs)
            highlight_vertex_indices = np.array(list(highlight_vertex_indices))

            mask = np.isin(triangles_full, highlight_vertex_indices).any(axis=1)
            sub_triangles = triangles_full[mask]

            highlight_mesh = go.Mesh3d(
                x=vertices_full[:, 0], y=vertices_full[:, 1], z=vertices_full[:, 2],
                i=sub_triangles[:, 0], j=sub_triangles[:, 1], k=sub_triangles[:, 2],
                color=color,
                opacity=0.85,
                flatshading=False,
                lighting=dict(ambient=0.5, diffuse=0.9, specular=0.8),
                name=f'{label} Object'
            )
            highlight_meshes.append(highlight_mesh)

    # === 3. Combine and return ===
    fig = go.Figure(data=[base_mesh] + highlight_meshes)
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data',
            camera=DEFAULT_CAMERA
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=400,
        showlegend=True
    )
    return fig
    
logger = logging.getLogger(__name__)
st.set_page_config(page_title="3D Spatial Query Demo", layout="wide")
st.title("🔭 Demo: Open-Vocabulary 3D Spatial Querying")

# Sidebar options
scene_choice = st.sidebar.selectbox("Choose a Scene", ["kitchen", "bedroom"])
scene_path = f"data/{scene_choice}/{scene_choice}_mesh.ply"

example_queries = [
    "Find the chair next to the table",
    "Which object is in front of the bed?",
    "Show the object under the desk"
]

if st.sidebar.button("🎲 Random Query"):
    query = np.random.choice(example_queries)
else:
    query = ""

with st.sidebar.expander("📐 Supported Spatial Relations"):
    st.markdown(
        """
        - in front of
        - behind
        - next to
        - on top of
        - under
        - above
        - below
        - to the left of
        - to the right of
        """
    )

# Initialize query engine
engine = QueryEngine(
    feature_path=f"data/{scene_choice}/{scene_choice}_object_features.npy",
    graph_path=f"data/{scene_choice}/{scene_choice}_graph_structure.pkl"
)

# Input query
query_input = st.text_input("Enter a spatial query:", value=query, key="user_query")
submitted = st.button("🔍 Submit Query", key="submit_query")

# Load mesh once
mesh = load_mesh(scene_path)

# Layout: 2 columns
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🖼️ Scene Preview")
    fig = plot_original_mesh(mesh)
    st.plotly_chart(fig, use_container_width=True, key="original_mesh", height=800)

with col2:
    if submitted and query_input.strip():
        result = engine.run_query(query_input.strip())
        best_match = result.get("best_match")
        if best_match:
            t_id, r_id, relation, t_sim, r_sim = best_match
            logger.info(f"Highlighting best match: {best_match}")
            downsample_pth_path = f"data/{scene_choice}/{scene_choice}_downsample.pth"
            sample = torch.load(downsample_pth_path, weights_only=False)
            sampled_coords = sample['sampled_coords']
            object_dict = np.load(f"data/{scene_choice}/object_dict_with_center.npy", allow_pickle=True).item()
            fig = plot_mesh_with_highlight(mesh, sampled_coords, object_dict, t_id, r_id)
            st.markdown("### 🎯 Highlighted Match")
            st.plotly_chart(fig, use_container_width=True, key="highlighted_mesh", height=800)
        else:
            st.markdown("### 🎯 Highlighted Match")
            st.info("No best match found.")

            