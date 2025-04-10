import os
import glob
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from utils.graph_constructor_fgnet import construct_initial_graph
from utils.lrc import LatentRelationCapturer
from utils.deep_feature_extraction import extract_deep_features
from dataset.embedding_dataset import EmbeddingDataset


# === Configurazione ===
patches_dir = "datasets/data/FGNET/patches"
images_dir = "datasets/data/FGNET/images/Train/images_preprocessed"
embeddings_root = "embeddings_FGNET"
image_size = (224, 224)
num_patches_x, num_patches_y = 6, 6
threshold_distance = 0.88
tau_random_walk = 0.87
num_heads = 12
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
num_layers = 4  # per ResGCN

# === Elenco immagini preprocessate ===
image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for image_filename in image_files:
    image_name = os.path.splitext(image_filename)[0]
    csv_path = os.path.join(patches_dir, f"{image_name}_patches.csv")
    image_path = os.path.join(images_dir, image_filename)
    embedding_dir = os.path.join(embeddings_root, image_name)
    os.makedirs(embedding_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        print(f"⚠️  Salto {image_name}: CSV mancante ({csv_path})")
        continue

    print(f"\n📂 Processing {image_name}...")

    # === Step 1: Costruzione grafo iniziale e RW
    graph_initial, graph_augmented, patches_tensor, patch_to_node = construct_initial_graph(
        csv_path=csv_path,
        image_path=image_path,
        image_size=image_size,
        num_patches_x=num_patches_x,
        num_patches_y=num_patches_y,
        threshold_distance=threshold_distance,
        tau_random_walk=tau_random_walk,
        embedding_dir=embedding_dir
    )

    # === Step 2: Salvataggio grafi iniziali
    torch.save(graph_initial, os.path.join(embedding_dir, "graph_initial.pt"))
    torch.save(graph_augmented, os.path.join(embedding_dir, "graph_rw.pt"))
    print("💾 Grafi iniziale e RW salvati.")

    # === Step 3: LRC – Attention & grafi completamente connessi
    X = graph_augmented.x
    lrc = LatentRelationCapturer(in_dim=X.shape[1], num_heads=num_heads)
    A_m_list = lrc(X)

    for idx, A_m in enumerate(A_m_list):
        edge_index_m, edge_weight_m = dense_to_sparse(A_m)
        graph_m = Data(x=X, edge_index=edge_index_m, edge_attr=edge_weight_m)
        torch.save(graph_m, os.path.join(embedding_dir, f"graph_lrc_{idx}.pt"))

    print(f"💾 Grafi LRC (12 teste) salvati in {embedding_dir}")

    # === Step 4: Deep Feature Extraction con ResGCN
    graph_paths = sorted(glob.glob(os.path.join(embedding_dir, "graph_lrc_*.pt")))
    deep_feats = extract_deep_features(graph_paths, feature_dim=X.shape[1], num_layers=num_layers, device=device)
    torch.save(deep_feats, os.path.join(embedding_dir, "deep_features.pt"))
    print("💾 Deep features salvate.")

print("\n✅ Tutte le immagini sono state elaborate.")
