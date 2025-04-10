import os
import torch
from torch_geometric.data import Dataset
import re

class AgeEstimationDataset(Dataset):
    def __init__(self, root_dir, dataset_name="MORPH", transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.dataset_name = dataset_name.upper()
        self.transform = transform

        image_dir = os.path.join(root_dir, "images", "Train")

        if self.dataset_name == "MORPH":
            embeddings_dir = "embeddings"
            age_regex = r'[MF](\d{1,3})'
        elif self.dataset_name == "FGNET":
            embeddings_dir = "embeddings_FGNET"
            age_regex = r'A(\d{1,3})'
        else:
            raise ValueError(f"Dataset sconosciuto: {dataset_name}")

        self.samples = []
        for fname in os.listdir(image_dir):
            if fname.lower().endswith(".jpg"):
                match = re.search(age_regex, fname)
                if match:
                    age = int(match.group(1))
                    image_name = os.path.splitext(fname)[0]
                    graph_path = os.path.join(embeddings_dir, image_name, "graph_rw.pt")

                    if os.path.exists(graph_path):
                        self.samples.append({
                            "image_name": fname,
                            "age": age
                        })
                    else:
                        print(f"⚠️ Skipping {image_name}: missing graph_rw.pt")
                else:
                    print(f"⚠️ Impossibile estrarre l'età da: {fname}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_name = os.path.splitext(sample["image_name"])[0]
        age = sample["age"]

        graph_path = os.path.join(
            "embeddings_FGNET" if self.dataset_name == "FGNET" else "embeddings",
            image_name, "graph_rw.pt"
        )
        graph = torch.load(graph_path)
        graph.y = torch.tensor([age], dtype=torch.float)
        return graph
