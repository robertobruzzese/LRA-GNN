import os
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

class EmbeddingDataset(Dataset):
    def __init__(self, embeddings_dir):
        self.embeddings_dir = embeddings_dir
        self.files = [f for f in os.listdir(embeddings_dir) if f.endswith(".pt")]
        self.files.sort()  # per coerenza

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.embeddings_dir, self.files[idx])
        data = torch.load(path)

        # data dovrebbe essere un dizionario con chiavi: 'embedding', 'age', 'filename'
        embedding = data['embedding']           # Tensor shape (D,)
        age = data['age']                       # Valore float
        filename = data.get('filename', None)   # (opzionale) utile per debug o tracciamento

        return embedding, torch.tensor(age, dtype=torch.float32)

# Esempio d'uso:
# train_dataset = EmbeddingDataset("embeddings/train")
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)  # Per PRLAE
