import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset.dataloader_definition import embedding_loader
 # importa dal file dove lo hai definito


# 📦 Classe MLP per classificazione della decade (10 classi)
class AgeGroupClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)
    


# 🧠 Funzione per addestrare il classificatore
def train_classifier(embeddings, ages, input_dim, device='cpu', epochs=5500, batch_size=32):
    # 🎯 Calcolo delle etichette decade (0–9)
    targets = torch.tensor([int(age.item()) // 10 for age in ages], dtype=torch.long)

    dataset = TensorDataset(embeddings, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = AgeGroupClassifier(input_dim=input_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == yb).sum().item()

        accuracy = 100 * correct / len(dataset)
        print(f"📚 Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f} - Accuracy: {accuracy:.2f}%")

    # 💾 Salvataggio
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/classifier.pth")
    print("✅ Classificatore salvato in checkpoints/classifier.pth")
    return model
# 🔁 Estrai tutti i dati reali dal dataloader
def extract_embeddings_and_labels(dataloader, device):
    X_list = []
    y_list = []

    for embedding, age in dataloader:
        X_list.append(embedding.to(device))
        y_list.append(age.to(device))

    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)
    return X, y

# 🔽 Esecuzione diretta per usare i tuoi dati reali
if __name__ == '__main__':
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Estrai dati reali dal tuo embedding_loader
    X_real, y_real = extract_embeddings_and_labels(embedding_loader, device)

    # Addestra il classificatore
    train_classifier(X_real, y_real, input_dim=X_real.shape[1], device=device)