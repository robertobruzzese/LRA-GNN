import torch
from torch.utils.data import DataLoader
from dataset.embedding_dataset import EmbeddingDataset
from models.progressive_rl import ProgressiveRLAgent
from models.lra_gnn import LRA_GNN
from training.rl_environment import RLEnvironment
from models.classifier import AgeGroupClassifier

# 📌 Parametri
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_dir = "embeddings/val"  # cambia con "train" o "test" se vuoi
checkpoint_path = "checkpoints/rl_agent_best.pth"

# ⚙️ Hyperparametri coerenti col training
state_dim = 134  # esempio: embedding (128) + delta_x + delta_y + pos
action_dim = 5   # su, giù, sinistra, destra, resta

# 🔁 Carica il dataset
dataset = EmbeddingDataset(embedding_dir)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# 🎯 Carica il modello LRA-GNN (serve per feature estratte se non già incluse)
model = LRA_GNN()  # se serve per il forward
model.to(device)
model.eval()

# 🔁 Carica l’agente RL
agent = ProgressiveRLAgent(state_dim=state_dim, action_dim=action_dim)
# 📥 Carica il modello salvato
agent.q_network.load_state_dict(torch.load("checkpoints/best_agent.pth"))
print(agent.q_network.fc1.weight[0][:5])  # ad esempio

# ➕ Inizializza il classificatore e assegna all'agente
embedding_dim = next(iter(dataloader))[0].shape[1]
classifier = AgeGroupClassifier(input_dim=embedding_dim).to(device)
classifier.load_state_dict(torch.load("checkpoints/classifier.pth"))
classifier.eval()

agent.classifier = classifier
# 🔬 Ora puoi usarlo per valutare
results = agent.evaluate(model=None, dataloader=dataloader, device=device)
print(results)

