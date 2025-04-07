import torch
from torch.utils.data import DataLoader
from models.progressive_rl import ProgressiveRLAgent
from training.train_rl import train_prlae
from dataset.embedding_dataset import EmbeddingDataset
from models.classifier import AgeGroupClassifier
import os


# Dispositivo
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Carica dataset di embedding
embedding_dataset = EmbeddingDataset("embeddings/train")
embedding_loader = DataLoader(embedding_dataset, batch_size=1, shuffle=True)

# Calcola dinamicamente la dimensione dello stato
first_embedding, _ = embedding_dataset[0]
embedding_dim = first_embedding.shape[0]
state_dim = embedding_dim + 6  # embedding + delta_x + delta_y + pos_r + pos_c
action_dim = 5  # su, giù, sinistra, destra, resta

# 🔹 Carica il classificatore già addestrato (oppure inizializzalo se non addestrato ancora)
classifier = AgeGroupClassifier(input_dim=128).to(device)

if os.path.exists("checkpoints/classifier.pth"):
    try:
        classifier.load_state_dict(torch.load("checkpoints/classifier.pth", map_location=device))
        classifier.eval()
        print("✅ Classificatore caricato correttamente.")
    except RuntimeError as e:
        print("⚠️ Errore nel caricamento del classificatore:", str(e))
        print("👉 Esegui di nuovo `train_classifier.py` per rigenerare il file.")
        exit(1)
else:
    print("⚠️ Nessun classificatore trovato: esegui prima train_classifier.py")
    exit(1)


#classifier.load_state_dict(torch.load("checkpoints/classifier.pth"))  # solo se già addestrato
#classifier.eval()  # metti in modalità eval se non deve essere allenato

# 🔹 Istanzia l'agente con il classificatore
agent = ProgressiveRLAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    device=device,
    classifier=classifier  # 👈 passa il classificatore
)
agent.q_network.to(device)

# Allena l'agente
if __name__ == "__main__":
    train_prlae(
        agent=agent,
        dataloader=embedding_loader,
        device=device,
        num_episodes=80
    )


# Salva l'agente
#agent.save("checkpoints/rl_agent_best.pth")
print("\n🏁 Training RL completato!")
