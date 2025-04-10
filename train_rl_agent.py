import argparse
import os
import torch
from torch.utils.data import DataLoader
from models.progressive_rl import ProgressiveRLAgent
from training.train_rl import train_prlae
from dataset.embedding_dataset import EmbeddingDataset
from models.classifier import AgeGroupClassifier
from datetime import datetime

# 🔧 Argomento per scegliere il dataset
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="MORPH", help="Nome del dataset (MORPH o FGNET)")
args = parser.parse_args()
dataset_name = args.dataset.upper()

# ⚙️ Dispositivo
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 📂 Seleziona cartella embeddings corretta
embedding_dir = "embeddings/train" if dataset_name == "MORPH" else "embeddings_FGNET/train"

# 📥 Carica dataset di embedding
embedding_dataset = EmbeddingDataset(embedding_dir)
embedding_loader = DataLoader(embedding_dataset, batch_size=1, shuffle=True)

# 🔢 Calcola dinamicamente la dimensione dello stato
first_embedding, _ = embedding_dataset[0]
embedding_dim = first_embedding.shape[0]
state_dim = embedding_dim + 6  # embedding + delta_x + delta_y + pos_r + pos_c
action_dim = 5  # su, giù, sinistra, destra, resta

# 🔍 Classificatore pre-addestrato
classifier = AgeGroupClassifier(input_dim=128).to(device)
classifier_path = os.path.join("checkpoints", dataset_name, "classifier.pth")


if os.path.exists(classifier_path):
    try:
        classifier.load_state_dict(torch.load(classifier_path, map_location=device))
        classifier.eval()
        print(f"✅ Classificatore caricato da {classifier_path}")
    except RuntimeError as e:
        print("⚠️ Errore nel caricamento del classificatore:", str(e))
        print("👉 Esegui di nuovo `train_classifier.py` per rigenerare il file.")
        exit(1)
else:
    print(f"⚠️ Nessun classificatore trovato in {classifier_path}: esegui prima train_classifier.py")
    exit(1)


# 🤖 Istanzia l’agente RL
agent = ProgressiveRLAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    device=device,
    classifier=classifier
)
agent.q_network.to(device)

# 🚀 Allena l’agente
if __name__ == "__main__":
    train_prlae(
        agent=agent,
        dataloader=embedding_loader,
        device=device,
        dataset_name=dataset_name,
        num_episodes=200
    )
     # 💾 Salva il modello in cartella diversa per dataset
    # 🔹 Crea la directory di salvataggio in base al dataset
    checkpoint_dir = os.path.join("checkpoints", dataset_name.upper())  # garantisce maiuscolo
    os.makedirs(checkpoint_dir, exist_ok=True)
    # 🔹 Costruisci il percorso del file di salvataggio
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(checkpoint_dir, f"rl_agent_{timestamp}.pth")

    # 🔹 Salva l'agente
    agent.save(save_path)
    print(f"\n💾 RL agent salvato in: {save_path}")
    


# 💾 (Opzionale) Salva il modello addestrato
# agent.save("checkpoints/rl_agent_best.pth")

print("\n🏁 Training RL completato!")
