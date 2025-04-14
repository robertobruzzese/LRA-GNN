import torch
from torch.utils.data import DataLoader
from dataset.embedding_dataset import EmbeddingDataset
from models.progressive_rl import ProgressiveRLAgent
from models.lra_gnn import LRA_GNN
from training.rl_environment import RLEnvironment
from models.classifier import AgeGroupClassifier
import os
import argparse
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import glob



# 🔧 Parsing argomento --dataset
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="MORPH", help="Dataset: MORPH o FGNET")
args = parser.parse_args()
dataset_name = args.dataset.upper()

# 📌 Parametri
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
embedding_dir = "embeddings/val" if dataset_name == "MORPH" else "embeddings_FGNET/val"
checkpoint_dir = os.path.join("checkpoints", dataset_name)
# 🔍 Cerca tutti i best_agent con timestamp nella cartella
agent_files = glob.glob(os.path.join(checkpoint_dir, "best_agent_*.pth"))

if not agent_files:
    raise FileNotFoundError("❌ Nessun file best_agent_*.pth trovato in checkpoints!")

# 📅 Ordina per timestamp (in base al nome)
agent_files.sort()  # ordine alfabetico equivale a ordine temporale grazie al formato YYYY-MM-DD_HH-MM-SS

# 🆕 Prende l'ultimo
agent_path = agent_files[-1]

print(f"📂 File best_agent selezionato: {agent_path}")

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
agent_files = glob.glob(os.path.join(checkpoint_dir, "best_agent_*.pth"))
if not agent_files:
    raise FileNotFoundError("❌ Nessun file best_agent_*.pth trovato in checkpoints!")
agent_files.sort()
agent_path = agent_files[-1]
print(f"📂 File best_agent selezionato: {agent_path}")

agent.q_network.load_state_dict(torch.load(agent_path, map_location=device))



#agent.q_network.load_state_dict(torch.load("checkpoints/best_agent.pth"))
agent.q_network.to(device)  # 👈 Sposta la rete su MPS o CUDA

print(agent.q_network.fc1.weight[0][:5])  # ad esempio

# ➕ Inizializza il classificatore e assegna all'agente
embedding_dim = next(iter(dataloader))[0].shape[1]
classifier_path = os.path.join(checkpoint_dir, "classifier.pth")
classifier = AgeGroupClassifier(input_dim=embedding_dim).to(device)
classifier.load_state_dict(torch.load(classifier_path, map_location=device))
classifier.eval()

agent.classifier = classifier
# 🔬 Ora puoi usarlo per valutare
results = agent.evaluate(model=None, dataloader=dataloader, device=device)
print(results)

true_labels = results["true_labels"]
predicted_labels = results["predicted_labels"]
# Calcola le classi realmente presenti
all_labels = sorted(set(true_labels) | set(predicted_labels))  # unione insiemistica
target_names = [f"{i*10}s" for i in all_labels]
# 📋 Report
print(classification_report(true_labels, predicted_labels, labels=all_labels, target_names=target_names, zero_division=0))
# 📊 Classification Report
report_dict = classification_report(true_labels, predicted_labels, labels=all_labels, target_names=target_names, output_dict=True, zero_division=0)
report_table = pd.DataFrame(report_dict).transpose()

# 🖼️ Stampa in tabella
fig, ax = plt.subplots(figsize=(12, 5))
ax.axis('off')
table = ax.table(cellText=report_table.round(2).values,
                 colLabels=report_table.columns,
                 rowLabels=report_table.index,
                 loc='center',
                 cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)
plt.title("📊 Classification Report - RL Agent", pad=20)
plt.tight_layout()
plt.show()

# 🔲 Confusion Matrix
cm = confusion_matrix(true_labels, predicted_labels, labels=all_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title(f"Confusion Matrix - {dataset_name}")
plt.tight_layout()
plt.show()

# 🔢 Accuracy
acc = accuracy_score(true_labels, predicted_labels)
print(f"\n🎯 Accuracy: {acc:.3f}")
