import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from utils.save_embeddings import save_embeddings
from models.lra_gnn import LRA_GNN
from training.train_model import train_model
from dataset.age_estimation_dataset import AgeEstimationDataset

# === Parser degli argomenti

# 🔹 Parsing degli argomenti da linea di comando
parser = argparse.ArgumentParser(description="Training LRA-GNN")
parser.add_argument("--dataset", type=str, default="MORPH", help="Nome del dataset: MORPH o FGNET")
args = parser.parse_args()

# === Configurazione dinamica
DATASET_NAME = args.dataset.upper()
root_dir = f"datasets/data/{DATASET_NAME}/"
checkpoint_dir = os.path.join("checkpoints", DATASET_NAME)
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, f"best_lra_gnn_{DATASET_NAME.lower()}.pth")


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # 🔹 Dataset path
    if args.dataset == "MORPH":
        data_path = "datasets/data/MORPH"
    elif args.dataset == "FGNET":
        data_path = "datasets/data/FGNET"
    else:
        raise ValueError(f"Dataset sconosciuto: {args.dataset}")
    # 🔹 Dataset
    #dataset = AgeEstimationDataset(root_dir=root_dir)
    
    dataset = AgeEstimationDataset(root_dir=data_path, dataset_name=args.dataset)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # 🔹 DataLoader
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 🔹 Modello
    model = LRA_GNN(
        num_layers=12,
        num_heads=8,
        in_channels=512,
        hidden_channels=128,
        out_channels=1,
        num_steps=5
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # 🔹 Training
    train_loss, val_loss, val_mae, cs5, eps = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=50,
        save_path=checkpoint_path,
        scheduler=scheduler,
        early_stopping_patience=10,
        show_plot=True
    )

    print(f"\n✅ Training LRA-GNN completato per {DATASET_NAME}!")
    dataset_name = args.dataset.upper()
    # 🔹 Salva embedding
    # 🔹 Determina la cartella di salvataggio in base al dataset
    if dataset_name.upper() == "FGNET":
        train_dir = "embeddings_FGNET/train"
        val_dir = "embeddings_FGNET/val"
    else:
        train_dir = "embeddings/train"
        val_dir = "embeddings/val"

    # 🔹 Salva embedding
    save_embeddings(model, train_loader, device, save_dir=train_dir)
    save_embeddings(model, val_loader, device, save_dir=val_dir)



    # 🔍 Determina dinamicamente la dimensione embedding
    #sample_batch = next(iter(train_loader)).to(device)
    #embedding = model(sample_batch, return_features=True)
    #embedding_dim = embedding.shape[1]  # es. 130
    

    #state_dim = embedding_dim + 4        # embedding + delta_x + delta_y + r + c
    #action_dim = 5                      # su, giù, sinistra, destra, resta
    #print(f"[DEBUG] Embedding shape: {embedding.shape}")  # Deve essere (B, E)
    #print(f"[DEBUG] State dim: {state_dim}")
    


    # 🔹 Inizializza e addestra l'agente RL
    #rl_agent = ProgressiveRLAgent(state_dim=state_dim, action_dim=action_dim)
    #rl_agent.q_network.to(device)
    #rl_agent.target_network.to(device)
  
    # 🔁 Ridefinisci il DataLoader con batch_size=1 SOLO per RL
    #embedding_dataset = EmbeddingDataset("embeddings/train")
    #embedding_loader = DataLoader(embedding_dataset, batch_size=1, shuffle=True)


    #train_prlae(
    #    agent=rl_agent,
    #    dataloader=embedding_loader,
    #    device=device,
    #    num_episodes=80
    #)

    #print("\n🏁 Training RL completato!")

    # 🔹 Salva agente RL
    #rl_agent.save("checkpoints/rl_agent_best.pth")

    # 🔹 Valuta le prestazioni RL
    #rl_agent.evaluate(model, embedding_loader, device)

