import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import os
from utils.save_embeddings import save_embeddings  # 👈 importa funzione
from models.lra_gnn import LRA_GNN
from models.progressive_rl import ProgressiveRLAgent
from training.train_model import train_model
from training.train_rl import train_prlae
from dataset.age_estimation_dataset import AgeEstimationDataset
from dataset.embedding_dataset import EmbeddingDataset


if __name__ == "__main__":
    # 🔹 Dispositivo
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # 🔹 Dataset
    dataset = AgeEstimationDataset(root_dir="datasets/data/MORPH/")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # 🔹 DataLoader
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 🔹 Modello LRA-GNN
    model = LRA_GNN(
        num_layers=12,
        num_heads=8,
        in_channels=512,
        hidden_channels=128,
        out_channels=1,
        num_steps=5
    ).to(device)

    # 🔹 Ottimizzatore e criterio
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # 🔹 Training LRA-GNN
    num_epochs = 50
    save_path = "checkpoints/best_lra_gnn.pth"
    os.makedirs("checkpoints", exist_ok=True)

    train_loss, val_loss, val_mae, cs5, eps = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=num_epochs,
        save_path=save_path,
        scheduler=scheduler,
        early_stopping_patience=10,
        show_plot=True
    )

    print("\n✅ Training LRA-GNN completato. Ora inizio fase RL...")

    # 🔍 Visualizza gli embedding del training set
    print("\n🎯 Stampa embedding del training set:")
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            embeddings = model(batch, return_features=True)
            for i, emb in enumerate(embeddings):
                print(f"[Batch {batch_idx} - Sample {i}] Embedding: {emb.tolist()}")


    # 🔹 Salva gli embedding del training set
    save_embeddings(model, train_loader, device, save_dir="embeddings/train")

    # (opzionale) Salva anche gli embedding del validation set
    save_embeddings(model, val_loader, device, save_dir="embeddings/val")

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

