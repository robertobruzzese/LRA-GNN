import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.metrics import calculate_metrics  # Assicurati che metrics.py sia in utils/
import matplotlib.pyplot as plt


def train_model(model, train_loader, val_loader, optimizer, num_epochs, device, save_path,
                criterion, scheduler=None, early_stopping_patience=10, show_plot=True):  # ✅ aggiunto show_plot
    model = model.to(device)
    loss_fn = criterion
    best_val_loss = float('inf')

    # 🔢 Liste per tenere traccia delle metriche
    train_losses = []
    val_losses = []
    val_mae_list = []
    cs5_list = []
    epsilon_list = []
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch = batch.to(device)
            outputs = model(batch)
            loss = loss_fn(outputs.squeeze(), batch.y.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 🔍 Validazione
        model.eval()
        with torch.no_grad():
            val_loss = 0
            all_preds = []
            all_labels = []
            for batch in val_loader:
                batch = batch.to(device)
                preds = model(batch).squeeze()
                loss = loss_fn(preds, batch.y.float())
                val_loss += loss.item()
                all_preds.append(preds)
                all_labels.append(batch.y)

            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            all_preds = torch.cat(all_preds, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            metrics = calculate_metrics(all_preds, all_labels)

            val_mae_list.append(metrics['MAE'])
            cs5_list.append(metrics['CS_5'])
            epsilon_list.append(metrics['Epsilon_Error'])

        print(f"\n📉 Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        print(f"📊 Val MAE = {metrics['MAE']:.2f}, CS@5 = {metrics['CS_5']:.2f}%, ϵ = {metrics['Epsilon_Error']:.4f}")

        if scheduler:
            scheduler.step(avg_val_loss)

        if save_path and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"✅ Model saved to {save_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"⏳ No improvement ({epochs_no_improve}/{early_stopping_patience})")

        if epochs_no_improve >= early_stopping_patience:
            print("🛑 Early stopping triggered!")
            break

    # 📈 Visualizza grafici solo se richiesto
    if show_plot:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(val_mae_list, label='Val MAE')
        plt.plot(cs5_list, label='CS@5')
        plt.plot(epsilon_list, label='Epsilon Error')
        plt.title('Validation Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Metric Value')
        plt.legend()

        plt.tight_layout()
        plt.show()

    return avg_train_loss, avg_val_loss, metrics['MAE'], metrics['CS_5'], metrics['Epsilon_Error']
