import torch
import torch.nn.functional as F
from tqdm import tqdm
from training.rl_environment import RLEnvironment
from models.progressive_rl import ProgressiveRLAgent
import matplotlib.pyplot as plt
from collections import defaultdict
from collections import Counter
from datetime import datetime
from models.classifier import AgeGroupClassifier
import random
from models.regressor import AgeRegressor
from utils.metrics import compute_mae_rmse
import numpy as np
import os



# Funzione per disegnare la griglia e la traiettoria
def plot_grid_trajectory(trajectory, target, title="Traiettoria"):
    grid_size = 10
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    ax.grid(True)

    xs = [pos[1] for pos in trajectory]
    ys = [pos[0] for pos in trajectory]

    ax.plot(xs, ys, marker='o', linestyle='-', color='blue', label="Percorso")
    ax.scatter(target[1], target[0], color='red', marker='X', s=100, label="Target")
    ax.scatter(xs[0], ys[0], color='green', marker='s', s=100, label="Start")
    ax.invert_yaxis()
    ax.set_title(title)
    ax.legend()
    plt.show()

def focal_loss(probs, targets, gamma=1.3):
    ce_loss = -torch.log(probs.gather(1, targets.view(-1, 1)))
    focal = (1 - probs.gather(1, targets.view(-1, 1))) ** gamma * ce_loss
    return focal.mean()

def mae_loss(pred, target):
    return torch.mean(torch.abs(pred - target))

def prlae_loss(pred_probs, pred_ages, target_class, target_age, eta=0.4, gamma=1.3):
    fl = focal_loss(pred_probs, target_class, gamma=gamma)
    mae = torch.abs(pred_ages - target_age).mean()
    return eta * fl + (1 - eta) * mae

def train_prlae(agent, dataloader, device, dataset_name="MORPH", num_episodes=200):
    env = RLEnvironment()

    # 📦 Classificatore per stimare il gruppo iniziale (decade)

    embedding_dim = next(iter(dataloader))[0].shape[1]
    # Classificatore
    classifier_path = os.path.join("checkpoints", dataset_name, "classifier.pth")

    classifier = AgeGroupClassifier(input_dim=embedding_dim).to(device)
    classifier.load_state_dict(torch.load(classifier_path))

    classifier.eval()

    # 👉 assegna al tuo agente
    agent.classifier = classifier
    # 🔢 Regressore continuo
    regressor = AgeRegressor(input_dim=embedding_dim).to(device)
    regressor_optimizer = torch.optim.Adam(regressor.parameters(), lr=1e-3)
    regressor.train()

    # 🔢 Estrai tutte le età dai dati (i secondi elementi delle tuple)
    all_ys = torch.tensor([age.item() if isinstance(age, torch.Tensor) else age for _, age in dataloader], dtype=torch.float)

    age_groups = [int(y) // 10 for y in all_ys]
    counts = Counter(age_groups)

    env.group_counts = dict(counts)
    env.majority_class_count = max(counts.values())

    print("\n📊 Conteggio per gruppi di età:")
    for k in sorted(env.group_counts):
        print(f"  Gruppo {k}: {env.group_counts[k]} campioni")
    print(f"📌 Majority class count = {env.majority_class_count}")

    start_row = 0
    all_episode_accuracies = []
    all_decade_accuracies = []
    episode_losses = []
    agent.q_network.train()
     # ⬇️ Aggiungi questa inizializzazione
    best_accuracy = 0.0
    predicted_ages_list = []
    real_ages_list = []
    all_real_ages = []         # 👈 Aggiungi queste due righe
    all_predicted_ages = []    # 👈


    for episode in range(num_episodes):
        #epsilon = max(0.1, 1.0 - episode / num_episodes)
        #epsilon = max(0.1, 1.0 - episode / (num_episodes * 2))
        #epsilon = max(0.05, 1.0 - episode / num_episodes)
        #epsilon = max(0.05, 0.95 ** episode)
        #epsilon = max(0.05, 0.99 ** episode)
        epsilon = max(0.05, 0.98**episode)
        #epsilon = max(0.05, 1.0 / (1.0 + 0.02 * episode))


        #epsilon = 0.1
        print(f"[EPISODIO {episode+1}] Epsilon: {epsilon:.4f}")

        total_reward = 0
        step_losses = []

        # Aggiunto: conta i target centrati
        correct_predictions = 0
        total_samples = 0
        only_row_correct = 0
        agent.q_network.train()

        # 🔀 Ricrea un nuovo dataloader con shuffle abilitato
        shuffled_loader = torch.utils.data.DataLoader(
            dataloader.dataset,  # usa lo stesso dataset
            batch_size=1,
            shuffle=True
        )

        #for embedding, age in tqdm(dataloader, desc=f"Episode {episode + 1}/{num_episodes}"):
        for embedding, age in tqdm(shuffled_loader, desc=f"Episode {episode + 1}/{num_episodes}"):

            xi = embedding.to(device).detach()     # già [1, dim]
            yi = age.to(device).view(-1)           # [1]
            # 🔍 Stima del gruppo con il classificatore (se presente)
            # Età target: decade e unità
            gi = int(yi.item()) // 10
            ei = int(yi.item()) % 10

            # 🔍 Stima del gruppo di partenza con classificatore
            if agent.classifier is not None:
                with torch.no_grad():
                    group_logits = agent.classifier(xi)
                    group_probs = F.softmax(group_logits, dim=1)
                    age_group_pred = torch.argmax(group_probs, dim=1).item()
            else:
                age_group_pred = 9  # fallback se non esiste

            # Imposta riga di partenza (stimata dal classificatore)
            start_row = age_group_pred
            start_col = 5  # facoltativo: puoi anche stimarlo o fissarlo

            

            Ngi = env.group_counts.get(gi, 1)
            NgM = env.majority_class_count
            imbalance_ratio = min(NgM / max(Ngi, 1), 1e5)

            print(f"📊 Imbalance ratio per gruppo {gi}: {imbalance_ratio:.2f}")
            print(f"[🎯 Target: {yi.item():.1f}] Posizione desiderata: ({gi}, {ei})")

            #env.reset(x=xi, target_row=gi, target_col=ei, imbalance_ratio=imbalance_ratio, actual_age=yi.item())
            env.reset(
                x=xi,
                target_row=gi,
                target_col=ei,
                imbalance_ratio=imbalance_ratio,
                actual_age=yi.item(),
                start_row=age_group_pred  # 👈 usa la decade stimata
            )

            trajectory = [(env.r, env.c)]
            done = False
            total_samples += 1
            

            for step in range(30):
                   
                # Coordinate normalizzate
                delta_x = torch.tensor([[(gi - env.r) / 10]], device=xi.device)
                delta_y = torch.tensor([[(ei - env.c) / 10]], device=xi.device)

                # Stato finale = embedding + delta_x + delta_y
                #state_tensor = torch.cat([xi, delta_x, delta_y], dim=1)
                pos_tensor = torch.tensor([[env.r / 10, env.c / 10]], device=xi.device)
                target_tensor = torch.tensor([[gi / 10, ei / 10]], device=xi.device)
                state_tensor = torch.cat([xi, delta_x, delta_y, pos_tensor,target_tensor], dim=1)

                action = agent.select_action(state_tensor, epsilon=epsilon)

                
                _, reward, done = env.step(action)

                trajectory.append((env.r, env.c))
                
                next_delta_x = torch.tensor([[(gi - env.r) / 10]], device=xi.device)
                next_delta_y = torch.tensor([[(ei - env.c) / 10]], device=xi.device)
                #next_state_tensor = torch.cat([xi, next_delta_x, next_delta_y], dim=1)
                next_pos_tensor = torch.tensor([[env.r / 10, env.c / 10]], device=xi.device)
                next_target_tensor = torch.tensor([[gi / 10, ei / 10]], device=xi.device)
                next_state_tensor = torch.cat([xi, next_delta_x, next_delta_y, next_pos_tensor, next_target_tensor], dim=1)

                # next_state_tensor = torch.cat([xi, next_pos_tensor, target_tensor], dim=1)
                
                done = (env.r == gi and env.c == ei)
                if done:
                    reward += 10.0

                loss = agent.update(state_tensor, action, reward, next_state_tensor, done)
                step_losses.append(loss.item())  # loss deve essere restituita da agent.update

                total_reward += reward

                action_meaning = ["up", "down", "left", "right", "stay"]
                print(f"Step {step+1:02d} → Action: {action} ({action_meaning[action]}), Pos: ({env.r}, {env.c}), Target: ({gi}, {ei}), Reward: {reward:.2f}")
                
                
                
                if done:
                    correct_predictions += 1
                    # 🎯 Regressione continua
                    y_pred = regressor(xi)
                    age_loss = F.l1_loss(y_pred.view(-1), yi)
                    regressor_optimizer.zero_grad()
                    age_loss.backward()
                    regressor_optimizer.step()
                    all_real_ages.append(yi.item())
                    all_predicted_ages.append(y_pred.item())
                    predicted_ages_list.append(y_pred.item())
                    real_ages_list.append(yi.item())

                    print(f"🎯 Regressione età: predetta={y_pred.item():.2f}, reale={yi.item():.2f}")
                    break


                
            if env.r == gi and env.c != ei:
                only_row_correct += 1
                print(f"🔵 Solo decade corretta (riga {env.r}, colonna {env.c} vs {gi}, {ei})")
    
            #plot_grid_trajectory(
            #        trajectory,
            #        target=(gi, ei),
            #        title=f"🎯 Età target: {yi.item():.1f} | Episodio {episode + 1}"
            #)
        episode_accuracy = 100.0 * correct_predictions / total_samples if total_samples > 0 else 0.0
        decade_accuracy = 100.0 * only_row_correct / total_samples if total_samples > 0 else 0.0
        all_episode_accuracies.append(episode_accuracy)
        all_decade_accuracies.append(decade_accuracy)
         # 🔽 Se migliora, salvalo
        if episode_accuracy > best_accuracy:
            best_accuracy = episode_accuracy
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            
            checkpoint_dir = os.path.join("checkpoints", dataset_name)
            os.makedirs(checkpoint_dir, exist_ok=True)
            save_path = os.path.join(checkpoint_dir, f"best_agent_{timestamp}.pth")

            
            agent.save(save_path)
            print(f"💾 Miglior modello salvato (Episode {episode+1}, Accuracy: {episode_accuracy:.2f}%)")
        # ALLA FINE DI OGNI EPISODIO
        mean_loss = sum(step_losses) / len(step_losses) if step_losses else 0
        episode_losses.append(mean_loss)
        print(f"📉 Episode {episode+1}: Mean Loss = {mean_loss:.4f}")
        #input("⏸️ Premi INVIO per continuare con il prossimo episodio...")

    
        print(f"✅ Episode {episode+1}: Accuracy COMPLETA per episodio = {episode_accuracy:.2f}%")
        print(f"🔹 Episode {episode+1}: Accuracy SOLO DECADE per episodio = {decade_accuracy:.2f}%")
        print(f"📦 Totale corretti per episodio : {correct_predictions} su {total_samples}")

        # 🔁 Aggiorna la rete target (Double DQN soft update)
        agent.soft_update_target_network()

    mae, rmse = compute_mae_rmse(all_real_ages, all_predicted_ages)
    print(f"\n📏 MAE (errore assoluto medio): {mae:.2f} anni")
    print(f"📏 RMSE (errore quadratico medio): {rmse:.2f} anni")

    print("\n🏁 Training RL completato!")
    # 📈 Plot finale
    plt.figure(figsize=(8, 4))
    plt.plot(all_episode_accuracies, marker='o')
    plt.title("📈 Accuracy per episodio")
    plt.xlabel("Episodio")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # 📈 Plot finale
    plt.figure(figsize=(8, 4))
    plt.plot(all_decade_accuracies, marker='o')
    plt.title("📈 Accuracy per decade")
    plt.xlabel("Episodio")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(episode_losses, label='Training Loss', color='blue')
    plt.xlabel("Episodio")
    plt.ylabel("Loss")
    plt.title("Andamento della Loss PRLAE")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
        
    # 📊 Plot regressione finale
    if predicted_ages_list:
        plt.figure(figsize=(6,6))
        plt.scatter(real_ages_list, predicted_ages_list, alpha=0.6, color='blue', label='Predizioni')
        plt.plot([min(real_ages_list), max(real_ages_list)], [min(real_ages_list), max(real_ages_list)], 'r--', label='y = x')
        plt.xlabel("Età Reale")
        plt.ylabel("Età Predetta")
        plt.title("📊 Regressione Età: Predetta vs Reale")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        errors = np.abs(np.array(real_ages_list) - np.array(predicted_ages_list))
        plt.figure(figsize=(8,4))
        plt.hist(errors, bins=20, color='purple', alpha=0.7)
        plt.xlabel("Errore Assoluto (|predetta - reale|)")
        plt.ylabel("Frequenza")
        plt.title("📉 Distribuzione degli errori assoluti")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        mae = np.mean(errors)
        rmse = np.sqrt(np.mean((np.array(real_ages_list) - np.array(predicted_ages_list)) ** 2))
        print(f"📏 MAE: {mae:.2f} | RMSE: {rmse:.2f}")
