import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
from collections import Counter
from training.rl_environment import RLEnvironment
import matplotlib.pyplot as plt
from models.classifier import AgeGroupClassifier



class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(QNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.output = nn.Linear(hidden_dim, action_dim)

        self.apply(self.init_weights)  # inizializza i pesi

    def forward(self, state):
        x = F.leaky_relu(self.norm1(self.fc1(state)))
        x = F.leaky_relu(self.norm2(self.fc2(x)))
        x = F.leaky_relu(self.norm3(self.fc3(x)))
        return self.output(x)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
            nn.init.constant_(m.bias, 0)
    
class PRLAELoss(nn.Module):
    def __init__(self, eta=0.5, tau=1.3):
        super().__init__()
        self.eta = eta
        self.tau = tau
        self.mae = nn.L1Loss()

    def forward(self, pred, target):
        # Assume pred is raw score, apply sigmoid to get probability
        prob = torch.sigmoid(pred)
        focal = -((1 - prob) ** self.tau) * torch.log(prob + 1e-8)
        mae = self.mae(pred, target)
        return self.eta * focal.mean() + (1 - self.eta) * mae

class ProgressiveRLAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.00001, gamma=0.99, tau=1.0, device='cpu',            classifier=None):
        self.q_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network = copy.deepcopy(self.q_network).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.tau = tau
        # Usa PRLAELoss invece di MSE
        # self.loss_fn = PRLAELoss(eta=0.8, tau=4.0)
        #self.loss_fn = nn.MSELoss()
        # self.loss_fn = nn.MSELoss()
        self.loss_fn = nn.SmoothL1Loss()
        # 🔽 Salva il classificatore passato
        self.classifier = classifier.to(device) if classifier is not None else None

        self.action_dim = action_dim
        self.device = device  # 👉 Salva il device usato

    def select_action(self, state, epsilon):
        print(f"👉 select_action chiamato con epsilon={epsilon:.4f}")
        if np.random.rand() < epsilon:
            action = np.random.choice(self.action_dim)
            print(f"🎲 Azione casuale scelta (ε-greedy): {action}")
            return np.random.choice(self.action_dim)
        else:
            with torch.no_grad():
                q_values = self.q_network(state)
                # 🔍 Stampa Q-values per debugging
                print(f"   🔢 Q-values: {q_values.cpu().numpy().flatten()}")
                action = torch.argmax(q_values).item()
                print(f"✅ Azione greedy scelta: {action}")
                
                return torch.argmax(q_values).item()

    
    def update(self, state, action, reward, next_state, done):
        state = state.float()
        next_state = next_state.float()

        # Copia dei pesi prima dell'update
        #params_after = list(self.q_network.parameters())
        #params_before = copy.deepcopy(list(self.q_network.parameters()))
        q_values = self.q_network(state)
        print(f"📊 Q-values aggiornati: {q_values.detach().cpu().numpy().flatten()}")
        q_value = q_values[0, action]
        with torch.no_grad():
            # Double DQN: seleziona azione migliore con q_network
            next_q_values_online = self.q_network(next_state)
            best_action = torch.argmax(next_q_values_online).item()

                          # Q(s, a)
            # Valore di quella azione dalla target_network
            #next_q_values_target = self.target_network(next_state)
            next_q_values_target = self.target_network(next_state.to(self.q_network.fc1.weight.device))

            next_q_value = next_q_values_target[0, best_action]

            target = reward if done else reward + self.gamma * next_q_value
            target = torch.tensor([target], dtype=torch.float32, device=state.device)

        loss = self.loss_fn(q_value.unsqueeze(0), target)  # matcha shape: [1]
            
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # 👇 Esegui aggiornamento morbido della rete target
        self.soft_update_target_network()

        return loss

    def soft_update_target_network(self):
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save(self, path="checkpoints/rl_agent_best.pth"):
        torch.save(self.q_network.state_dict(), path)
    
    def load(self, path):
        self.q_network.load_state_dict(torch.load(path, map_location=self.device))
        self.q_network.eval()
        self.target_network.load_state_dict(self.q_network.state_dict())
        print(f"📥 Modello caricato da: {path}")


    def evaluate(self, model, dataloader, device):
        print("🧪 evaluate() DEFINITIVA in uso ✅")
        print("\n🧪 Inizio evaluate: nuovo round di valutazione\n")

        if model is not None:
            model.eval()
        #model.eval()
        env = RLEnvironment()
        # 🔢 Estrai tutte le età
        #all_ys = torch.cat([batch.y for batch in dataloader])
        all_ys = torch.cat([torch.tensor([age], dtype=torch.float32) for _, age in dataloader])

        age_groups = [int(y.item()) // 10 for y in all_ys]
        counts = Counter(age_groups)

        # 📊 Salva nel tuo ambiente RL
        env.group_counts = dict(counts)
        env.majority_class_count = max(counts.values())

        total_reward = 0
        total_correct = 0
        total_samples = 0
        only_row_correct = 0 
        total_correct_decade = 0
        accuracy_list = []
        decade_accuracy_list = []

        with torch.no_grad():

            #for batch_idx, batch in enumerate(dataloader):
            #    print(f"📦 Batch {batch_idx + 1}")

             #   batch = batch.to(device)
             #   x = model(batch, return_features=True)
             #   print(f"📦 Batch size corrente: {x.size(0)}")

              #  y_true = batch.y.float()
            for batch_idx, (embedding, age) in enumerate(dataloader):
                embedding = embedding.to(device)
                x = embedding  # è già l'embedding, non serve passarla a model
                y_true = age.to(device).float()

                for i in range(x.size(0)):
                    
                    

                    xi = x[i].unsqueeze(0)
                    yi = y_true[i]
                    gi = int(yi.item()) // 10
                    ei = int(yi.item()) % 10

                    Ngi = env.group_counts[gi]
                    NgM = env.majority_class_count
                    imbalance_ratio = min(NgM / max(Ngi, 1), 1e5)

                    # Classificatore per stimare la riga (gruppo di età)
                    with torch.no_grad():
                        group_logits = self.classifier(xi)  # MLP che predice la decade
                        age_group_pred = torch.argmax(group_logits).item()

                    #env.reset(x=xi, target_row=gi, target_col=ei, imbalance_ratio=imbalance_ratio)
                    env.reset(x=xi, target_row=gi, target_col=ei, imbalance_ratio=imbalance_ratio, actual_age=yi.item(), start_row=age_group_pred)


                    reached_target = False

                    for step in range(30):  # o max_steps
                        #pos_encoding = torch.tensor([[env.r / 10, env.c / 10]], device=device)
                        #target_encoding = torch.tensor([[gi / 10, ei / 10]], device=device)
                        target_tensor = torch.tensor([[gi / 10, ei / 10]], device=xi.device)
                        delta_x = torch.tensor([[(gi - env.r) / 10.0]], device=device)
                        delta_y = torch.tensor([[(ei - env.c) / 10.0]], device=device)
                        pos_tensor = torch.tensor([[env.r / 10.0, env.c / 10.0]], device=device)

                        state_tensor = torch.cat([xi, delta_x, delta_y, pos_tensor, target_tensor], dim=1)



                        #state_tensor = torch.cat([xi, pos_encoding, target_encoding], dim=1)
                        # Salva la posizione precedente
                        prev_r, prev_c = env.r, env.c
                        action = self.select_action(state_tensor, epsilon=0.0)
                        _, reward, done = env.step(action)

                         #📍 Reward intermedio: distanza manhattan
                        dist_prev = abs(prev_r - gi) + abs(prev_c - ei)
                        dist_now = abs(env.r - gi) + abs(env.c - ei)
                        delta_dist = dist_prev - dist_now
                        reward += delta_dist * 0.5 

                        if env.r == gi and env.c == ei:
                            reward += 20.0
                            reached_target = True
                            break  # ✅ uscita anticipata se il target è raggiunto
                    
                    predicted_decade = env.r
                    actual_decade = gi

                    if predicted_decade == actual_decade:
                       total_correct_decade += 1
                       decade_accuracy_list.append(1)
                    else:
                       decade_accuracy_list.append(0)

                    if reached_target:
                       total_correct += 1
                       accuracy_list.append(1)
                       desc = "✅ CORRETTO"
                    else:
                       accuracy_list.append(0)
                       desc = "❌ ERRATO"
                       print(f"❌ Predicted: ({env.r}, {env.c}) | GT: ({gi}, {ei})")
                    # Manca questa riga importante nel loop interno
                    total_samples += 1

                        # 🔹 Nuova metrica: riga corretta, colonna errata
                    if env.r == gi and env.c != ei:
                        only_row_correct += 1
                        print(f"🔵 Solo RIGA corretta: pred ({env.r}, {env.c}) vs target ({gi}, {ei})")

             # 📊 RISULTATI FINALI
            print(f"🔢 Numero totale di sample valutati: {total_samples}")
            print(f"✅ Target esatti raggiunti: {total_correct}")
            print(f"🔹 Solo RIGA corretta (decade): {only_row_correct}")
            print(f"📊 RL Evaluation → Accuracy Finale: {(100.0 * total_correct / total_samples):.2f}%, Avg Reward: {total_reward / total_samples:.4f}")
            print(f"📊 RL Evaluation → Accuracy sulla decade: {(100.0 * total_correct_decade / total_samples):.2f}%")
            print(f"📊 RL Evaluation → Accuracy SOLO decade (riga corretta, colonna errata): {(100.0 * only_row_correct / total_samples):.2f}%")

            # 📈 Accuracy per sample
            plt.figure(figsize=(10, 4))
            plt.bar(range(len(accuracy_list)), accuracy_list,
                    color=['green' if a == 1 else 'red' for a in accuracy_list])
            plt.title("Accuratezza per campione")
            plt.xlabel("Campione")
            plt.ylabel("Esito (1 = corretto, 0 = errato)")
            plt.grid(True, axis='y')
            plt.tight_layout()
            plt.xticks(range(len(accuracy_list)), [f"#{i+1}" for i in range(len(accuracy_list))], rotation=45)
            plt.show()

            # 📈 Accuracy per decade
            plt.figure(figsize=(10, 4))
            plt.bar(range(len(decade_accuracy_list)), decade_accuracy_list,
                    color=['blue' if a == 1 else 'orange' for a in decade_accuracy_list])
            plt.title("Accuratezza per decade (0 = errato, 1 = corretto)")
            plt.xlabel("Campione")
            plt.ylabel("Esito Decade")
            plt.grid(True, axis='y')
            plt.tight_layout()
            plt.xticks(range(len(decade_accuracy_list)), [f"#{i+1}" for i in range(len(decade_accuracy_list))], rotation=45)
            plt.show()

            return {
                "accuracy": 100.0 * total_correct / total_samples,
                "decade_accuracy": 100.0 * total_correct_decade / total_samples,
                "only_row_accuracy": 100.0 * only_row_correct / total_samples,
                "avg_reward": total_reward / total_samples,
                "accuracy_list": accuracy_list,
                "decade_accuracy_list": decade_accuracy_list
            }