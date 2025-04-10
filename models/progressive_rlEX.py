import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class QNetwork(nn.Module):

    """
    Q-Network for reinforcement learning.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ProgressiveRLAgent:

    """
    Progressive Reinforcement Learning Agent for age estimation.
    """

    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99):
        self.q_network = QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.loss_fn = nn.MSELoss()

    def select_action(self, state, epsilon):

        
        if np.random.rand() < epsilon:
            return np.random.choice(action_dim)
        else:
            with torch.no_grad():
                state = state.to(self.device)  # 👈 Fix: assicura che il tensore sia sullo stesso device del modello
                q_values = self.q_network(state)
                #q_values = self.q_network(state)
                return torch.argmax(q_values).item()

    def update(self, state, action, reward, next_state, done):
       
        q_values = self.q_network(state)
        next_q_values = self.q_network(next_state)
        q_value = q_values[0, action]
        next_q_value = torch.max(next_q_values)
        
        if done:
            target = reward
        else:
            target = reward + self.gamma * next_q_value
        
        loss = self.loss_fn(q_value, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def calculate_reward(self, predicted_age, actual_age, age_group):
  
        # Reward function based on the formula provided in the original paper
        distance = abs(predicted_age - actual_age)
        imbalance_ratio = 1 / (age_group + 1)  # Example imbalance ratio
        reward = -distance * imbalance_ratio
        return reward

if __name__ == "__main__":
    # Assume a simple state and action space for age estimation
    state_dim = 10  # Example state dimension (e.g., features from LRA-GNN)
    action_dim = 5  # Example action dimension (e.g., age groups)
    learning_rate = 0.001
    gamma = 0.99
    epsilon = 0.1

    # Initialize the Progressive RL Agent
    agent = ProgressiveRLAgent(state_dim, action_dim, learning_rate, gamma)

    # Example state and next state (randomly generated for demonstration)
    state = torch.randn(1, state_dim)
    next_state = torch.randn(1, state_dim)
    action = agent.select_action(state, epsilon)
    predicted_age = action * 10  # Example mapping of action to predicted age
    actual_age = 35  # Example actual age
    age_group = 3  # Example age group
    reward = agent.calculate_reward(predicted_age, actual_age, age_group)
    done = False  # Example done flag

    # Update the Q-Network
    agent.update(state, action, reward, next_state, done)

    print("Action selected:", action)
    print("Reward:", reward)