import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

# Deep Q-Learning Agent with RNN


class DQNAgent:
    def __init__(self, env, hidden_dim=64, learning_rate=0.001, discount_factor=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, buffer_size=10000, batch_size=64, Network=RNNQNetwork):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim

        self.memory = deque(maxlen=buffer_size)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = Network(
            env.observation_space.n, hidden_dim, env.action_space.n).to(self.device)
        self.optimizer = optim.Adam(
            self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        def choose_action(self, state, hidden):
            state_tensor = torch.tensor(
                state, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values, hidden = self.q_network(state_tensor, hidden)
            if np.random.rand() < self.epsilon:
                return self.env.action_space.sample(), hidden  # Explore
            else:
                return torch.argmax(q_values).item(), hidden  # Exploit

        def store_transition(self, state, action, reward, next_state, done):
            self.memory.append((state, action, reward, next_state, done))

        def learn(self):
            if len(self.memory) < self.batch_size:
                return

            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            # Convert lists to numpy arrays and concatenate
            states = np.array(states, dtype=np.float32)
            actions = np.array(actions, dtype=np.int64)
            rewards = np.array(rewards, dtype=np.float32)
            next_states = np.array(next_states, dtype=np.float32)
            dones = np.array(dones, dtype=np.float32)

            # Convert concatenated numpy arrays to PyTorch tensors
            states_tensor = torch.tensor(states).to(self.device)
            actions_tensor = torch.tensor(
                actions).unsqueeze(-1).to(self.device)
            rewards_tensor = torch.tensor(
                rewards).unsqueeze(-1).to(self.device)
            next_states_tensor = torch.tensor(next_states).to(self.device)
            dones_tensor = torch.tensor(dones).unsqueeze(-1).to(self.device)

            # Initialize the hidden state for the first batch item
            hidden = self.q_network.init_hidden(
                self.batch_size).to(self.device)

            current_q_values, _ = self.q_network(states_tensor, hidden)
            current_q_values = current_q_values.gather(1, actions_tensor)
            next_q_values, _ = self.q_network(next_states_tensor, hidden)
            next_q_values = next_q_values.max(1)[0].unsqueeze(-1)
            target_q_values = rewards_tensor + self.discount_factor * \
                next_q_values * (1 - dones_tensor)

            loss = self.criterion(current_q_values, target_q_values)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.epsilon = max(
                self.epsilon_min, self.epsilon * self.epsilon_decay)
