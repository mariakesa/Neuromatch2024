import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
from networks import TransformerDecoderQNetwork

# Deep Q-Learning Agent with RNN


class DQNAgent:
    def __init__(self, Network, env, hidden_dim=64, learning_rate=0.001, discount_factor=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, buffer_size=10000, batch_size=64):
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
        self.q_network = Network.to(self.device)
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


import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

# Sinusoidal positional encoding function


def get_sinusoidal_positional_encoding(position, d_model):
    angle_rates = 1 / \
        np.power(10000, (2 * (np.arange(d_model) // 2)) / np.float32(d_model))
    angle_rads = position * angle_rates
    sines = np.sin(angle_rads[0::2])
    cosines = np.cos(angle_rads[1::2])
    pos_encoding = np.zeros(d_model)
    pos_encoding[0::2] = sines
    pos_encoding[1::2] = cosines
    return pos_encoding

# Transformer Decoder for approximating Q-values


class TransformerDecoderQNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, max_seq_len, num_heads, num_layers):
        super(TransformerDecoderQNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_seq_len = max_seq_len

        # Embedding layers
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = self.create_positional_encoding(
            max_seq_len, hidden_dim)

        # Transformer decoder layers
        self.decoder_layers = nn.ModuleList(
            [nn.TransformerDecoderLayer(hidden_dim, num_heads)
             for _ in range(num_layers)]
        )

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def create_positional_encoding(self, max_seq_len, d_model):
        pos_enc = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pos_enc[pos, i] = np.sin(pos / (10000 ** ((2 * i) / d_model)))
                pos_enc[pos, i +
                        1] = np.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pos_enc = pos_enc.unsqueeze(0)  # Add batch dimension
        return pos_enc

    def forward(self, x, tgt_mask=None):
        # Add input embeddings
        x = self.input_embedding(
            x) + self.positional_encoding[:, :x.size(1), :].to(x.device)

        # Pass through decoder layers
        for layer in self.decoder_layers:
            x = layer(x, x, tgt_mask=tgt_mask)

        # Output Q-values
        output = self.output_layer(x)
        return output

# Deep Q-Learning Agent with Transformer


class DQNAgent:
    def __init__(self, env, hidden_dim=64, learning_rate=0.001, discount_factor=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, buffer_size=10000, batch_size=64, max_seq_len=100, num_heads=8, num_layers=6):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        self.memory = deque(maxlen=buffer_size)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = TransformerDecoderQNetwork(
            env.observation_space.n, hidden_dim, env.action_space.n, max_seq_len, num_heads, num_layers).to(self.device)
        self.optimizer = optim.Adam(
            self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def choose_action(self, state, position):
        state_tensor = torch.tensor(
            state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor, tgt_mask=None)
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()  # Explore
        else:
            return torch.argmax(q_values).item()  # Exploit

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
        actions_tensor = torch.tensor(actions).unsqueeze(-1).to(self.device)
        rewards_tensor = torch.tensor(rewards).unsqueeze(-1).to(self.device)
        next_states_tensor = torch.tensor(next_states).to(self.device)
        dones_tensor = torch.tensor(dones).unsqueeze(-1).to(self.device)

        # Forward pass for current and next Q-values
        current_q_values = self.q_network(
            states_tensor, tgt_mask=None).gather(1, actions_tensor)
        next_q_values = self.q_network(next_states_tensor, tgt_mask=None).max(1)[
            0].unsqueeze(-1)
        target_q_values = rewards_tensor + self.discount_factor * \
            next_q_values * (1 - dones_tensor)

        loss = self.criterion(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class DQNAgentTransformer:
    def __init__(self, network, env, hidden_dim=64, learning_rate=0.001, discount_factor=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, buffer_size=10000, batch_size=64, max_seq_len=100, num_heads=8, num_layers=6):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        self.memory = deque(maxlen=buffer_size)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = network.to(self.device)
        # self.q_network = TransformerDecoderQNetwork(
        # env.observation_space.n, hidden_dim, env.action_space.n, max_seq_len, num_heads, num_layers).to(self.device)
        self.optimizer = optim.Adam(
            self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def choose_action(self, state, position):
        state_tensor = torch.tensor(
            state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor, tgt_mask=None)
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()  # Explore
        else:
            return torch.argmax(q_values).item()  # Exploit

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
        actions_tensor = torch.tensor(actions).unsqueeze(-1).to(self.device)
        rewards_tensor = torch.tensor(rewards).unsqueeze(-1).to(self.device)
        next_states_tensor = torch.tensor(next_states).to(self.device)
        dones_tensor = torch.tensor(dones).unsqueeze(-1).to(self.device)

        # Forward pass for current and next Q-values
        current_q_values = self.q_network(states_tensor, tgt_mask=None)
        current_q_values = current_q_values.gather(
            1, actions_tensor)  # Gather Q-values for chosen actions

        next_q_values = self.q_network(next_states_tensor, tgt_mask=None)
        # Take maximum Q-value for next states
        next_q_values = next_q_values.max(1)[0].unsqueeze(-1)

        target_q_values = rewards_tensor + self.discount_factor * \
            next_q_values * (1 - dones_tensor)

        # Compute loss and optimize Q-network
        loss = self.criterion(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
