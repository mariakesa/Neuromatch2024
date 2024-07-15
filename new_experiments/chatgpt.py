import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import namedtuple, deque

# Define the Transition namedtuple
Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward', 'hidden'))


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class RNNQNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNQNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        q_values = self.fc(out)
        return q_values, hidden


class Agent:
    def __init__(self, state_size, action_size, hidden_size, capacity, batch_size, lr, gamma):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.memory = ReplayMemory(capacity)
        self.batch_size = batch_size
        self.gamma = gamma

        self.q_network = RNNQNetwork(state_size, hidden_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.q_network.to(self.device)

    def store_transition(self, state, action, reward, next_state, hidden):
        self.memory.push(state, action, reward, next_state, hidden)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        states = torch.tensor(np.array(batch.state),
                              dtype=torch.float32).to(self.device)
        actions = torch.tensor(
            batch.action, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(
            batch.reward, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(
            np.array(batch.next_state), dtype=torch.float32).to(self.device)
        hidden = torch.tensor(np.array(batch.hidden),
                              dtype=torch.float32).to(self.device)

        state_action_values, _ = self.q_network(states, hidden)
        state_action_values = state_action_values.gather(1, actions)

        next_state_values, _ = self.q_network(next_states, hidden)
        next_state_values = next_state_values.max(1)[0].detach()

        expected_state_action_values = rewards + \
            (self.gamma * next_state_values)

        loss = self.criterion(state_action_values,
                              expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# Example usage
state_size = 3
action_size = 2
hidden_size = 5
capacity = 10000
batch_size = 64
lr = 0.001
gamma = 0.99

agent = Agent(state_size, action_size, hidden_size,
              capacity, batch_size, lr, gamma)
state = [1, 2, 3]
action = 0
reward = 1.0
next_state = [4, 5, 6]
hidden = [0.1, 0.2, 0.3, 0.4, 0.5]

agent.store_transition(state, action, reward, next_state, hidden)
agent.optimize_model()
