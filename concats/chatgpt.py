import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


# Convert the dataset to a numpy array for easy indexing
# Load embeddings and labels
train_data = np.load('/home/maria/Neuromatch2024/convnet/data/embeddings.npy')
train_labels = np.load('/home/maria/Neuromatch2024/convnet/data/labels.npy')

# Adjust train_labels as per the original intent (adding 1)
train_labels = train_labels + 1

# Create a row of zeros with the same number of columns as train_data
zeros_row = np.zeros((1, train_data.shape[1],))
print(train_data.shape, zeros_row.shape)
train_data = torch.tensor(
    np.vstack((zeros_row, train_data)), dtype=torch.float32).to(device='cuda')

# Append the label corresponding to the zeros row
train_labels = np.hstack((0, train_labels))

print("Shape of train_labels_with_zeros:", train_labels.shape)
print("Shape of train_data_with_zeros:", train_data.shape)

# Create a dictionary to store indices of each class
class_dct = {}
for i in range(12):  # Adjusted to iterate from 0 to 10 (inclusive)
    class_dct[i] = np.where(train_labels == i)[0]

# Print example usage of class_dct
print("Indices of class 0:", class_dct[0])

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import namedtuple, deque
from environment import DelaySampleToMatchEnv
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Define the Transition namedtuple
Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward', 'hidden', 'next_hidden', 'done'))


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
        self.rnn = nn.RNN(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        return out, hidden

    def init_hidden(self, batch_size):
        return torch.ones(1, batch_size, self.hidden_size)


class MyDQN(nn.Module):
    def __init__(self, cnn_feature_size, hidden_size, action_size):
        super(MyDQN, self).__init__()
        self.fc1 = nn.Linear(cnn_feature_size + hidden_size, 128)
        self.fc2 = nn.Linear(128, action_size)

    def forward(self, cnn_features, rnn_hidden):
        x = torch.cat((cnn_features, rnn_hidden), dim=-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Agent:
    def __init__(self, state_size, action_size, hidden_size, capacity, batch_size, lr, gamma):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.memory = ReplayMemory(capacity)
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = 0.99
        self.eps_end = 0.01
        self.eps_decay = 0.995
        self.epsilon = self.eps_start

        self.q_network = RNNQNetwork(
            state_size, hidden_size, action_size).to(device='cuda')
        self.dqn_network = MyDQN(cnn_feature_size=10,
                                 hidden_size=hidden_size, action_size=action_size).to(device='cuda')
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.q_network.to(self.device)
        self.dqn_network.to(self.device)

    def select_action(self, state, hidden):
        rand = random.random()
        if rand > self.epsilon:
            with torch.no_grad():
                state = state.unsqueeze(0).unsqueeze(
                    0)  # Shape [1, 1, state_size]
                rnn_out, hidden = self.q_network(state, hidden)
                # print(rnn_out.shape)
                # rnn_out = rnn_out.squeeze(0).squeeze(0)  # Shape [hidden_size]
                q_values = self.dqn_network(state, rnn_out)  # .unsqueeze(0))
                # print(q_values.shape)
                action = q_values.max(2).indices.item()
        else:
            action = random.randrange(self.action_size)
        return action, hidden

    def store_transition(self, state, action, next_state, reward, hidden, next_hidden, done):
        self.memory.push(state, action, next_state,
                         reward, hidden, next_hidden, done)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)

        batch = Transition(*zip(*transitions))

        states_tensor = torch.stack(
            [s.clone().detach().requires_grad_(True) for s in batch.state]).unsqueeze(0).to(self.device)

        hidden_tensor = torch.stack(
            [h.clone().detach().requires_grad_(True).squeeze(0).squeeze(0) for h in batch.hidden]).unsqueeze(0).to(self.device)

        next_states_tensor = torch.stack(
            [s.clone().detach().requires_grad_(True) for s in batch.next_state]).unsqueeze(0).to(self.device)

        next_hidden_tensor = torch.stack(
            [h.clone().detach().requires_grad_(True).squeeze(0).squeeze(0) for h in batch.next_hidden]).unsqueeze(0).to(self.device)

        with torch.no_grad():
            next_rnn_out, _ = self.q_network(
                next_states_tensor, next_hidden_tensor)
            next_rnn_out = next_rnn_out.squeeze(
                0).unsqueeze(0)  # Shape [hidden_size]
            next_q_values = self.dqn_network(
                next_states_tensor, next_rnn_out)
            # print(next_q_values.shape)
            next_q_values = next_q_values.max(2).values

        rewards_tensor = torch.tensor(
            batch.reward, dtype=torch.float32).to(self.device)
        actions_tensor = torch.tensor(
            batch.action, dtype=torch.int64).to(self.device)
        dones_tensor = torch.tensor(
            batch.done, dtype=torch.float32).to(self.device)

        rnn_out, _ = self.q_network(states_tensor, hidden_tensor)
        # print(rnn_out.shape)
        # print(states_tensor.shape)
        # rnn_out = rnn_out.squeeze(0).squeeze(0)  # Shape [hidden_size]
        # print(states_tensor.shape, rnn_out.shape)
        current_q_values = self.dqn_network(
            states_tensor, rnn_out).squeeze(0)
        # print(current_q_values.shape)
        actions_tensor = actions_tensor.unsqueeze(1)  # Shape [batch_size, 1]
        # print(actions_tensor.shape)
        current_q_values = current_q_values.gather(
            1, actions_tensor).squeeze(0)
        # print(current_q_values.shape)
        # print('mynext', next_q_values.shape)
        # print(rewards_tensor.shape)
        target_q_values = rewards_tensor + self.gamma * \
            next_q_values * (1 - dones_tensor)

        # print(current_q_values.shape)
        current_q_values = current_q_values.squeeze(1)  # Shape [64]
        target_q_values = target_q_values.squeeze(0)

        loss = self.criterion(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.eps_end:
            self.epsilon *= self.eps_decay


# Example usage
state_size = 10
action_size = 6
hidden_size = 64
capacity = 100000
batch_size = 32
lr = 0.001
gamma = 0.99
env = DelaySampleToMatchEnv()
agent = Agent(state_size, action_size, hidden_size,
              capacity, batch_size, lr, gamma)

n_episodes = 4000
win_pct_list = []
scores = []

for i in range(n_episodes):
    done = False
    state = env.reset()
    indices = class_dct[int(state)]
    random_index = np.random.choice(indices)
    state = train_data[random_index].flatten()
    score = 0
    hidden = agent.q_network.init_hidden(1).to(agent.device)
    while not done:
        action, next_hidden = agent.select_action(state, hidden)
        next_state, reward, done, info = env.step(action)
        indices = class_dct[int(next_state)]
        random_index = np.random.choice(indices)
        next_state = train_data[random_index].flatten()
        agent.store_transition(state, action, next_state,
                               reward, hidden, next_hidden, done)
        agent.learn()
        hidden = next_hidden
        state = next_state
        score += reward
    scores.append(score)

    if i % 100 == 0:
        avg_score = np.mean(scores[-100:])
        print(f"Episode {i}, Average score: {avg_score}")

torch.save({
    'q_network_state_dict': agent.q_network.state_dict(),
    'dqn_network_state_dict': agent.dqn_network.state_dict(),
    'optimizer_state_dict': agent.optimizer.state_dict(),
}, 'rnn_dqn_network.pth')
