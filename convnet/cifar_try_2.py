import sys
import os

# Get the current working directory
current_dir = os.getcwd()

# Add the current directory to sys.path
if current_dir not in sys.path:
    sys.path.append(current_dir)
print(current_dir)

from mmodel import MyDQN, CNNFeatureExtractor, RNNMemory
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import namedtuple, deque
from cifar_environment import DelaySampleToMatchEnv
import torch.nn.functional as F
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# Define a transform to convert the CIFAR-10 images to tensors
transform = transforms.Compose(
    [transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])

# Download and load the CIFAR-10 training dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

# Convert the dataset to a numpy array for easy indexing
train_data = np.array([trainset[i][0].numpy() for i in range(len(trainset))])
train_labels = np.array([trainset[i][1] for i in range(len(trainset))])

class_dct = {}
for i in range(1, 11):
    class_dct[i] = np.where(train_labels == i - 1)[0]

black_image = torch.tensor(
    np.zeros((1024,), dtype=np.float32)).to(device='cuda')

# Define the Transition namedtuple
Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward', 'hidden', 'next_hidden', 'done'))


class Agent:
    def __init__(self, action_size, device):
        self.device = device
        self.cnn = CNNFeatureExtractor().to(device)
        self.rnn = RNNMemory(
            input_size=512, hidden_size=256, num_layers=1).to(device)
        self.q_network = MyDQN(
            cnn_feature_size=512, hidden_size=256, action_size=action_size).to(device)
        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(), lr=0.001)
        self.memory = []

    def select_action(self, state, hidden):
        with torch.no_grad():
            cnn_features = self.cnn(state)
            rnn_output, hidden = self.rnn(cnn_features.unsqueeze(0), hidden)
            action_values = self.q_network(cnn_features, rnn_output[:, -1, :])
            action = action_values.argmax().item()
        return action, hidden

    def store_transition(self, state, action, next_state, reward, hidden, next_hidden, done):
        self.memory.append(
            (state, action, next_state, reward, hidden, next_hidden, done))


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
        next_q_values, _ = self.q_network(
            next_states_tensor, next_hidden_tensor)
        next_q_values = next_q_values.max(2).values

    rewards_tensor = torch.tensor(
        batch.reward, dtype=torch.float32).to(self.device)
    actions_tensor = torch.tensor(
        batch.action, dtype=torch.int64).to(self.device)
    dones_tensor = torch.tensor(
        batch.done, dtype=torch.float32).to(self.device)

    # Compute current Q-values for all states in the batch
    current_q_values, _ = self.q_network(states_tensor, hidden_tensor)

    # Remove the leading dimension from current_q_values
    current_q_values = current_q_values.squeeze(0)  # Shape [64, 6]

    # Expand dimensions of actions_tensor to match the shape required for gather
    actions_tensor = actions_tensor.unsqueeze(1)  # Shape [64, 1]

    # Gather the q_values corresponding to the actions
    current_q_values = current_q_values.gather(
        1, actions_tensor)

    # Compute target Q-values using the Bellman equation
    target_q_values = rewards_tensor + self.gamma * \
        next_q_values * (1 - dones_tensor)

    current_q_values = current_q_values.squeeze(1)  # Shape [64]
    target_q_values = target_q_values.squeeze(0)
    # print(target_q_values.shape, current_q_values.shape)
    # Compute the loss
    loss = self.criterion(current_q_values, target_q_values)

    # Optimize the model
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    if self.epsilon > self.eps_end:
        self.epsilon *= self.eps_decay


# Example usage
state_size = 256
action_size = 2
hidden_size = 128
capacity = 100000
batch_size = 64
lr = 0.001
gamma = 0.99
env = DelaySampleToMatchEnv()
agent = Agent(action_size, device='cuda')


n_episodes = 4000
win_pct_list = []
scores = []

# Training loop
for i in range(n_episodes):
    state = env.reset()  # Reset the environment
    state_ = int(state)
    if state_ != 0:
        indices = class_dct[int(state)]
        random_index = np.random.choice(indices)
        state = torch.tensor(
            train_data[random_index].flatten()).to(device=agent.device)
    else:
        state = black_image
    done = False
    score = 0
    hidden = agent.q_network.init_hidden(1).to(agent.device)
    counter = 0
    while not done:
        if state_ == 0:
            state = black_image
        action, next_hidden = agent.select_action(state, hidden)
        next_state, reward, done, info = env.step(action)  # Take the action
        next_state_ = int(next_state)
        if counter == 0:
            first_stimulus = next_state
            # print(first_stimulus)
            indices = class_dct[int(next_state)]
            random_index = np.random.choice(indices)
            memory_state = torch.tensor(
                train_data[random_index].flatten()).to(device=agent.device)
        if first_stimulus != next_state:
            if next_state_ == 0:
                next_state = torch.tensor(black_image).to(device=agent.device)
            else:
                indices = class_dct[int(next_state)]
                random_index = np.random.choice(indices)
                next_state = torch.tensor(
                    train_data[random_index].flatten()).to(device=agent.device)
        else:
            next_state = memory_state
        counter += 1
        # ('state', 'action', 'next_state', 'reward', 'hidden', 'next_hidden', 'done'))
        agent.store_transition(state, action, next_state,
                               reward, hidden, next_hidden, done)
        agent.learn()  # Update Q-network
        hidden = next_hidden
        state_ = next_state_
        state = next_state  # Move to the next state
        score += reward
    scores.append(score)

    if i % 100 == 0:
        avg_score = np.mean(scores[-100:])
        print(f"Episode {i} - Average Score: {avg_score:.2f}")

torch.save({
    'model_state_dict': agent.q_network.state_dict(),
    'optimizer_state_dict': agent.optimizer.state_dict(),
}, 'rnn_q_network_variation.pth')
