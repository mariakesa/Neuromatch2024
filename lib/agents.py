import random
from collections import namedtuple
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import optim

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


class DQNAgent(nn.Module):
    def __init__(self, config_dict):
        super(DQNAgent, self).__init__()
        self.save_path = config_dict['model_path']
        self.load_path = config_dict['model_path']
        # possible modes: 'train-from-zero', 'train', 'eval'
        if config_dict['mode'] == 'train-from-zero':
            self.q_network = config_dict['q_network'](
                config_dict['state_size'], config_dict['hidden_size'], config_dict['action_size'])
        else:
            self.q_network = self.load_model(
                mode=config_dict['mode'])
        self.memory = config_dict['loss'](config_dict['capacity'])
        self.batch_size = config_dict.get('batch_size', 64)
        self.gamma = config_dict.get('gamma', 0.99)
        self.eps_start = config_dict.get('eps_start', 0.99)
        self.eps_end = config_dict.get('eps_start', 0.01)
        self.eps_decay = config_dict.get('eps_decay', 0.995)
        self.epsilon = self.eps_start
        self.lr = config_dict.get('lr', 0.001)

        self.optimizer = config_dict.get('opt', optim.Adam)(
            self.q_network.parameters(), lr=self.lr)
        self.criterion = config_dict['loss']()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.q_network.to(self.device)

    def load_model(self, mode):
        checkpoint = torch.load(self.load_path)
        self.q_network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if mode == 'eval':
            self.q_network.eval()
        else:
            self.q_network()

    def save_model(self):
        torch.save({
            'model_state_dict': self.q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, self.save_path)

    def select_action(self, state, hidden):
        rand = random.random()
        with torch.no_grad():
            q_values, hidden = self.q_network(
                state.reshape(1, 1, -1), hidden)
        if rand > self.epsilon:
            action = q_values.max(2).indices.item()
        else:
            action = random.randrange(self.action_size)
        return action, hidden

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
        loss = self.criterion(current_q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.eps_end:
            self.epsilon *= self.eps_decay
