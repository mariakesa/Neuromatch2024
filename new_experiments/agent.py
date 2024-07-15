import torch.nn as nn
import torch
from environment import DelaySampleToMatchEnv
from collections import namedtuple, deque
import random
import numpy as np
import math

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'hidden'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class RNNQNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RNNQNetwork, self).__init__()
        self.hidden_dim = hidden_dim

        self.i2h = nn.Linear(input_dim, hidden_dim)
        self.h2h = nn.Linear(hidden_dim, hidden_dim)
        self.h2o = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden):
        hidden = torch.tanh(self.i2h(x) + self.h2h(hidden))
        output = self.h2o(hidden)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_dim)


BATCH_SIZE = 1
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4


class Agent:
    def __init__(self, network, env):
        self.env = env
        self.steps_done = 0
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.memory = ReplayMemory(1)
        self.q_network = network.to(self.device)

    def select_action(self, state, hidden):
        # global steps_done
        print(state, hidden)
        hidden = torch.tensor(hidden).to(self.device)
        state = torch.tensor(state).to(self.device)
        sample = random.random()
        eps_threshold = EPS_END + \
            (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        n_state, hidden = self.q_network(state, hidden)
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.

                return state[0].max(1).indices.view(1, 1), hidden
        else:
            return torch.tensor([[env.action_space.sample()]], device=self.device, dtype=torch.long), hidden

    def store_transition(self, state, action, reward, next_state, hidden):
        transition = Transition(state, action, next_state, reward, hidden)
        self.memory.push(transition)

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

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
        hidden_tensor = torch.tensor(hidden).to(self.device)

        # hidden = self.q_network.init_hidden(self.batch_size).to(self.device)

        # current_q_values, _ = self.q_network(
        # states_tensor, hidden_tensor).gather(1, actions_tensor)
        current_q_values = current_q_values.gather(
            2, actions_tensor.unsqueeze(-1)).squeeze(-1)  # Shape (batch_size, 1)
        current_q_values = current_q_values.gather(1, actions_tensor)
        next_q_values, _ = self.q_network(next_states_tensor, hidden_tensor)
        next_q_values = next_q_values.max(1)[0].unsqueeze(-1)
        target_q_values = rewards_tensor + self.discount_factor * \
            next_q_values * (1 - dones_tensor)

        loss = self.criterion(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# Function to one-hot encode state


def one_hot_encode(state, state_space):
    one_hot = np.zeros(state_space)
    if isinstance(state, tuple):
        # Extract the state from the tuple if reset() returns a tuple
        state = state[0]
    one_hot[int(state)] = 1  # Ensure state is converted to integer
    return one_hot


env = DelaySampleToMatchEnv()
agent = Agent(RNNQNetwork(env.observation_space.n,
                          128, env.action_space.n), env)

n_episodes = 10000
win_pct_list = []
scores = []

# Training loop
for i in range(n_episodes):
    state = agent.env.reset()  # Reset the environment
    state = one_hot_encode(state, env.observation_space.n)
    done = False
    score = 0
    hidden = agent.q_network.init_hidden(1).to(agent.device)
    while not done:
        # hidden= agent.q_network.init_hidden(1).to(agent.device)
        # Choose action based on epsilon-greedy policy
        action, hidden = agent.select_action(state, hidden)
        next_state, reward, done, info = agent.env.step(
            action)  # Take the action
        # print(next_state, reward, done, info)
        next_state = one_hot_encode(next_state, env.observation_space.n)
        agent.store_transition(state, action, reward, next_state, done, hidden)
        agent.optimize_model()  # Update Q-network
        state = next_state  # Move to the next state
        score += reward
