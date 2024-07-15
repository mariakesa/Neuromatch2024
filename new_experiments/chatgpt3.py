import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import namedtuple, deque
from environment import DelaySampleToMatchEnv

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

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)


class Agent:
    def __init__(self, state_size, action_size, hidden_size, capacity, batch_size, lr, gamma):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.memory = ReplayMemory(capacity)
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = 1.0
        self.eps_end = 0.01
        self.eps_decay = 0.995
        self.epsilon = self.eps_start

        self.q_network = RNNQNetwork(state_size, hidden_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.q_network.to(self.device)

    def select_action(self, state, hidden):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(
            0).unsqueeze(0).to(self.device)
        if random.random() > self.epsilon:
            with torch.no_grad():
                q_values, hidden = self.q_network(state, hidden)
                action = q_values.max(2)[1].item()
        else:
            action = random.randrange(self.action_size)
        return action, hidden

    def store_transition(self, state, action, reward, next_state, hidden):
        self.memory.push(state, action, reward,
                         next_state, hidden.cpu().numpy())

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        losses = []

        for i in range(self.batch_size):
            print('boom', batch.state[i])
            state = torch.tensor(np.array(batch.state[i]), dtype=torch.float32).unsqueeze(
                0).unsqueeze(0).to(self.device)
            action = torch.tensor(batch.action[i], dtype=torch.int64).unsqueeze(
                0).unsqueeze(0).to(self.device)
            reward = torch.tensor(
                batch.reward[i], dtype=torch.float32).unsqueeze(0).to(self.device)
            next_state = torch.tensor(np.array(batch.next_state[i]), dtype=torch.float32).unsqueeze(
                0).unsqueeze(0).to(self.device)
            hidden = torch.tensor(
                batch.hidden[i], dtype=torch.float32).unsqueeze(0).to(self.device).squeeze(0)
            print(next_state.shape)

            state_action_values, hidden = self.q_network(state, hidden)
            state_action_values = state_action_values.gather(
                2, action.unsqueeze(-1)).squeeze(-1)

            next_state_values, _ = self.q_network(next_state, hidden)
            next_state_values = next_state_values.max(2)[0].detach()

            expected_state_action_values = reward + \
                (self.gamma * next_state_values)

            loss = self.criterion(state_action_values,
                                  expected_state_action_values.unsqueeze(1))
            losses.append(loss)

        total_loss = torch.stack(losses).mean()
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Update epsilon
        if self.epsilon > self.eps_end:
            self.epsilon *= self.eps_decay


# Example usage
state_size = 6
action_size = 6
hidden_size = 5
capacity = 10000
batch_size = 64
lr = 0.001
gamma = 0.99


def one_hot_encode(state, state_space):
    one_hot = np.zeros(state_space)
    if isinstance(state, tuple):
        # Extract the state from the tuple if reset() returns a tuple
        state = state[0]
    one_hot[int(state)] = 1  # Ensure state is converted to integer
    return one_hot


env = DelaySampleToMatchEnv()
agent = Agent(state_size, action_size, hidden_size,
              capacity, batch_size, lr, gamma)


n_episodes = 10000
win_pct_list = []
scores = []

# Training loop
for i in range(n_episodes):
    state = env.reset()  # Reset the environment
    state = one_hot_encode(state, env.observation_space.n)
    done = False
    score = 0
    hidden = agent.q_network.init_hidden(1).to(agent.device)
    while not done:
        action, hidden = agent.select_action(state, hidden)
        next_state, reward, done, info = env.step(action)  # Take the action
        next_state = one_hot_encode(next_state, env.observation_space.n)
        agent.store_transition(state, action, reward, next_state, hidden)
        agent.optimize_model()  # Update Q-network
        state = next_state  # Move to the next state
        score += reward
    scores.append(score)

    if i % 100 == 0:
        avg_score = np.mean(scores[-100:])
        print(f"Episode {i} - Average Score: {avg_score:.2f}")

print
