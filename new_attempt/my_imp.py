import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import namedtuple, deque
from environment import DelaySampleToMatchEnv
import torch.nn.functional as F

# Define the Transition namedtuple
Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward', 'hidden', 'next_hidden', 'done'))

# agent.store_transition(state, action, reward, next_state, hidden)

# next_state, reward, done, info


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
        q_values = self.fc(out)
        return q_values, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, 1, self.hidden_size)


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

        self.q_network = RNNQNetwork(state_size, hidden_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.q_network.to(self.device)

    def select_action(self, state, hidden):
        rand = random.random()
        # print(rand)
        if rand > self.epsilon:
            with torch.no_grad():
                q_values, hidden = self.q_network(
                    state.reshape(1, 1, -1), hidden)
                action = q_values.max(2).indices.item()
        else:
            action = random.randrange(self.action_size)
        return action, hidden

    def store_transition(self, state, action, next_state, reward, hidden, next_hidden, done):
        # ('state', 'action', 'next_state', 'reward', 'hidden', 'next_hidden', 'done'))
        self.memory.push(state, action, next_state,
                         reward, hidden, next_hidden, done)

    def learn_(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        for i in range(self.batch_size):
            state = torch.tensor(batch.state[i]).reshape(
                1, 1, -1).to(agent.device)
            hidden = torch.tensor(batch.hidden[i]).to(agent.device)
            next_states_tensor = torch.tensor(batch.next_state[i]).reshape(
                1, 1, -1).to(dtype=torch.float32, device=agent.device)
            rewards_tensor = torch.tensor(batch.reward[i]).to(agent.device)
            actions_tensor = torch.tensor(batch.action[i]).to(agent.device)
            dones_tensor = torch.tensor(batch.done[i]).to(agent.device)
            next_hidden = torch.tensor(batch.next_hidden[i]).to(
                dtype=torch.float32, device=agent.device)

            # state_action_values, hidden = self.q_network(state, hidden)

            current_q_values = self.q_network(
                state, hidden)[0]

            current_q_values = current_q_values.gather(
                2, actions_tensor.unsqueeze(-1)).squeeze(-1)  # Shape (batch_size,
            # print(current_q_values)
            next_q_values = self.q_network(
                next_states_tensor, next_hidden)[0].max(2).values
            # print(next_q_values)
            # actions=q_values.max(2).indices.item()
            if dones_tensor == False:
                dones_tensor = 0
            else:
                dones_tensor = 1
            target_q_values = rewards_tensor + self.gamma * \
                next_q_values * (1 - dones_tensor)

            # print(current_q_values.shape, next_q_values.shape)

            loss = self.criterion(current_q_values, target_q_values)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def learn_(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        for i in range(self.batch_size):
            state = torch.tensor(batch.state[i]).reshape(
                1, 1, -1).to(self.device)
            hidden = torch.tensor(batch.hidden[i]).to(self.device)
            next_states_tensor = torch.tensor(batch.next_state[i]).reshape(
                1, 1, -1).to(dtype=torch.float32, device=self.device)
            rewards_tensor = torch.tensor(batch.reward[i]).to(self.device)
            actions_tensor = torch.tensor(batch.action[i]).to(self.device)
            dones_tensor = torch.tensor(
                batch.done[i], dtype=torch.float32).to(self.device)
            next_hidden = torch.tensor(batch.next_hidden[i]).to(
                dtype=torch.float32, device=self.device)

            # Compute current Q-values and gather the Q-value for the action taken
            current_q_values, _ = self.q_network(state, hidden)
            current_q_values = current_q_values.gather(
                2, actions_tensor.view(1, 1, 1)).squeeze(-1)  # Shape (1, 1)

            # Compute next Q-values and take the max value across the action dimension
            next_q_values, _ = self.q_network(next_states_tensor, next_hidden)
            next_q_values = next_q_values.max(2)[0]  # Shape (1, 1)

            # Compute target Q-values
            target_q_values = rewards_tensor + self.gamma * \
                next_q_values * (1 - dones_tensor)

            # Ensure shapes are consistent for loss computation
            current_q_values = current_q_values.squeeze(0)  # Shape (1)
            target_q_values = target_q_values.squeeze(0)    # Shape (1)

            # Compute the loss
            loss = self.criterion(current_q_values, target_q_values)

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

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
            1, actions_tensor)  # Shape [64, 1]

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


# Example usage
state_size = 6
action_size = 6
hidden_size = 64
capacity = 10000
batch_size = 64
lr = 0.001
gamma = 0.99
env = DelaySampleToMatchEnv()
agent = Agent(state_size, action_size, hidden_size,
              capacity, batch_size, lr, gamma)


n_episodes = 1000
win_pct_list = []
scores = []

# Training loop
for i in range(n_episodes):
    state = env.reset()  # Reset the environment
    state = F.one_hot(torch.tensor(state),
                      env.observation_space.n).to(dtype=torch.float32, device=agent.device)
    done = False
    score = 0
    hidden = agent.q_network.init_hidden(1).to(agent.device)
    while not done:
        action, next_hidden = agent.select_action(state, hidden)
        next_state, reward, done, info = env.step(action)  # Take the action
        next_state = F.one_hot(torch.tensor(next_state),
                               env.observation_space.n).to(dtype=torch.float32, device=agent.device)
        # ('state', 'action', 'next_state', 'reward', 'hidden', 'next_hidden', 'done'))
        agent.store_transition(state, action, next_state,
                               reward, hidden, next_hidden, done)
        agent.learn()  # Update Q-network
        hidden = next_hidden
        state = next_state  # Move to the next state
        score += reward
    scores.append(score)

    if i % 100 == 0:
        avg_score = np.mean(scores[-100:])
        print(f"Episode {i} - Average Score: {avg_score:.2f}")
