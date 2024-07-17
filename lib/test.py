from q_networks import RNNQNetwork
from agents import DQNAgent
import torch.nn as nn

config_dict = {
    'q_network': RNNQNetwork,
    'loss': nn.MSELoss,
    'state_size': 6,
    'action_size': 6,
    'hidden_size': 64,
    'capacity': 100000,
    'batch_size': 64,
    'lr': 0.001,
    'gamma': 0.99,
    'model_path': 'model.pth',
    'mode': 'train-from-zero'
}

# q_network, loss, state_size, action_size, hidden_size, capacity, batch_size, lr, gamma
# env = DelaySampleToMatchEnv()

#agent = DQNAgent(config_dict)

#agent.load_model(mode='eval')

from experiments import experiments

print(experiments.keys())
