from q_networks import RNNQNetwork, LSTMQNetwork, GRUQNetwork
from agents import DQNAgent
import torch.nn as nn
from train import vanilla_train, mnist_train, cifar_train

# 6 different configurations
# Short sequences-- 6 stimuli different stimuli (including blank):
# Easy env
# MNIST
# CIFAR10

# Longish sequences-- 11 different stimuli (including blank):
# Easy env
# MNIST
# CIFAR10

# Each of these categories has experimental variables:
# RNN,LSTM,GRU
# LSTM+MSELoss, LSTM+SmoothL1Loss

# There are all together 6*4=24 models

# Small sequences

experiments = {}

# 1
experiments[('short-seq', 'easy-seq', 'MSELoss', 'RNN')] = {
    'q_network': RNNQNetwork,
    'loss': nn.MSELoss,
    'state_size': 6,
    'action_size': 6,
    'hidden_size': 64,
    'capacity': 100000,
    'batch_size': 32,
    'lr': 0.001,
    'gamma': 0.99,
    'model_path': 'short-seq_easy-seq_MSELoss_RNN.pth',
    'mode': 'train-from-zero',
    'train_function': vanilla_train,
    'train_steps': 50000
}
# 2
experiments[('short-seq', 'easy-seq', 'MSELoss', 'GRU')] = {
    'q_network': GRUQNetwork,
    'loss': nn.MSELoss,
    'state_size': 6,
    'action_size': 6,
    'hidden_size': 64,
    'capacity': 100000,
    'batch_size': 32,
    'lr': 0.001,
    'gamma': 0.99,
    'model_path': 'short-seq_easy-seq_MSELoss_GRU.pth',
    'mode': 'train-from-zero',
    'train_function': vanilla_train,
    'train_steps': 50000
}
# 3
experiments[('short-seq', 'easy-seq', 'MSELoss', 'LSTM')] = {
    'q_network': LSTMQNetwork,
    'loss': nn.MSELoss,
    'state_size': 6,
    'action_size': 6,
    'hidden_size': 64,
    'capacity': 100000,
    'batch_size': 32,
    'lr': 0.001,
    'gamma': 0.99,
    'model_path': 'short-seq_easy-seq_MSELoss_LSTM.pth',
    'mode': 'train-from-zero',
    'train_function': vanilla_train,
    'train_steps': 50000
}
# 4
experiments[('short-seq', 'easy-seq', 'L1SmoothLoss', 'LSTM')] = {
    'q_network': LSTMQNetwork,
    'loss': nn.SmoothL1Loss,
    'state_size': 6,
    'action_size': 6,
    'hidden_size': 64,
    'capacity': 100000,
    'batch_size': 64,
    'lr': 0.001,
    'gamma': 0.99,
    'model_path': 'model.pth',
    'mode': 'train-from-zero',
    'model_path': 'short-seq_easy-seq_SmoothL1Loss_LSTM.pth',
    'train_function': vanilla_train,
    'train_steps': 50000
}
# 5
experiments[('long-seq', 'easy-seq', 'MSELoss', 'RNN')] = {
    'q_network': RNNQNetwork,
    'loss': nn.MSELoss,
    'state_size': 11,
    'action_size': 11,
    'hidden_size': 128,
    'capacity': 100000,
    'batch_size': 32,
    'lr': 0.001,
    'gamma': 0.99,
    'model_path': 'short-seq_easy-seq_MSELoss_RNN.pth',
    'mode': 'train-from-zero',
    'train_function': vanilla_train,
    'train_steps': 50000
}
# 6
experiments[('long-seq', 'easy-seq', 'MSELoss', 'GRU')] = {
    'q_network': GRUQNetwork,
    'loss': nn.MSELoss,
    'state_size': 11,
    'action_size': 11,
    'hidden_size': 128,
    'capacity': 100000,
    'batch_size': 32,
    'lr': 0.001,
    'gamma': 0.99,
    'model_path': 'short-seq_easy-seq_MSELoss_GRU.pth',
    'mode': 'train-from-zero',
    'train_function': vanilla_train,
    'train_steps': 50000
}
# 7
experiments[('short-seq', 'easy-seq', 'MSELoss', 'LSTM')] = {
    'q_network': LSTMQNetwork,
    'loss': nn.MSELoss,
    'state_size': 11,
    'action_size': 11,
    'hidden_size': 128,
    'capacity': 100000,
    'batch_size': 32,
    'lr': 0.001,
    'gamma': 0.99,
    'model_path': 'short-seq_easy-seq_MSELoss_LSTM.pth',
    'mode': 'train-from-zero',
    'train_function': vanilla_train,
    'train_steps': 50000
}
# 8
experiments[('long-seq', 'easy-seq', 'L1SmoothLoss', 'LSTM')] = {
    'q_network': LSTMQNetwork,
    'loss': nn.SmoothL1Loss,
    'state_size': 11,
    'action_size': 11,
    'hidden_size': 128,
    'capacity': 100000,
    'batch_size': 64,
    'lr': 0.001,
    'gamma': 0.99,
    'model_path': 'model.pth',
    'mode': 'train-from-zero',
    'model_path': 'long-seq_easy-seq_SmoothL1Loss_LSTM.pth',
    'train_function': vanilla_train,
    'train_steps': 50000
}
