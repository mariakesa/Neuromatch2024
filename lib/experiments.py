from q_networks import RNNQNetwork, LSTMQNetwork, GRUQNetwork
from agents import DQNAgent
import torch.nn as nn
from train import vanilla_train, mnist_train, cifar_train

# Define configurations and parameters
configurations = [
    ("short-seq", "easy-seq"),
    ("short-seq", "mnist-seq"),
    ("short-seq", "cifar-seq"),
    ("long-seq", "easy-seq"),
    ("long-seq", "mnist-seq"),
    ("long-seq", "cifar-seq")
]

networks = {
    "RNN": RNNQNetwork,
    "LSTM": LSTMQNetwork,
    # "GRU": GRUQNetwork
}

losses = {
    "MSELoss": nn.MSELoss,
    "SmoothL1Loss": nn.SmoothL1Loss
}

state_sizes = {
    "short-seq": 6,
    "long-seq": 11
}

train_functions = {
    "easy-seq": vanilla_train,
    "mnist-seq": mnist_train,
    "cifar-seq": cifar_train
}

# Generate experiments dictionary
experiments = {}

for seq_type, env_type in configurations:
    for loss_name, loss_fn in losses.items():
        for net_name, net_class in networks.items():
            config_key = (seq_type, env_type, loss_name, net_name)
            state_size = state_sizes[seq_type]
            action_size = state_size
            hidden_size = 64 if seq_type == "short-seq" else 128
            model_path = f"{seq_type}_{env_type}_{loss_name}_{net_name}.pth"
            learning_curve_path = f"{seq_type}_{env_type}_{loss_name}_{net_name}.npy"
            train_function = train_functions[env_type]

            experiments[config_key] = {
                'q_network': net_class,
                'loss': loss_fn,
                'state_size': state_size,
                'action_size': action_size,
                'hidden_size': hidden_size,
                'capacity': 100000,
                'batch_size': 32,
                'lr': 0.001,
                'gamma': 0.99,
                'model_path': model_path,
                'learning_curve_path': learning_curve_path,
                'mode': 'train-from-zero',
                'train_function': train_function,
                'train_steps': 50000
            }

# Print to verify
# cntr = 0
# for key, value in experiments.items():
    # print(key, value)
    # print(cntr)
    # cntr += 1

# Run experiments
