import torch
import torch.nn as nn


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
        return torch.ones(1, 1, self.hidden_size)
