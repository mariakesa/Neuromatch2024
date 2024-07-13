import torch
import torch.nn as nn

# RNN for approximating Q-values


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

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_dim)


class RNNQNetworkEmbeddings(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, max_seq_len):
        super(RNNQNetworkEmbeddings, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        # Define the positional embeddings
        self.positional_embeddings = nn.Embedding(max_seq_len, input_dim)

        self.i2h = nn.Linear(input_dim, hidden_dim)
        self.h2h = nn.Linear(hidden_dim, hidden_dim)
        self.h2o = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden, position):
        # Add positional embeddings to the input
        pos_embedding = self.positional_embeddings(position)
        x = x + pos_embedding

        hidden = torch.tanh(self.i2h(x) + self.h2h(hidden))
        output = self.h2o(hidden)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_dim)
