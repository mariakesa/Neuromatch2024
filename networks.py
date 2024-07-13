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


import torch
import torch.nn as nn


class TransformerDecoderQNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, max_seq_len, num_heads, num_layers):
        super(TransformerDecoderQNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_seq_len = max_seq_len

        # Embedding layers
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = self.create_positional_encoding(
            max_seq_len, hidden_dim)

        # Transformer decoder layers
        self.decoder_layers = nn.ModuleList(
            [nn.TransformerDecoderLayer(hidden_dim, num_heads)
             for _ in range(num_layers)]
        )

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def create_positional_encoding(self, max_seq_len, d_model):
        pos_enc = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pos_enc[pos, i] = np.sin(pos / (10000 ** ((2 * i) / d_model)))
                pos_enc[pos, i +
                        1] = np.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pos_enc = pos_enc.unsqueeze(0)  # Add batch dimension
        return pos_enc

    def forward(self, x, tgt_mask=None):
        # Add input embeddings
        x = self.input_embedding(
            x) + self.positional_encoding[:, :x.size(1), :].to(x.device)

        # Pass through decoder layers
        for layer in self.decoder_layers:
            x = layer(x, x, tgt_mask=tgt_mask)

        # Output Q-values
        output = self.output_layer(x)
        return output

# Example instantiation
# input_dim = observation space size
# hidden_dim = model dimension
# output_dim = action space size
# max_seq_len = maximum length of the input sequence
# num_heads = number of attention heads
# num_layers = number of transformer decoder layers


'''
input_dim = 10  # Example value
hidden_dim = 64
output_dim = 4  # Example value
max_seq_len = 100
num_heads = 8
num_layers = 6

q_transformer = TransformerDecoderQNetwork(input_dim, hidden_dim, output_dim, max_seq_len, num_heads, num_layers)
'''
