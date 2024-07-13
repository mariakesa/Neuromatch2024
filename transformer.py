import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.q = nn.Linear(input_dim, 16, bias=False)
        self.k = nn.Linear(input_dim, 16, bias=False)
        self.value = nn.Linear(input_dim, 16, bias=False)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.value(x)

        att = torch.matmul(q, k.transpose(-2, -1)) * \
            (1.0 / math.sqrt(k.size(-1)))
        mask = torch.tril(torch.ones_like(att))
        att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        values = torch.matmul(att, v)

        return values


class SequenceDecoder(nn.Module):
    def __init__(self, input_size, action_space):
        super(SequenceDecoder, self).__init__()
        self.embedding = nn.Embedding(input_size, 24)
        self.attention = Attention(24)
        # Projection layer for Q-values
        self.output_layer = nn.Linear(16, action_space)

    def forward(self, x):
        x = self.embedding(x)
        x = self.attention(x)
        q_values = self.output_layer(x)  # Assuming batch size 1 for simplicity

        return q_values
