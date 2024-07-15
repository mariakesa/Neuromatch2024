import torch.nn as nn
from torch.functional import F
import torch


class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # Adjust dimensions based on your image size
        self.fc1 = nn.Linear(64 * 8 * 8, 512)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)  # Flatten feature map
        x = F.relu(self.fc1(x))
        return x


class RNNMemory(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNNMemory, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True)

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        return out, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))


class MyDQN(nn.Module):
    def __init__(self, cnn_feature_size, hidden_size, action_size):
        super(MyDQN, self).__init__()
        self.fc1 = nn.Linear(cnn_feature_size + hidden_size, 256)
        self.fc2 = nn.Linear(256, action_size)

    def forward(self, cnn_features, rnn_hidden):
        x = torch.cat((cnn_features, rnn_hidden), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
