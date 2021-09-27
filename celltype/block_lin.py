import torch
import torch.nn as nn


class BlockLinearLayer(nn.Module):
    def __init__(self, input_size, block_size, output_size):
        super().__init__()
        self.n_blocks = input_size // block_size
        self.block_size = block_size

        assert output_size % self.n_blocks == 0

        self.conv1d = nn.Conv1d(self.n_blocks, output_size, block_size, groups=self.n_blocks, bias=True)

    def forward(self, x):
        assert x.size(1) == self.n_blocks * self.block_size
        x = x.view(x.size(0), self.n_blocks, -1)
        emb = self.conv1d(x)
        return emb.squeeze(-1)


class BlockMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = BlockLinearLayer(128, 4, 128)
        self.linear1b = BlockLinearLayer(128, 4, 128)
        self.linear2 = BlockLinearLayer(128, 8, 64)
        self.linear3 = BlockLinearLayer(64, 16, 64)
        self.linear4 = BlockLinearLayer(64, 16, 32)
        self.linear5 = BlockLinearLayer(32, 16, 32)
        self.linear6 = nn.Linear(32, 32)
        self.linear7 = nn.Linear(32, 32)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = x + torch.relu(self.linear1b(x))
        x = torch.relu(self.linear2(x))
        x = x + torch.relu(self.linear3(x))
        x = torch.relu(self.linear4(x))
        x = x + torch.relu(self.linear5(x))
        x = x + torch.relu(self.linear6(x))
        x = torch.relu(self.linear7(x))
        return x
