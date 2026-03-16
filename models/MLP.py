import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, dropout=0.2):
        super().__init__()
        self.func1 = nn.Linear(input_size, hidden_size1)
        self.relu = nn.ReLU()
        self.func2 = nn.Linear(hidden_size1, hidden_size2)
        self.func3 = nn.Linear(hidden_size2, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.func1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.func2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.func3(out)
        return out
