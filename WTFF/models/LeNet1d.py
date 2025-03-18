import torch.nn as nn


# -----------------------input size>=32---------------------------------
class LeNet(nn.Module):
    def __init__(self, in_channel=1, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channel, 6, 5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(6, 16, 5),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d((5)),  # adaptive change the outputsize to (16,5)
            nn.Flatten()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5, 30),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(30, 10),
            nn.ReLU()
        )
        self.fc = nn.Linear(10, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc(x)
        return x

