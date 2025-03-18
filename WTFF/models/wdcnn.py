import torch.nn as nn

class WDCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(WDCNN, self).__init__()
        self.wd = nn.Sequential(nn.Conv1d(1, 16, kernel_size=64, stride=16, padding=24),
                                nn.BatchNorm1d(16),
                                nn.ReLU(inplace=True),
                                nn.MaxPool1d(2, stride=2),
                                nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm1d(32),
                                nn.ReLU(inplace=True),
                                nn.MaxPool1d(2, stride=2),
                                nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm1d(64),
                                nn.ReLU(inplace=True),
                                nn.MaxPool1d(2, stride=2),
                                nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm1d(64),
                                nn.MaxPool1d(2, stride=2),
                                nn.ReLU(inplace=True),
                                nn.Conv1d(64, 64, kernel_size=3, stride=1),
                                nn.BatchNorm1d(64),
                                nn.ReLU(inplace=True),
                                nn.MaxPool1d(2, stride=2),
                                nn.Flatten(),
                                nn.Linear(192, 100),
                                nn.BatchNorm1d(100),
                                nn.ReLU(inplace=True)
                                )
        self.fc = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.wd(x)
        x = self.fc(x)
        return x
