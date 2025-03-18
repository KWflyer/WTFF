import torch.nn as nn
import torch.nn.functional as F

class SelfCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SelfCNN, self).__init__()
        self.selfcnn = nn.Sequential(nn.Conv1d(1, 16, kernel_size=12, stride=1, padding=6),
                                nn.BatchNorm1d(16),
                                nn.ReLU(inplace=True),
                                nn.MaxPool1d(2, stride=2),
                                nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm1d(32),
                                nn.ReLU(inplace=True),
                                nn.MaxPool1d(2, stride=2),
                                nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm1d(64),
                                nn.ReLU(inplace=True),
                                nn.MaxPool1d(2, stride=2),
                                nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm1d(64),
                                nn.MaxPool1d(2, stride=2),
                                nn.ReLU(inplace=True),
                                nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm1d(64),
                                nn.ReLU(inplace=True),
                                nn.MaxPool1d(2, stride=2),
                                nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm1d(64),
                                nn.ReLU(inplace=True),
                                nn.MaxPool1d(2, stride=2),
                                nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm1d(64),
                                nn.ReLU(inplace=True),
                                nn.MaxPool1d(2, stride=2),
                                nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm1d(64),
                                nn.ReLU(inplace=True),
                                nn.MaxPool1d(2, stride=2),
                                nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm1d(64),
                                nn.ReLU(inplace=True),
                                nn.MaxPool1d(2, stride=2),
                                nn.AdaptiveAvgPool1d(),
                                nn.Linear(256, num_classes)
                                )

    def forward(self, x):
        x = self.selfcnn(x)
        return x
