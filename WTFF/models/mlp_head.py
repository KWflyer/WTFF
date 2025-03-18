from torch import nn


class MLPHead(nn.Module):
    def __init__(self, args):
        super(MLPHead, self).__init__()
        if args.in_channels == 1:
            self.net = nn.Sequential(
                nn.Linear(args.in_channels, args.mlp_hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(args.mlp_hidden_size, args.projection_size)
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(args.in_channels, args.mlp_hidden_size),
                nn.BatchNorm1d(args.mlp_hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(args.mlp_hidden_size, args.projection_size)
            )

    def forward(self, x):
        return self.net(x)


class MLPHeadV2(nn.Module):
    def __init__(self, args):
        super(MLPHeadV2, self).__init__()
        if args.in_channels == 1:
            self.net = nn.Sequential(
                nn.Linear(args.in_channels, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, args.projection_size)
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(args.in_channels, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Linear(128, args.projection_size)
            )

    def forward(self, x):
        return self.net(x)
