import torch
import models

class BYOL_Model(torch.nn.Module):
    def __init__(self, args):
        super(BYOL_Model, self).__init__()
        self.model = getattr(models, args.backbone)()
        args.in_channels = self.model.fc.in_features
        self.model.fc = models.mlphead(args=args)

    def forward(self, x):
        return self.model(x)
