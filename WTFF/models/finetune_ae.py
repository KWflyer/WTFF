import torch
from torch import nn


class Finetine_ae(nn.Module):
    def __init__(self, Encoder, Classifier, args):
        super(Finetine_ae, self).__init__()
        self.mode = args.backbone
        self.encoder = Encoder()
        self.classifier = Classifier(num_classes=args.num_classes)
        if args.backbone in ['resnet18_1d', 'resnet50_1d']:
            self.encoder = torch.nn.Sequential(*(list(self.encoder.children())[:-1]))

    def forward(self, x):
        if self.mode in ["rwkdcae", "dcae"]:
            x, _ = self.encoder(x)
            if isinstance(x, list):
                x = x[-1]
        else:
            x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
