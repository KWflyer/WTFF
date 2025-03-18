# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

import models


class SimCLR_Model(nn.Module):
    def __init__(self, args):
        '''
        SimCLR model.

        Adapted for use in personal Boilerplate for unsupervised/self-supervised contrastive learning.

        Additionally, too inspiration from: https://github.com/HobbitLong/CMC.

        Args:
            init:
                args (dict): Program arguments/commandline arguments.

                temperature (float): Temperature used in the InfoNCE / NT_Xent contrastive losses. (default: 0.07)

            forward:
                x_q (Tensor): Reprentation of view intended for the query_encoder.

                x_k (Tensor): Reprentation of view intended for the key_encoder.

        returns:

            logit (Tensor): Positve and negative logits computed as by InfoNCE loss. (bsz, queue_size + 1)

            label (Tensor): Labels of the positve and negative logits to be used in softmax cross entropy. (bsz, 1)

        '''
        super(SimCLR_Model, self).__init__()

        self.args = args

        # Load model
        self.encoder = getattr(models, args.backbone)()  # Query Encoder

        # Add the mlp head
        args.in_channels = self.encoder.fc.in_features
        self.encoder.fc = models.mlphead(args)

    def InfoNCE_logits(self, f):
        '''
        Compute the similarity logits between positive
         samples and positve to all negatives in the memory.

        args:
            f_q (Tensor): Feature reprentations of the view x_q computed by the query_encoder.

            f_k (Tensor): Feature reprentations of the view x_k computed by the key_encoder.

        returns:
            logit (Tensor): Positve and negative logits computed as by InfoNCE loss. (bsz, queue_size + 1)

            label (Tensor): Labels of the positve and negative logits to be used in softmax cross entropy. (bsz, 1)
        '''

        # Normalize the feature representations
        f = nn.functional.normalize(f, dim=1)

        sim = torch.matmul(f, f.T)

        labels = torch.cat([torch.arange(sim.shape[0] // 2) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        # Compute sim between postive and all negatives in the memory
        mask = torch.eye(sim.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(sim.shape[0], -1)
        sim = sim[~mask].view(sim.shape[0], -1)

        # select and combine multiple positives
        pos = sim[labels.bool()].view(sim.shape[0], -1)

        # select only the negatives
        neg = sim[~labels.bool()].view(sim.shape[0], -1)

        logits = torch.cat((pos, neg), dim=1)

        logits /= self.args.temperature

        # Create labels, first logit is postive, all others are negative
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        return logits, labels

    def forward(self, x_q, x_k):
        
        x = torch.cat((x_q, x_k), dim=0)
        # Feature representations of the query view from the query encoder
        features = self.encoder(x)

        # Compute the logits for the InfoNCE contrastive loss.
        logit, label = self.InfoNCE_logits(features)

        return logit, label
