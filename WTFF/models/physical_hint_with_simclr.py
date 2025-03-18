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

    def InfoNCE_logits(self, f, labels):
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

        labels = labels.repeat(2).to(self.args.device)
        mask = (labels.unsqueeze(0) == labels.unsqueeze(1))
        # percent physical hint contrast
        labels2 = labels.clone()
        labels2[labels2 == 0] = -1
        mask = (labels.unsqueeze(0) == labels2.unsqueeze(1))
        
        
        # Compute sim between postive and all negatives in the memory
        labels_mask = torch.eye(sim.shape[0], dtype=torch.bool).to(self.args.device)
        mask = mask[~labels_mask].view(sim.shape[0], -1)
        sim = sim[~labels_mask].view(sim.shape[0], -1)

        # select and combine multiple positives
        num_pos = torch.sum(mask, dim=1)
        num_pos[num_pos == 0] = 1
        pos = (torch.sum(sim * mask, dim=1) / num_pos).view(-1, 1)
        pos[pos == 0] = 1

        # select only the negatives
        num_neg = torch.sum(~mask, dim=1)
        num_neg[num_neg == 0] = 1
        neg = (torch.sum(sim * (~mask), dim=1) / num_neg).view(-1, 1)

        logits = torch.cat((pos, neg), dim=1)

        logits /= self.args.temperature

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(2)], dim=0).to(self.args.device)
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1))
        pos_mask = pos_mask[~labels_mask].view(sim.shape[0], -1)

        # select and combine multiple positives
        pos2 = sim[pos_mask.bool()].view(sim.shape[0], -1)

        # select only the negatives
        neg2 = sim[~pos_mask.bool()].view(sim.shape[0], -1)

        logits2 = torch.cat((pos2, neg2), dim=1)

        logits2 /= self.args.temperature

        # Create labels, first logit is postive, all others are negative
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        return logits, logits2, labels

    def forward(self, x_q, x_k, labels):
        
        x = torch.cat((x_q, x_k), dim=0)
        # Feature representations of the query view from the query encoder
        features = self.encoder(x)

        # Compute the logits for the InfoNCE contrastive loss.
        logit, logits2, label = self.InfoNCE_logits(features, labels)

        return logit, logits2, label
