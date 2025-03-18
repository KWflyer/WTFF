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

    def InfoNCE_logits(self, f_q, f_k):
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
        f_q = nn.functional.normalize(f_q, dim=1)
        f_k = nn.functional.normalize(f_k, dim=1)

        mask = torch.eye(f_q.size(0), dtype=bool)
        # Compute sim between normal data
        pos = torch.mean(torch.mm(f_q, f_q.T)[~mask].view(f_k.size(0),-1), dim=1).unsqueeze(-1)

        # Compute sim between normal and others
        neg = torch.cat((torch.mm(f_k, f_k.T)[~mask].view(f_k.size(0),-1),
                        torch.mm(f_q, f_k.T).view(f_k.size(0),-1)), dim=1)

        logits = torch.cat((pos, neg), dim=1)

        logits /= self.args.temperature

        # Create labels, first logit is postive, all others are negative
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        return logits, labels

    def forward(self, x_k, x_q):
        feat_q = self.encoder(x_q)
        feat_k = self.encoder(x_k)

        # Compute the logits for the InfoNCE contrastive loss.
        logit, label = self.InfoNCE_logits(feat_q, feat_k)

        return logit, label
