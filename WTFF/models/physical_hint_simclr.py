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

        # generate mask for sim
        mask_labels = torch.cat([torch.arange(sim.shape[0] // 2, dtype=torch.int64) for i in range(2)], dim=0).to(self.args.device)
        self.pos_mask = (mask_labels.unsqueeze(0) == mask_labels.unsqueeze(1))
        labels_mask = torch.eye(self.pos_mask.shape[0], dtype=torch.bool).to(self.args.device)
        self.pos_mask = self.pos_mask[~labels_mask].view(self.pos_mask.shape[0], -1)

        # if label was generated by random
        if self.args.random_pseudo_label:
            labels = torch.randint(1, 12, (sim.shape[0] // 2,)) * torch.randint(0, 2, (sim.shape[0] // 2,))

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

        if self.args.loss == "mcc":
            if self.args.select_positive == 'combine_inst_dis':
                mask += self.pos_mask
            if self.args.select_positive == 'only_inst_dis':
                mask = self.pos_mask
            loss = self.multilabel_categorical_crossentropy(sim, mask.float())
            return loss, None
        elif self.args.loss == "mpc":
            pos, neg = self.multipositive_crossentropy(sim, mask)

            logits = torch.cat((pos, neg), dim=1)

            logits /= self.args.temperature

            # Create labels, first logit is postive, all others are negative
            labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

            return logits, labels
        elif self.args.loss in ["mse", 'bce']:
            mask += self.pos_mask
            return sim, mask.float()
        else:
            raise 'No such loss function!'
    
    def multipositive_crossentropy(self, sim, mask):
        num_pos = torch.sum(mask, dim=1)
        num_pos[num_pos == 0] = 1
        if self.args.select_positive == 'cluster_positive':
            no_pos_mask = (self.pos_mask + mask) ^ mask
            sim[no_pos_mask] = 1
        mask = self.pos_mask + mask
        if self.args.select_positive == 'only_inst_dis':
            mask = self.pos_mask
        pos = sim[mask].view(-1, 1)
        neg = sim * (~mask)
        neg = torch.repeat_interleave(sim, num_pos, 0)
        return pos, neg

    def multilabel_categorical_crossentropy(self, y_pred, y_true):
        y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
        y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
        y_pred_pos = y_pred - (1 - y_true) * 1e12 # mask the pred outputs of neg classes
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        return (neg_loss + pos_loss).mean()

    def forward(self, x_q, x_k, labels):
        
        x = torch.cat((x_q, x_k), dim=0)
        # Feature representations of the query view from the query encoder
        features = self.encoder(x)

        # Compute the logits for the InfoNCE contrastive loss.
        logit, label = self.InfoNCE_logits(features, labels)

        return logit, label
