# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.11
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MILossInfoNCE(nn.Module):
    def __init__(self, in_dim, proj_dim=128, temperature=0.05):
        super().__init__()
        self.temperature = temperature

        self.target_proj = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )
        self.learn_proj = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )

    def mean_pool(self, x, mask=None):
        if mask is None:
            return x.mean(dim=1)

        mask = rearrange(mask.squeeze(-1), "b num M -> b (num M)")
        mask = mask.unsqueeze(-1)  # [B, L, 1]
        x = x * mask
        denom = mask.sum(dim=1).clamp(min=1e-6)  # [B, 1]
        return x.sum(dim=1) / denom

    def forward(self, target_fea, learn_fea, target_mask=None, learn_mask=None):
        target_vec = self.mean_pool(target_fea, target_mask)
        learn_vec = self.mean_pool(learn_fea, learn_mask)
        target_vec = self.target_proj(target_vec)
        learn_vec = self.learn_proj(learn_vec)
        target_vec = F.normalize(target_vec, dim=-1)
        learn_vec = F.normalize(learn_vec, dim=-1)
        logits = torch.matmul(target_vec, learn_vec.transpose(0, 1))
        logits = logits / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_t2l = F.cross_entropy(logits, labels)
        loss_l2t = F.cross_entropy(logits.transpose(0, 1), labels)

        loss = 0.5 * (loss_t2l + loss_l2t)
        return loss
