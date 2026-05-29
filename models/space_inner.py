# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.11

import torch
from torch import Tensor
from torch.nn import functional as F


def feat_inner(
    features: Tensor,  # [B, N, C]
    vectors: Tensor,  # [B, M, C]
    *,
    basis: bool = True,
) -> Tensor:
    if vectors.dim() == 1:
        vectors = vectors.unsqueeze(dim=0)
        scales = torch.inner(features, vectors) / vectors.square().sum(dim=1)
        return scales @ vectors
    else:
        vector_basis = (
            vectors if basis else torch.linalg.svd(vectors, full_matrices=False)[2]
        )
        return (features @ vector_basis.T) @ vector_basis
    