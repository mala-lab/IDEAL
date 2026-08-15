# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.11
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from functools import partial
import numpy as np
from torch import Tensor
from torch.nn.init import trunc_normal_
from einops import rearrange
from models.attention import (
    CrossAttentionLayer,
    PositionEmbeddingSine,
)
from utils.loss import FocalLoss, DiceLoss
from models.vision_layer import Mlp


class IDEAL(nn.Module):
    def __init__(self, args):
        super(IDEAL, self).__init__()
        self.args = args
        self.backbone_name = args.backbone_name
        self.vision_encoder = torch.hub.load(
            "facebookresearch/dinov2", args.backbone_name
        )
        self.hidden_dim = 384  # Vit-S/14=384 & Vit-B=768
        self.d_scale = 14
        self.vision_encoder.eval()
        self.mask_downsample = nn.Conv2d(
            1, 1, kernel_size=self.d_scale, stride=self.d_scale, padding=1, bias=False
        )
        nn.init.constant_(self.mask_downsample.weight, 1.0)

        self.nheads = args.nheads
        self.pre_norm = False  # pre norm
        self.learnable_proxies = nn.Embedding(
            args.num_learnable_vectors, self.hidden_dim
        )
        self.cro_attn = CrossAttentionLayer(
            d_model=self.hidden_dim,
            nhead=self.nheads,
            dropout=0.1,
            normalize_before=self.pre_norm,
        )
        self.ffn_layer = Mlp(
            in_features=self.hidden_dim,
            hidden_features=int(self.hidden_dim * 4),
            act_layer=nn.GELU,  # nn.ReLU? nn.GELU?
            drop=0.0,
        )
        norm_layer = partial(nn.LayerNorm, eps=1e-8)
        self.norm_out = norm_layer(self.hidden_dim)
        self.init_weights(nn.ModuleList([self.cro_attn, self.ffn_layer, self.norm_out]))
        # positional encoding
        self.pe_layer = PositionEmbeddingSine(self.hidden_dim // 2, normalize=True)
        # losses
        self.cross_entropy_loss = nn.CrossEntropyLoss().to(args.device)
        self.focal_loss = FocalLoss().to(args.device)
        self.dice_loss = DiceLoss().to(args.device)

    def init_weights(self, init_modules):
        for m in init_modules.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def orthogonality_loss(self, prototypes):
        """
        make prototype vectors mutually orthogonal
        prototypes=(b, M, c)
        """
        b, M, c = prototypes.shape
        V = F.normalize(prototypes, dim=-1)
        Gram = torch.bmm(V, V.transpose(1, 2))  # (b, M, M), Gram[i,j] = v_i * v_j
        I = (
            torch.eye(M, device=prototypes.device, dtype=prototypes.dtype)
            .unsqueeze(0)
            .expand(b, -1, -1)
        )
        return (Gram - I).pow(2).mean()

    def gather_loss(self, query, q_mask, keys):
        q_mask = rearrange(q_mask.squeeze(-1), "b num M -> b (num M)")
        query = query * q_mask.unsqueeze(-1)
        self.distribution = 1.0 - F.cosine_similarity(
            query.unsqueeze(2), keys.unsqueeze(1), dim=-1
        )
        self.distance, self.cluster_index = torch.min(self.distribution, dim=2)
        gather_loss = self.distance.mean()
        return gather_loss

    @torch.no_grad()
    def feature_forward(self, image):
        b, num, c, h, w = image.shape
        feat = self.vision_encoder.get_intermediate_layers(image.view(-1, c, h, w))[0]
        feat = feat.view(b, num, -1, self.hidden_dim)
        return feat

    def get_mask(self, mask, target_size):
        b, _, _, _ = mask.shape
        mask = rearrange(mask, "b num h w -> (b num) h w").unsqueeze(1)
        mask = self.mask_downsample(mask)
        mask[mask >= 1] = 1
        if mask.shape[-2] != target_size:
            mask = F.interpolate(
                mask,
                size=(target_size, target_size),
                mode="nearest",
                align_corners=True,
            )
        mask = rearrange(mask, "(b num) 1 p_h p_w -> b num (p_h p_w)", b=b).unsqueeze(
            -1
        )  # b, num, p_h*p_w, 1
        return mask

    def nn_search(self, query_feat, support_feat, support_mask=None, mode="max"):
        """
        query_feat: (b, num_q, M, c)
        support_feat: (b, num_s, N, c)
        support_mask: (b, num_s, h, w)
        """
        _, num_q, M, _ = query_feat.shape
        _, num_s, N, _ = support_feat.shape
        query_feat = rearrange(
            F.normalize(query_feat, dim=-1), "b num_q M c -> b (num_q M) c"
        )
        support_feat = rearrange(
            F.normalize(support_feat, dim=-1), "b num_s N c -> b (num_s N) c"
        )

        if support_mask is not None:
            support_mask_ = self.get_mask(support_mask, int(N**0.5))
            support_mask = rearrange(
                support_mask_, "b num_s N 1 -> b 1 (num_s N)"
            ).repeat(1, M, 1)
        else:
            support_mask_ = None
            support_mask = 1

        similarity_map = (
            (1 + torch.einsum("bmc,bnc->bmn", query_feat, support_feat))
            / 2
            * support_mask
        )

        if mode == "max":
            pseudo_mask = similarity_map.max(dim=-1)[0]
        elif mode == "mean":
            pseudo_mask = similarity_map.mean(dim=-1)
        pseudo_mask = rearrange(pseudo_mask, "b (num_q M) -> b num_q M 1", num_q=num_q)

        return pseudo_mask, support_mask_

    def cos_sim_search(self, query_feat, support_feat, mode="max"):
        """
        query_feat: (b, num_q, M, c)
        support_feat: (b, num_s, N, c)
        support_mask: (b, num_s, h, w)
        """
        b_q, num_q, M, c_q = query_feat.shape
        b_s, num_s, N, c_s = support_feat.shape
        query_feat = rearrange(
            F.normalize(query_feat, dim=-1), "b num_q M c -> b (num_q M) c"
        )
        support_feat = rearrange(
            F.normalize(support_feat, dim=-1), "b num_s N c -> b (num_s N) c"
        )

        similarity_map = (
            1 + torch.einsum("bmc,bnc->bmn", query_feat, support_feat)
        ) / 2.0

        if mode == "max":
            pseudo_mask = similarity_map.max(dim=-1)[0]
        elif mode == "mean":
            pseudo_mask = similarity_map.mean(dim=-1)

        pseudo_mask = rearrange(pseudo_mask, "b (num_q M) -> b num_q M 1", num_q=num_q)

        return pseudo_mask

    def attention_forward(
        self, cross_layer, self_layer, query_embed, key_feat, value_feat, feat_mask=None
    ):
        """
        query_embed: (num_q, c)
        key_feat: (b, num_s, M, c)
        value_feat: (b, num_s, M, c)
        feat_mask: (b, num_s, M, 1)
        """
        B, _, _, C = value_feat.shape

        if isinstance(query_embed, nn.Embedding):
            # gaussian init
            # nn.init.normal_(query_embed.weight, mean=0, std=0.02)
            q_supp_out = query_embed.weight.unsqueeze(1).repeat(
                1, B, 1
            )  # (num_q, c) -> (num_q, b, c)
        elif query_embed.dim() == 2:
            q_supp_out = query_embed.unsqueeze(1).repeat(
                1, B, 1
            )  # (num_q, c) -> (num_q, b, c)
        else:
            q_supp_out = query_embed.permute(1, 0, 2)  # (b, num_q, c) -> (num_q, b, c)

        key = rearrange(key_feat, "b num M c -> (num M) b c")
        pos_embedding = self.pe_layer(value_feat, feat_mask)
        value = rearrange(value_feat, "b num M c -> (num M) b c")

        if feat_mask is not None:
            attn_mask = (
                rearrange(feat_mask.squeeze(-1), "b num M -> b (num M)")
                .unsqueeze(1)
                .repeat(self.nheads, q_supp_out.shape[0], 1)
            )  # (batch*nheads, query_len, key_len)
            attn_mask = -1e9 * (1 - attn_mask)
        else:
            attn_mask = feat_mask

        output = cross_layer(
            q_supp_out,
            key,
            value,
            memory_mask=attn_mask,
            memory_key_padding_mask=None,
            pos=pos_embedding,
            query_pos=None,
        )
        output = self_layer(
            output, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None
        )

        return output.permute(1, 0, 2)  # (num_q, b, c) -> (b, num_q, c)

    def attention_FFN_forward(
        self, cross_layer, ffn_layer, query_embed, key_feat, value_feat, feat_mask=None
    ):
        """
        query_embed: (num_q, c)
        key_feat: (b, num_s, M, c)
        value_feat: (b, num_s, M, c)
        feat_mask: (b, num_s, M, 1)
        """
        B, _, _, C = value_feat.shape

        if isinstance(query_embed, nn.Embedding):
            # gaussian init
            # nn.init.normal_(query_embed.weight, mean=0, std=0.02)
            q_supp_out = query_embed.weight.unsqueeze(1).repeat(
                1, B, 1
            )  # (num_q, c) -> (num_q, b, c)
        elif query_embed.dim() == 2:
            q_supp_out = query_embed.unsqueeze(1).repeat(
                1, B, 1
            )  # (num_q, c) -> (num_q, b, c)
        else:
            q_supp_out = query_embed.permute(1, 0, 2)  # (b, num_q, c) -> (num_q, b, c)

        key = rearrange(key_feat, "b num M c -> (num M) b c")
        pos_embedding = self.pe_layer(value_feat, feat_mask)
        value = rearrange(value_feat, "b num M c -> (num M) b c")

        if feat_mask is not None:
            attn_mask = (
                rearrange(feat_mask.squeeze(-1), "b num M -> b (num M)")
                .unsqueeze(1)
                .repeat(self.nheads, q_supp_out.shape[0], 1)
            )  # (batch*nheads, query_len, key_len)
            attn_mask = -1e9 * (1 - attn_mask)
        else:
            attn_mask = feat_mask

        output = cross_layer(
            q_supp_out,
            key,
            value,
            memory_mask=attn_mask,
            memory_key_padding_mask=None,
            pos=pos_embedding,
            query_pos=None,
        )
        output = output.permute(1, 0, 2)  # (num_q, b, c) -> (b, num_q, c)
        output = output + ffn_layer(output)
        output = self.norm_out(output)

        return output  # (b, num_q, c)

    def get_res_feat(self, ori_feat, temp_feat):
        """
        ori_feat: (b, num, h*w, c)
        """
        b, num, h_w, c = ori_feat.shape
        ori_feat = rearrange(ori_feat, "b num h_w c -> b (num h_w) c")
        temp_feat = rearrange(temp_feat, "b num h_w c -> b (num h_w) c")
        sim_map = (1 + torch.einsum("bmc,bnc->bmn", ori_feat, temp_feat)) / 2
        max_idx = sim_map.max(dim=-1)[1]
        most_sim_feat = torch.gather(
            temp_feat, 1, max_idx.unsqueeze(-1).repeat(1, 1, c)
        )
        res_feat = ori_feat - most_sim_feat
        return res_feat  # [b, num, c]

    def get_real_residual_feat(self, target_fea, normal_fea):
        """
        target_fea:  (b, num_t, h*w, c)
        """
        tar_b, tar_num, _, _ = target_fea.shape

        target_fea = rearrange(target_fea, "b num h_w c -> b (num h_w) c")
        normal_fea = rearrange(normal_fea, "b num h_w c -> b (num h_w) c")

        B, Nq, C = target_fea.shape
        _, Nn, _ = normal_fea.shape
        k = min(self.args.topk, Nn)
        r = min(self.args.topr, k)
        # -------- 1) nearest normal retrieval
        q = F.normalize(target_fea, dim=-1)  # [B, Nq, C]
        n = F.normalize(normal_fea, dim=-1)  # [B, Nn, C]
        sim = torch.matmul(q, n.transpose(-1, -2))  # [B, Nq, Nn]
        topk_sim, topk_idx = torch.topk(sim, k=k, dim=-1, largest=True)  # [B, Nq, k]
        normal_expand = normal_fea.unsqueeze(1).expand(B, Nq, Nn, C)
        idx_expand = topk_idx.unsqueeze(-1).expand(B, Nq, k, C)
        knn_feat = torch.gather(normal_expand, dim=2, index=idx_expand)  # [B, Nq, k, C]

        # -------- 2) local weighted normal center
        topk_weights = F.softmax(topk_sim, dim=-1)
        normal_center = torch.sum(
            knn_feat * topk_weights.unsqueeze(-1), dim=2
        )  # [B, Nq, C]
        # == raw residual ==
        raw_residual = target_fea - normal_center  # [B, Nq, C]

        # -------- 3) local PCA
        centered_knn = knn_feat - normal_center.unsqueeze(2)  # [B, Nq, k, C]
        sqrt_w = torch.sqrt(topk_weights + 1e-12).unsqueeze(-1)
        xw = centered_knn * sqrt_w  # [B, Nq, k, C]
        # gram matrix: [B, Nq, k, k]
        G = xw @ xw.transpose(-1, -2)
        eigvals, eigvecs = torch.linalg.eigh(G)  # ascending
        r = min(r, G.size(-1))
        eigvals = eigvals[..., -r:]  # [B, Nq, r]
        eigvecs = eigvecs[..., -r:]  # [B, Nq, k, r]
        U = xw.transpose(-1, -2) @ eigvecs
        U = U / torch.sqrt(eigvals.unsqueeze(-2) + 1e-8)
        U = F.normalize(U, dim=-2)
        rr = raw_residual.unsqueeze(-1)  # [B, Nq, C, 1]
        coeff = U.transpose(-1, -2) @ rr  # [B, Nq, r, 1]
        normal_proj = (U @ coeff).squeeze(-1)  # [B, Nq, C]
        clean_residual = raw_residual - self.args.proj_alpha * normal_proj
        return clean_residual

    @staticmethod
    def get_residual_pseudo_mask(residual_feat):
        """
        Select the top 1% largest residual-norm patches for each abnormal reference.
        """
        num_patches = residual_feat.shape[2]
        topk_len = max(1, int(num_patches * 0.01))
        residual_scores = torch.norm(residual_feat, p=2, dim=-1)
        topk_indices = torch.topk(residual_scores, topk_len, dim=-1)[1]
        pseudo_mask = torch.zeros_like(residual_scores)
        pseudo_mask.scatter_(-1, topk_indices, 1.0)
        return pseudo_mask.unsqueeze(-1)

    def feat_inner(self, features, vectors, basis=True):
        if vectors.dim() == 1:
            vectors = vectors.unsqueeze(dim=0)
            scales = torch.inner(features, vectors) / vectors.square().sum(dim=1)
            return scales @ vectors
        else:
            vector_basis = (
                vectors if basis else torch.linalg.svd(vectors, full_matrices=False)[2]
            )
            return (features @ vector_basis.mT) @ vector_basis

    def knn_cos_distance(self, query_fea, ano_fea, mode="max"):
        x1_norm = F.normalize(query_fea, dim=-1)  # [b, num_q, c]
        x2_norm = F.normalize(ano_fea, dim=-1)  # [b, num_a, c]
        # [-1, 1] -> [0, 1]
        similarity_map = (1 + torch.einsum("bmc,bnc->bmn", x1_norm, x2_norm)) / 2.0
        if mode == "max":
            scores = similarity_map.max(dim=-1)[0]
        elif mode == "mean":
            scores = similarity_map.mean(dim=-1)
        return scores

    def get_l2_distance(self, x1, x2):
        x1 = F.normalize(x1, p=2, dim=-1)
        x2 = F.normalize(x2, p=2, dim=-1)
        dist_l2_score = torch.norm(x1 - x2, p=2, dim=-1) / 2
        return dist_l2_score  # [b, num_q]

    def prepare_test_image(self, img, transform):
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        image_tensor = transform(img)
        # Crop image to dimensions that are a multiple of the patch size
        height, width = image_tensor.shape[1:]  # C x H x W
        cropped_width, cropped_height = (
            width - width % self.vision_encoder.patch_size,
            height - height % self.vision_encoder.patch_size,
        )
        image_tensor = image_tensor[:, :cropped_height, :cropped_width]

        grid_size = (
            cropped_height // self.vision_encoder.patch_size,
            cropped_width // self.vision_encoder.patch_size,
        )
        # return image_tensor
        return image_tensor, grid_size

    def forward(
        self,
        args,
        query_image,
        query_mask,
        query_label,
        support_normal,
        support_abnormal,
        mode="train",
    ):
        query_feat = self.feature_forward(query_image)
        q_B, q_num, _, q_C = query_feat.shape
        loss_i = 0
        loss_p = 0
        if args.n_shot > 0:
            support_n_image, support_n_mask_ = support_normal
            support_n_feat = self.feature_forward(support_n_image)
            n_pseudo_mask, support_n_mask = self.nn_search(
                query_feat, support_n_feat, 1 - support_n_mask_
            )
            s_n = n_pseudo_mask.squeeze(-1)

        if args.a_shot > 0:
            use_residual_pseudo_mask = getattr(args, "use_residual_pseudo_mask", False)
            support_a_image = support_abnormal[0]
            support_a_feat = self.feature_forward(support_a_image)
            if use_residual_pseudo_mask and args.n_shot == 0:
                raise ValueError("use_residual_pseudo_mask requires n_shot > 0.")

            if not use_residual_pseudo_mask:
                support_a_mask_ = support_abnormal[1]
                _, support_a_mask = self.nn_search(
                    query_feat, support_a_feat, support_a_mask_
                )

                if args.n_shot == 0:  # only abnormal as reference
                    support_n_feat = support_a_feat.masked_select(
                        (1 - support_a_mask).bool()
                    ).view(
                        support_a_feat.shape[0],
                        1,
                        -1,
                        support_a_feat.shape[-1],
                    )
                    n_pseudo_mask, support_n_mask = self.nn_search(
                        query_feat, support_a_feat, 1 - support_a_mask_
                    )
                    s_n = n_pseudo_mask.squeeze(-1)

            support_res_feat = self.get_real_residual_feat(
                support_a_feat, support_n_feat
            )  # (b, num*p_h*p_w, c)
            support_real_res_feat = rearrange(
                support_res_feat,
                "b (num h_w) c -> b num h_w c",
                b=support_a_feat.shape[0],
                num=support_a_feat.shape[1],
            )
            if use_residual_pseudo_mask:
                support_a_mask = self.get_residual_pseudo_mask(support_real_res_feat)
            anomaly_directions = self.attention_FFN_forward(
                self.cro_attn,
                self.ffn_layer,
                self.learnable_proxies,  # Q
                support_a_feat,  # K
                support_real_res_feat,  # V
                support_a_mask,
            )
            g_loss = self.gather_loss(
                support_res_feat, support_a_mask, anomaly_directions
            )
            ortho_loss = self.orthogonality_loss(anomaly_directions)
            query_res_feat = self.get_real_residual_feat(query_feat, support_n_feat)
            guided_query_res_feat = self.feat_inner(query_res_feat, anomaly_directions)
            query_l2_dist = self.get_l2_distance(
                query_res_feat, guided_query_res_feat
            )  # [b, num_q]
            a_out = 1 - query_l2_dist
            s_a = a_out.reshape(q_B, 1, a_out.shape[-1])

        if args.n_shot > 0 and args.a_shot == 0:
            s_a = 1 - s_n
        elif args.n_shot == 0 and args.a_shot > 0:
            s_n = 1 - s_a
        else:
            assert (
                args.n_shot > 0 or args.a_shot > 0
            ), "n_shot and a_shot should not be both 0"

        pixel_level_logits = torch.cat([s_n, s_a], dim=1)  # (b, 2, h*w)

        a_score = (s_a + (1 - s_n)) / 2  # [b, 1, h*w]

        if mode == "train":
            # top-k mean = 1%
            topk_len = max(1, int(a_score.shape[-1] * 0.01))
            a_score_topk = torch.topk(a_score, topk_len, dim=-1)[0].mean(dim=-1)
            image_level_logits = torch.cat([1 - a_score_topk, a_score_topk], dim=-1)

            # Image Level
            loss_i += self.cross_entropy_loss(image_level_logits, query_label.long())

            # Pixel Level
            l = int(pixel_level_logits.shape[-1] ** 0.5)
            pixel_level_logits = rearrange(
                pixel_level_logits, "b n (h w) -> b n h w", h=l
            )
            pixel_level_logits = F.interpolate(
                pixel_level_logits, size=query_mask.shape[-2:], mode="bilinear"
            )
            query_mask_n = torch.stack([1 - query_mask, query_mask], dim=1)
            loss_p += self.focal_loss(pixel_level_logits, query_mask_n)
            loss_p += self.dice_loss(pixel_level_logits, query_mask_n)
            if args.a_shot > 0:
                loss_p += args.g_loss_w * g_loss + args.or_loss_w * ortho_loss

            return image_level_logits, pixel_level_logits, loss_i, loss_p

        elif mode == "test":
            return a_score
