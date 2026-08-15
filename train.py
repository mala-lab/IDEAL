# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.11
import os
import time
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
import argparse
from tqdm import tqdm
import numpy as np
from models.model import IDEAL
from utils.utils_common import (
    setup_seed,
    get_transform,
    get_crop_transform,
    get_strong_transforms,
    WarmCosineScheduler,
)
from utils.dataset_all import FSDataset
from utils.pre_metrics import FSMetric
from utils.optimizers import StableAdamW


class SeenUnseenSimilarityMeter:
    def __init__(self):
        self.by_product = {}

    def update(self, product, sims):
        if sims is None:
            return
        vals = np.asarray(sims, dtype=np.float32).reshape(-1)
        if vals.size == 0:
            return
        self.by_product.setdefault(product, []).extend(vals.tolist())

    def summarize(self):
        summary = {}
        for p, vals in self.by_product.items():
            arr = np.asarray(vals, dtype=np.float32)
            summary[p] = {
                "sus_mean": float(arr.mean()),
                "sus_std": float(arr.std()),
                "count": int(arr.size),
            }
        if summary:
            all_vals = np.concatenate(
                [np.asarray(v, dtype=np.float32) for v in self.by_product.values()]
            )
            summary["__all__"] = {
                "sus_mean": float(all_vals.mean()),
                "sus_std": float(all_vals.std()),
                "count": int(all_vals.size),
            }
        else:
            summary["__all__"] = {
                "sus_mean": float("nan"),
                "sus_std": float("nan"),
                "count": 0,
            }
        return summary

    def print_summary(self):
        summary = self.summarize()
        print(f'{"Product":<20} {"SUS-Mean":<12} {"SUS-Std":<12} {"Count":<10}')
        for product, info in summary.items():
            if product == "__all__":
                continue
            print(
                f"{product:<20} {info['sus_mean']:<12.4f} {info['sus_std']:<12.4f} {info['count']:<10d}"
            )
        all_info = summary["__all__"]
        print(
            f'{"Mean(All)":<20} {all_info["sus_mean"]:<12.4f} {all_info["sus_std"]:<12.4f} {all_info["count"]:<10d}'
        )


def _masked_patch_mean(feat, mask_2d):
    b, num, t, c = feat.shape
    side = int(t**0.5)
    if side * side != t:
        return feat.mean(dim=(1, 2))
    m = mask_2d.float().view(b * num, 1, mask_2d.shape[-2], mask_2d.shape[-1])
    m = torch.nn.functional.interpolate(m, size=(side, side), mode="nearest")
    m = m.view(b, num, t, 1).to(feat.device)
    nume = (feat * m).sum(dim=(1, 2))
    deno = m.sum(dim=(1, 2)).clamp_min(1.0)
    out = nume / deno
    empty = (m.sum(dim=(1, 2, 3)) < 1.0).unsqueeze(-1)
    return torch.where(empty, feat.mean(dim=(1, 2)), out)


def _unpack_batch_tensors(data, args):
    query = data["query"]
    query_image = query[0].to(args.device)
    query_mask = query[1].squeeze(1).to(args.device)
    sample_product = data["sample_product"]
    image_level_label = data["image_level_label"][0].to(args.device)
    support_normal = data["support_normal"]
    support_abnormal = data["support_abnormal"]
    if support_normal is not None and isinstance(support_normal[0], torch.Tensor):
        support_n_img, support_n_mask = support_normal
        support_normal = (
            support_n_img.to(args.device),
            support_n_mask.to(args.device),
        )
    if support_abnormal is not None and isinstance(support_abnormal[0], torch.Tensor):
        support_a_img, support_a_mask = support_abnormal
        support_abnormal = (
            support_a_img.to(args.device),
            support_a_mask.to(args.device),
        )
    return query_image, query_mask, image_level_label, support_normal, support_abnormal


def run_model(
    args,
    model,
    dataloader,
    optimizer=None,
    scheduler=None,
    training=True,
    compute_pro=None,
):
    if training:
        model.train()
    else:
        model.eval()

    products = dataloader.dataset.products
    mean_loss, mean_loss_i, mean_loss_p = 0, 0, 0

    if compute_pro is None:
        compute_pro = not training
    fs_metric = FSMetric(
        args,
        products,
        compute_pro=compute_pro,
        pro_num_thresholds=args.pro_num_thresholds,
        apply_smoothing=not training,
    )
    sus_meter = None
    for i, data in enumerate(dataloader):
        sample_product = data["sample_product"]
        query_image, query_mask, image_level_label, support_normal, support_abnormal = (
            _unpack_batch_tensors(data, args)
        )
        image_level_logits, pixel_level_logits, loss_i, loss_p = model(
            args,
            query_image,
            query_mask,
            image_level_label,
            support_normal,
            support_abnormal,
        )
        loss = loss_i + loss_p
        fs_metric.update(
            image_level_logits,
            image_level_label,
            pixel_level_logits,
            query_mask,
            sample_product,
        )
        mean_loss += loss
        mean_loss_i += loss_i
        mean_loss_p += loss_p

        if i % args.print_freq == 0:
            current_iter = i + 1
            print(
                f"Iter: {i} \t || Total Loss: {mean_loss/current_iter:.4f}, I-Loss: {mean_loss_i/current_iter:.4f}, P-Loss: {mean_loss_p/current_iter:.4f}"
            )
        if training and args.a_shot > 0:
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
        elif sus_meter is not None:
            query_type_list = data.get("query_type", None)
            query_specie_list = data.get("query_specie_name", None)
            ref_abnormal_specie_list = data.get("ref_abnormal_specie_name", None)
            if (
                query_type_list is None
                or query_specie_list is None
                or ref_abnormal_specie_list is None
            ):
                continue

            with torch.no_grad():
                q_feat = model.feature_forward(query_image)  # (B, num, hw, C)
                s_a_feat = model.feature_forward(support_abnormal[0])
                q_vec = _masked_patch_mean(q_feat, query_mask.unsqueeze(1))  # (B, C)
                s_vec = _masked_patch_mean(s_a_feat, support_abnormal[1])  # (B, C)
                if support_normal is not None and isinstance(
                    support_normal[0], torch.Tensor
                ):
                    s_n_feat = model.feature_forward(
                        support_normal[0]
                    )  # (B, n_shot, hw, C)
                    n_vec = s_n_feat.mean(dim=(1, 2))
                    q_vec = q_vec - n_vec
                    s_vec = s_vec - n_vec
                sim_vec = torch.nn.functional.cosine_similarity(q_vec, s_vec, dim=-1)

            bsz = sim_vec.shape[0]
            for bi in range(bsz):
                q_type = query_type_list[bi]
                q_specie = query_specie_list[bi]
                ref_specie = ref_abnormal_specie_list[bi]
                if q_type != "abnormal":
                    continue
                if q_specie in set(ref_specie.split("|")):
                    continue
                sus_meter.update(sample_product[bi], [sim_vec[bi].item()])

    mean_i_roc, mean_p_roc = fs_metric.get_scores()
    fs_metric.print_metrics()
    metric_summary = dict(fs_metric.mean_metrics)
    if sus_meter is not None:
        sus_meter.print_summary()
        sus_summary = sus_meter.summarize()
        metric_summary["sus_mean"] = sus_summary["__all__"]["sus_mean"]
        metric_summary["sus_std"] = sus_summary["__all__"]["sus_std"]
        metric_summary["sus_count"] = sus_summary["__all__"]["count"]
    return metric_summary, mean_loss / len(dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("IDEAL Anomaly Detection", add_help=True)
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/wanghuan/dataset",
        help="dataset path",
    )
    parser.add_argument(
        "--data_target",
        type=str,
        default="MVTecAD",
        choices=[
            "MVTecAD",
            "VisA",
            "BTAD",
            "MPDD",
            "AITEX",
            "BraTS2021",
            "Liver",
            "RESC",
        ],
        help="target dataset",
    )
    parser.add_argument(
        "--test_ano_setting",
        type=str,
        default="general",
        choices=["general", "hard"],
        help="test anomaly setting: general or hard",
    )
    parser.add_argument(
        "--test_ref_object",
        type=str,
        default=None,
        help="test object whose fixed abnormal reference type is selected explicitly",
    )
    parser.add_argument(
        "--test_ref_anomaly_type",
        type=str,
        default=None,
        help="anomaly type used as the fixed reference pool for --test_ref_object",
    )
    parser.add_argument(
        "--save_path", type=str, default="./outputs", help="path to save results"
    )
    parser.add_argument("--worker", type=int, default=4, help="number of workers")
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help="number of cpu threads to use during batch generation",
    )
    parser.add_argument(
        "--port",
        type=str,
        default="1234",
        help="number of cpu threads to use during batch generation",
    )
    parser.add_argument(
        "--backbone_name",
        type=str,
        default="dinov2_vits14",
        help="the name of encoder",
    )
    parser.add_argument(
        "--n_shot", type=int, default=4, help="number of normal samples"
    )
    parser.add_argument(
        "--a_shot", type=int, default=1, help="number of abnormal samples"
    )
    parser.add_argument(
        "--a_ref_types",
        type=int,
        default=1,
        help="number of anomaly types represented by fixed test abnormal references",
    )
    parser.add_argument(
        "--use_residual_pseudo_mask",
        action="store_true",
        help="ignore abnormal-reference GT masks and use top-1 residual patches",
    )
    parser.add_argument("--epochs", type=int, default=20, help="epochs")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-5, help="learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="weight decay for AdamW"
    )
    parser.add_argument(
        "--scheduler_type",
        type=str,
        default="cosine",
        choices=["multistep", "cosine"],
        help="lr scheduler: multistep or cosine",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=2,
        help="warmup epochs for cosine scheduler",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=0.0,
        help="gradient clipping max norm, 0 to disable",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--image_size", type=int, default=448, help="image size")
    parser.add_argument("--crop_size", type=int, default=392, help="crop size")
    parser.add_argument("--train_choice", type=int, default=500, help="train_choice")
    parser.add_argument("--test_choice", type=int, default=-1, help="all test")
    parser.add_argument("--print_freq", type=int, default=50, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency")
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu index")
    parser.add_argument("--seed", type=int, default=17, help="random seed")
    parser.add_argument("--save_m", type=bool, default=True, help="save model")
    parser.add_argument(
        "--pro_num_thresholds",
        type=int,
        default=50,
        help="number of thresholds for PRO/AUPRO computation",
    )
    parser.add_argument(
        "--num_learnable_vectors",
        type=int,
        default=45,
        help="number of learnable vectors in IDE",
    )
    parser.add_argument(
        "--nheads",
        type=int,
        default=4,
        help="num of heads",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=8,
        help="num of k",
    )
    parser.add_argument(
        "--topr",
        type=int,
        default=4,
        help="num of r",
    )
    parser.add_argument(
        "--proj_alpha",
        type=float,
        default=0.6,
        help="alpha of proj",
    )
    parser.add_argument(
        "--g_loss_w",
        type=float,
        default=0.2,
        help="weight of g_loss",
    )
    parser.add_argument(
        "--or_loss_w",
        type=float,
        default=0.2,
        help="weight of ortho_loss",
    )

    args = parser.parse_args()
    print(args)

    setup_seed(args.seed)
    args.device = torch.device(
        f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    # IDEAL model
    model = IDEAL(args)
    model.to(args.device)
    freeze_params_key_name = ["vision_encoder", "mask_downsample"]
    trainable_params = []
    trainable_num = 0
    print("=" * 50)
    for param_name, param in model.named_parameters():
        if param_name.split(".")[0] not in freeze_params_key_name:
            print(param_name, param.shape)
            param.requires_grad_(True)
            trainable_params.append({"params": param})
        else:
            param.requires_grad_(False)

    transform = get_transform((args.image_size, args.image_size))
    train_data = FSDataset(
        data_root=args.data_root,
        data_target=args.data_target,
        split="train",
        shot=[args.n_shot, args.a_shot],
        transform=transform,
        choice=args.train_choice,
        test_ano_setting=args.test_ano_setting,
    )
    train_dataloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.worker,
        shuffle=True,
    )
    val_data = FSDataset(
        data_root=args.data_root,
        data_target=args.data_target,
        split="test",  # test
        shot=[args.n_shot, args.a_shot],
        transform=transform,
        choice=args.test_choice,
        test_ano_setting=args.test_ano_setting,
        test_ref_object=args.test_ref_object,
        test_ref_anomaly_type=args.test_ref_anomaly_type,
        a_ref_types=args.a_ref_types,
    )
    val_dataloader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.worker,
        shuffle=False,
    )

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
        amsgrad=True,
        eps=1e-10,
    )
    if args.scheduler_type == "cosine":
        warmup_start = args.learning_rate * 0.01
        for g in optimizer.param_groups:
            g["lr"] = warmup_start
        scheduler = WarmCosineScheduler(
            optimizer,
            base_value=args.learning_rate,
            final_value=args.learning_rate * 0.01,
            total_iters=args.epochs * len(train_dataloader),
            warmup_iters=len(train_dataloader) * 2,
            start_warmup_value=warmup_start,
        )
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[10, 15], gamma=0.1
        )

    # Training and Validation
    best_roc = 0
    best_f1max = 0
    best_epoch = -1
    best_i_roc, best_i_ap, best_i_f1max = 0.0, 0.0, 0.0
    best_p_roc, best_p_pro, best_p_f1max = 0.0, 0.0, 0.0
    all_i_roc, all_i_ap, all_i_f1max = [], [], []
    all_p_roc, all_p_pro, all_p_f1max = [], [], []
    train_wall_t0 = time.perf_counter()
    for epoch in range(args.epochs):
        print("-" * 88)
        print(f"======> Epoch: {epoch}, Learning Rate: {scheduler.get_last_lr()[0]}")
        print(f"======> Training:")
        train_metrics, mean_loss = run_model(
            args,
            model,
            train_dataloader,
            optimizer,
            scheduler,
            training=True,
            compute_pro=False,
        )

        print(
            "======> Train Results:\t|| "
            f"I-AUROC: {train_metrics['i_auroc']:.4f}, "
            f"I-AP: {train_metrics['i_ap']:.4f}, "
            f"I-F1max: {train_metrics['i_f1_max']:.4f}, "
            f"P-AUROC: {train_metrics['p_auroc']:.4f}, "
            f"P-F1max: {train_metrics['p_f1_max']:.4f}, "
            f"P-PRO: {train_metrics['p_pro']:.4f}, "
            f"Train Loss: {mean_loss:.4f}"
        )
        if args.save_m:
            torch.save(
                {
                    name: param
                    for name, param in model.named_parameters()
                    if param.requires_grad
                },
                f"{args.save_path}/n_{args.n_shot}_a_{args.a_shot}_{args.data_target}_last.pth",
            )

        with torch.no_grad():
            print(f"======> Validation:")
            val_metrics, mean_loss = run_model(
                args,
                model,
                val_dataloader,
                training=False,
                compute_pro=True,
            )
            print(
                "======> Validation Results:\t|| "
                f"I-AUROC: {val_metrics['i_auroc']:.4f}, "
                f"I-AP: {val_metrics['i_ap']:.4f}, "
                f"I-F1max: {val_metrics['i_f1_max']:.4f}, "
                f"P-AUROC: {val_metrics['p_auroc']:.4f}, "
                f"P-F1max: {val_metrics['p_f1_max']:.4f}, "
                f"P-PRO: {val_metrics['p_pro']:.4f}, "
                f"Val Loss: {mean_loss:.4f}"
            )

            all_i_roc.append(format(val_metrics["i_auroc"], ".4f"))
            all_i_ap.append(format(val_metrics["i_ap"], ".4f"))
            all_i_f1max.append(format(val_metrics["i_f1_max"], ".4f"))
            all_p_roc.append(format(val_metrics["p_auroc"], ".4f"))
            all_p_f1max.append(format(val_metrics["p_f1_max"], ".4f"))
            all_p_pro.append(format(val_metrics["p_pro"], ".4f"))

            if val_metrics["i_auroc"] + val_metrics["p_auroc"] >= best_roc:
                best_epoch = epoch
                best_i_roc = val_metrics["i_auroc"]
                best_p_roc = val_metrics["p_auroc"]
                best_i_f1max = val_metrics["i_f1_max"]
                best_p_f1max = val_metrics["p_f1_max"]
                best_roc = best_i_roc + best_p_roc
                best_f1max = best_i_f1max + best_p_f1max
                if args.save_m:
                    torch.save(
                        {
                            name: param
                            for name, param in model.named_parameters()
                            if param.requires_grad
                        },
                        f"{args.save_path}/n_{args.n_shot}_a_{args.a_shot}_{args.data_target}_best.pth",
                    )

            print(
                f"======> Previous Best I-AUROC: {best_i_roc:.4f}({best_epoch}) || Best P-AUROC: {best_p_roc:.4f}({best_epoch}) || Best I-F1max: {best_i_f1max:.4f}({best_epoch}) || Best P-F1max: {best_p_f1max:.4f}({best_epoch})"
            )
        print("-" * 88)
        print("\n")
    train_wall_secs = time.perf_counter() - train_wall_t0
    print(
        f"======> FINAL_RESULT I-AUROC: {max(all_i_roc)} || FINAL_RESULT P-AUROC: {max(all_p_roc)}"
    )
    print(
        f"======> FINAL_RESULT I-AP: {max(all_i_ap)} || FINAL_RESULT P-PRO: {max(all_p_pro)}"
    )
    print(
        f"======> FINAL_RESULT I-F1max: {max(all_i_f1max)} || FINAL_RESULT P-F1max: {max(all_p_f1max)}"
    )
