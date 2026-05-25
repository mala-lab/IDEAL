# !/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import json
import torch
from torch.utils.data import DataLoader
from models.our_model import IDEAL
from utils.utils_common import setup_seed, get_transform
from utils.dataset_all import FSDataset
from train_ours import run_model


DATA_TARGETS = [
    "MVTecAD",
    "VisA",
    "BTAD",
    "MPDD",
    "AITEX",
    "BraTS2021",
    "Liver",
    "RESC",
]


def infer_config_from_checkpoint(args):
    ckpt_path = args.checkpoint_path
    ckpt_name = os.path.basename(ckpt_path)
    ckpt_dir = os.path.basename(os.path.dirname(ckpt_path))

    if args.test_ano_setting is None:
        if "hard" in ckpt_dir.lower():
            args.test_ano_setting = "hard"
        else:
            args.test_ano_setting = "general"

    if args.n_shot is None:
        args.n_shot = 1
    if args.a_shot is None:
        args.a_shot = 1
    if args.data_target is None:
        raise ValueError(
            "Unable to infer --data_target from checkpoint name. "
            "Please pass --data_target explicitly."
        )


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    if unexpected:
        print("The unexpected keys as:", unexpected)


def parse_args():
    parser = argparse.ArgumentParser("IDEAL-Inference", add_help=True)
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to pretrained checkpoint",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./dataset/data",
        help="dataset root",
    )
    parser.add_argument(
        "--data_target",
        type=str,
        default="VisA",
        choices=DATA_TARGETS,
        help="target dataset",
    )
    parser.add_argument(
        "--test_ano_setting",
        type=str,
        default=None,
        choices=["general", "hard"],
        help="test anomaly setting",
    )
    parser.add_argument("--worker", type=int, default=4, help="number of workers")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--image_size", type=int, default=448, help="image size")
    parser.add_argument("--print_freq", type=int, default=50, help="print frequency")
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu index")
    parser.add_argument("--seed", type=int, default=77, help="random seed")
    parser.add_argument("--pro_num_thresholds", type=int, default=50, help="thresholds metrics")
    parser.add_argument("--backbone_name", type=str, default="dinov2_vits14")
    parser.add_argument("--n_shot", type=int, default=1)
    parser.add_argument("--a_shot", type=int, default=1)
    parser.add_argument("--deviation_vectors", type=int, default=45)
    parser.add_argument("--nheads", type=int, default=12)
    parser.add_argument("--topk", type=int, default=12)
    parser.add_argument("--topr", type=int, default=4)
    parser.add_argument("--proj_alpha", type=float, default=0.8)
    parser.add_argument("--g_loss_w", type=float, default=1.0)
    parser.add_argument("--or_loss_w", type=float, default=0.8)
    parser.add_argument("--grad_clip", type=float, default=0.0)

    return parser.parse_args()


def main():
    args = parse_args()
    infer_config_from_checkpoint(args)
    print(args)

    setup_seed(args.seed)
    args.device = torch.device(
        f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    model = IDEAL(args).to(args.device)
    load_checkpoint(model, args.checkpoint_path, args.device)
    model.eval()

    transform = get_transform((args.image_size, args.image_size))
    test_data = FSDataset(
        data_root=args.data_root,
        data_target=args.data_target,
        split="test",
        shot=[args.n_shot, args.a_shot],
        transform=transform,
        test_ano_setting=args.test_ano_setting
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.worker,
        shuffle=False
    )

    with torch.no_grad():
        metrics, mean_loss = run_model(
            args,
            model,
            test_dataloader,
            training=False,
            compute_pro=True,
        )

    print(
        "======> Test Inference Results:\t|| "
        f"I-AUROC: {metrics['i_auroc']:.4f}, "
        f"I-AP: {metrics['i_ap']:.4f}, "
        f"I-F1max: {metrics['i_f1_max']:.4f}, "
        f"P-AUROC: {metrics['p_auroc']:.4f}, "
        f"P-F1max: {metrics['p_f1_max']:.4f}, "
        f"P-PRO: {metrics['p_pro']:.4f}, "
        f"Loss: {mean_loss:.4f}"
    )


if __name__ == "__main__":
    main()
