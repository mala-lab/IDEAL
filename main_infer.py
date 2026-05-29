# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.11

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import numpy as np
from PIL import Image
from utils.common_tools import setup_seed, get_transform
from utils.pre_episode import EpisodeSet
from utils.eva_metrics import EvaMetric


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

# data sample endwith
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def _is_image_file(path):
    return os.path.isfile(path) and path.lower().endswith(IMG_EXTS)


def _list_images(folder):
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"reference folder not found: {folder}")
    paths = [
        os.path.join(folder, name)
        for name in sorted(os.listdir(folder))
        if _is_image_file(os.path.join(folder, name))
    ]
    if not paths:
        raise FileNotFoundError(f"no reference images found in: {folder}")
    return paths


def _repeat_to_count(paths, count):
    return [paths[i % len(paths)] for i in range(count)]


def _load_binary_mask(mask_path, out_hw):
    mask = np.array(Image.open(mask_path).convert("L"))
    mask = torch.from_numpy((mask > 0).astype(np.float32))
    mask = F.interpolate(
        mask.unsqueeze(0).unsqueeze(0),
        size=out_hw,
        mode="nearest"
    )
    return mask.squeeze(0).squeeze(0)


class ReferenceSupportLoader:
    def __init__(self, args, dataset, transform):
        root = args.ref_root
        if not root:
            raise ValueError("ref_root is empty")
        
        target_root = os.path.join(root, args.data_target)
        self.root = target_root if os.path.isdir(target_root) else root
        self.dataset = dataset
        self.transform = transform
        self.n_shot = args.n_shot
        self.a_shot = args.a_shot
        self.cache = {}

    def _find_abnormal_mask(self, image_path):
        stem = os.path.splitext(os.path.basename(image_path))[0]
        image_dir = os.path.dirname(image_path)
        for ext in IMG_EXTS:
            path = os.path.join(image_dir, f"{stem}_mask{ext}")
            if os.path.isfile(path):
                return path
        return None

    def _load_ref_image(self, path):
        img = Image.open(path).convert("RGB")
        img_tensor = self.transform(img)
        return img_tensor

    def _load_product_refs(self, product):
        product_root = os.path.join(self.root, product)
        normal_paths = _repeat_to_count(
            _list_images(os.path.join(product_root, "normal")),
            self.n_shot,
        )
        abnormal_all = _list_images(os.path.join(product_root, "abnormal"))
        abnormal_image_paths = [
            path
            for path in abnormal_all
            if "_mask" not in os.path.splitext(os.path.basename(path))[0]
        ]
        if not abnormal_image_paths:
            raise FileNotFoundError(
                f"no abnormal reference images found in: {os.path.join(product_root, 'abnormal')}"
            )
        abnormal_paths = _repeat_to_count(
            abnormal_image_paths,
            self.a_shot,
        )

        normal_imgs, normal_masks = [], []
        for path in normal_paths:
            img_tensor = self._load_ref_image(path)
            normal_imgs.append(img_tensor)
            normal_masks.append(torch.zeros(img_tensor.shape[-2:], dtype=torch.float32))

        abnormal_imgs, abnormal_masks = [], []
        for path in abnormal_paths:
            img_tensor = self._load_ref_image(path)
            mask_path = self._find_abnormal_mask(path)
            if mask_path is None:
                raise FileNotFoundError(
                    f"cannot find abnormal mask for reference image: {path}. "
                )
            abnormal_imgs.append(img_tensor)
            abnormal_masks.append(_load_binary_mask(mask_path, img_tensor.shape[-2:]))

        print(
            f"[refs] product={product} normal={normal_paths} abnormal={abnormal_paths}"
        )
        return (
            torch.stack(normal_imgs),
            torch.stack(normal_masks),
            torch.stack(abnormal_imgs),
            torch.stack(abnormal_masks),
        )

    def get(self, products, device):
        normal_imgs, normal_masks, abnormal_imgs, abnormal_masks = [], [], [], []
        for product in products:
            if product not in self.cache:
                self.cache[product] = self._load_product_refs(product)
            n_img, n_mask, a_img, a_mask = self.cache[product]
            normal_imgs.append(n_img)
            normal_masks.append(n_mask)
            abnormal_imgs.append(a_img)
            abnormal_masks.append(a_mask)
        return (
            torch.stack(normal_imgs).to(device),
            torch.stack(normal_masks).to(device),
            torch.stack(abnormal_imgs).to(device),
            torch.stack(abnormal_masks).to(device),
        )


def _as_product_list(sample_product):
    if isinstance(sample_product, (list, tuple)):
        return list(sample_product)
    return [sample_product]


def _unpack_trace_inputs(data, args, ref_loader=None):
    query = data["query"]
    query_image = query[0].to(args.device)
    query_mask = query[1].squeeze(1).to(args.device)
    image_level_label = data["image_level_label"][0].to(args.device)
    if ref_loader is not None:
        support_n_img, support_n_mask, support_a_img, support_a_mask = ref_loader.get(
            _as_product_list(data["sample_product"]),
            args.device,
        )
        return (
            query_image,
            query_mask,
            image_level_label,
            support_n_img,
            support_n_mask,
            support_a_img,
            support_a_mask,
        )

    support_normal = data["support_normal"]
    support_abnormal = data["support_abnormal"]

    if args.n_shot <= 0 or args.a_shot <= 0:
        raise ValueError("The traced model expects n_shot > 0 and a_shot > 0.")
    if not isinstance(support_normal[0], torch.Tensor) or not isinstance(
        support_abnormal[0], torch.Tensor
    ):
        raise ValueError("Trace inference needs tensor normal/abnormal supports.")

    support_n_img, support_n_mask = support_normal
    support_a_img, support_a_mask = support_abnormal
    return (
        query_image,
        query_mask,
        image_level_label,
        support_n_img.to(args.device),
        support_n_mask.to(args.device),
        support_a_img.to(args.device),
        support_a_mask.to(args.device),
    )


def run_trace_infer(args, model, dataloader, compute_pro=True, ref_loader=None):
    model.eval()
    metric = EvaMetric(
        args,
        dataloader.dataset.products,
        compute_pro=compute_pro,
        pro_num_thresholds=args.pro_num_thresholds,
        apply_smoothing=True,
    )

    for i, data in enumerate(dataloader):
        sample_product = data["sample_product"]
        (
            query_image,
            query_mask,
            image_level_label,
            support_n_img,
            support_n_mask,
            support_a_img,
            support_a_mask,
        ) = _unpack_trace_inputs(data, args, ref_loader)

        image_level_logits, pixel_level_logits = model(
            query_image,
            query_mask,
            image_level_label,
            support_n_img,
            support_n_mask,
            support_a_img,
            support_a_mask,
        )

        metric.update(
            image_level_logits,
            image_level_label,
            pixel_level_logits,
            query_mask,
            sample_product,
        )

        if i % args.print_freq == 0:
            print(f"Iteration on {args.data_target}: {i}")

    metric.get_scores()
    metric.print_metrics()
    return dict(metric.mean_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("IDEAL-Quick Inference", add_help=True)
    parser.add_argument("--data_root", type=str, default="/home/wanghuan/mycode/dataset", help="dataset path")
    parser.add_argument("--data_target", type=str, default="MVTecAD", choices=DATA_TARGETS)
    parser.add_argument("--test_ano_setting", type=str, default="general", 
                        choices=["general", "hard"], help="test anomaly setting: general or hard")
    parser.add_argument("--save_path", type=str, default="./outputs", help="path to save results")
    parser.add_argument("--worker", type=int, default=8, help="number of workers")
    parser.add_argument("--local_rank", type=int, default=0, help="number of cpu threads to use")
    parser.add_argument("--port", type=str, default="1234", help="number of cpu threads to use")
    parser.add_argument("--backbone_name", type=str, default="dinov2_vits14", help="the name of encoder")
    parser.add_argument("--n_shot", type=int, default=1, help="number of normal ref samples")
    parser.add_argument("--a_shot", type=int, default=1, help="number of abnormal ref samples")
    parser.add_argument("--ref_root", type=str, default="./dataset/data/fewshot_both_ref",
        help="fixed support root containing <data_target>/<product>/normal and abnormal")
    parser.add_argument("--epochs", type=int, default=20, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="weight decay")
    parser.add_argument("--scheduler_type", type=str, default="cosine",
        choices=["multistep", "cosine"], help="lr scheduler: multistep or cosine")
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--grad_clip", type=float, default=0.0, help="clipping max norm")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--image_size", type=int, default=448, help="image size")
    parser.add_argument("--print_freq", type=int, default=50, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency")
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu index")
    parser.add_argument("--seed", type=int, default=17, help="random seed")
    parser.add_argument("--save_m", type=bool, default=True, help="save model")
    parser.add_argument("--pro_num_thresholds", type=int, default=50, help="number of thresholds")
    parser.add_argument("--deviation_vectors", type=int, default=45, help="number of deviation vectors")
    parser.add_argument("--nheads", type=int, default=8, help="num of heads")
    parser.add_argument("--topk", type=int, default=12, help="num of k")
    parser.add_argument("--topr", "--top4", dest="topr", type=int, default=4, help="num of r")
    parser.add_argument("--proj_alpha", type=float, default=0.8, help="alpha of proj")
    parser.add_argument("--g_loss_w", type=float, default=1.0, help="weight of dual discri_loss")
    parser.add_argument("--or_loss_w", type=float, default=0.8, help="weight of dual ortho_loss")
    parser.add_argument("--trace_path", type=str, default="./test_ours/trace_mvtec_2_visa.pt", 
                        help="save path of torch.jit.trace model")

    args = parser.parse_args()
    print("IDEAL-Quick Inference:\n")
    print(args, "\n")
    setup_seed(args.seed)

    args.device = torch.device(
        f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    if not os.path.isfile(args.trace_path):
        raise FileNotFoundError(f"trace model not found: {args.trace_path}")

    model = torch.jit.load(args.trace_path, map_location=args.device)
    model.eval()

    transform = get_transform((args.image_size, args.image_size))
    val_data = EpisodeSet(
        data_root=args.data_root,
        data_target=args.data_target,
        split="test",
        shot=[args.n_shot, args.a_shot],
        transform=transform,
        test_ano_setting=args.test_ano_setting,
    )
    val_dataloader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        pin_memory=args.device.type == "cuda",
        num_workers=args.worker,
        shuffle=False,
    )

    ref_loader = None
    if args.ref_root:
        ref_loader = ReferenceSupportLoader(args, val_data, transform)

    with torch.inference_mode():
        val_metrics = run_trace_infer(
            args,
            model,
            val_dataloader,
            compute_pro=True,
            ref_loader=ref_loader,
        )

    print(
        f"======> Results on {args.data_target}:\t|| "
        f"I-AUROC: {val_metrics['i_auroc']:.4f}, "
        f"I-AP: {val_metrics['i_ap']:.4f}, "
        f"P-AUROC: {val_metrics['p_auroc']:.4f}, "
        f"P-PRO: {val_metrics['p_pro']:.4f}"
    )