# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.11
import torch
import random
import numpy as np
import cv2
from torchvision import transforms
from scipy.ndimage import gaussian_filter


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataset_info(dataset):
    if dataset == "MVTec":
        objects = [
            "bottle",
            "cable",
            "capsule",
            "carpet",
            "grid",
            "hazelnut",
            "leather",
            "metal_nut",
            "pill",
            "screw",
            "tile",
            "toothbrush",
            "transistor",
            "wood",
            "zipper",
        ]
        object_anomalies = {
            "bottle": ["broken_large", "broken_small", "contamination"],
            "cable": [
                "bent_wire",
                "cable_swap",
                "combined",
                "cut_inner_insulation",
                "cut_outer_insulation",
                "missing_wire",
                "missing_cable",
                "poke_insulation",
            ],
            "capsule": ["crack", "faulty_imprint", "poke", "scratch", "squeeze"],
            "carpet": ["color", "cut", "hole", "metal_contamination", "thread"],
            "grid": ["bent", "broken", "glue", "metal_contamination", "thread"],
            "hazelnut": ["crack", "cut", "hole", "print"],
            "leather": ["color", "cut", "fold", "glue", "poke"],
            "metal_nut": ["bent", "color", "flip", "scratch"],
            "pill": [
                "color",
                "combined",
                "contamination",
                "crack",
                "faulty_imprint",
                "pill_type",
                "scratch",
            ],
            "screw": [
                "manipulated_front",
                "scratch_head",
                "scratch_neck",
                "thread_side",
                "thread_top",
            ],
            "tile": ["crack", "glue_strip", "gray_stroke", "oil", "rough"],
            "toothbrush": ["defective"],
            "transistor": ["bent_lead", "cut_lead", "damaged_case", "misplaced"],
            "wood": ["color", "combined", "hole", "liquid", "scratch"],
            "zipper": [
                "broken_teeth",
                "combined",
                "fabric_border",
                "fabric_interior",
                "rough",
                "split_teeth",
                "squeezed_teeth",
            ],
        }

    elif dataset == "VisA":
        objects = [
            "candle",
            "capsules",
            "cashew",
            "chewinggum",
            "fryum",
            "macaroni1",
            "macaroni2",
            "pcb1",
            "pcb2",
            "pcb3",
            "pcb4",
            "pipe_fryum",
        ]
        object_anomalies = {
            "candle": ["bad"],
            "capsules": ["bad"],
            "cashew": ["bad"],
            "chewinggum": ["bad"],
            "fryum": ["bad"],
            "macaroni1": ["bad"],
            "macaroni2": ["bad"],
            "pcb1": ["bad"],
            "pcb2": ["bad"],
            "pcb3": ["bad"],
            "pcb4": ["bad"],
            "pipe_fryum": ["bad"],
        }

    elif dataset == "BTAD":
        objects = ["01", "02", "03"]
        object_anomalies = {
            "01": ["ko"],
            "02": ["ko"],
            "03": ["ko"],
        }

    elif dataset == "BraTS":
        objects = ["brain"]
        object_anomalies = {"brain": ["lesion"]}
    else:
        raise ValueError(f"Dataset '{dataset}' not yet covered!")

    return objects, object_anomalies


def dists2map(dists, img_shape):
    # resize and smooth the distance map
    dists = cv2.resize(
        dists, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_LINEAR
    )
    dists = gaussian_filter(dists, sigma=4)
    return dists


def get_transform(image_size):
    return transforms.Compose(
        [
            transforms.Resize(
                size=image_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def get_crop_transform(img_size, crop_size):
    return transforms.Compose(
        [
            transforms.Resize(
                size=(img_size, img_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def get_strong_transforms(size, isize):
    mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train
    std_train = [0.229, 0.224, 0.225] if std_train is None else std_train
    data_transforms = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.RandomResizedCrop((isize, isize), scale=(0.6, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return data_transforms


from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


class WarmCosineScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer,
        base_value,
        final_value,
        total_iters,
        warmup_iters=0,
        start_warmup_value=0,
    ):
        self.final_value = final_value
        self.total_iters = total_iters
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(total_iters - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (
            1 + np.cos(np.pi * iters / len(iters))
        )
        self.schedule = np.concatenate((warmup_schedule, schedule))

        super(WarmCosineScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch >= self.total_iters:
            return [self.final_value for base_lr in self.base_lrs]
        else:
            return [self.schedule[self.last_epoch] for base_lr in self.base_lrs]
