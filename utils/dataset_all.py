# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.11
import os
import json
import random
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import PIL.Image as Image
import numpy as np


class FSDataset(Dataset):
    def __init__(
        self,
        data_root: str = "/home/wanghuan/dataset",
        data_target: str = "MVTecAD",
        data_name_json: str = "meta.json",
        split: str = "train",  # "train" or "test"
        shot: list = [4, 1],  # 4 normal samples, 1 abnormal sample
        transform=None,
        choice=500,
        test_ano_setting="general",  # general or hard
        set_class=None,
        target_to_target: bool = False,
        select_ref_by_query: bool = False,
        test_ref_object=None,
        test_ref_anomaly_type=None,
        a_ref_types: int = 1,
    ):
        self.split = split  # "train" or "test"
        self.data_root = data_root
        self.set_class = set_class
        self.target_to_target = target_to_target
        self.select_ref_by_query = select_ref_by_query
        self.test_ref_object = test_ref_object
        self.test_ref_anomaly_type = test_ref_anomaly_type
        self.a_ref_types = a_ref_types
        if self.target_to_target:
            source_set = data_target
            target_set = data_target
            data = [target_set]
            self.test_ano_setting = test_ano_setting
        elif data_target == "MVTecAD":
            source_set = "VisA"
            target_set = "MVTecAD"
            data = [target_set, source_set]
            # source from visa, only "general" setting is available
            self.test_ano_setting = test_ano_setting
        else:  # mvtec -> others
            source_set = "MVTecAD"
            target_set = data_target
            data = [source_set, target_set]
            # source from mvtec, "general" or "hard" setting is available
            self.test_ano_setting = test_ano_setting

        self.source_set = source_set
        self.target_set = target_set
        self.data_name_json = data_name_json
        self.choice = choice
        self.n_shot, self.a_shot = shot
        self.transform = transform
        self.initialize(data)
        self.class_ids = self.build_class_ids()
        self._validate_a_ref_types()
        self._validate_explicit_test_reference()
        self.data_path_list = None
        if self.split == "test":
            self._ensure_all_test_schedules()

    def initialize(self, data):
        self.metadata = {}
        self.all_product = []
        for d in data:
            data_path = os.path.join(self.data_root, d)
            with open(os.path.join(data_path, self.data_name_json)) as f:
                data_info = json.load(f)
            data_info = data_info["test"]
            products = list(data_info.keys())
            products.sort()  # class_names
            self.all_product += products

            for product in products:
                self.metadata.setdefault(product, {"normal": {}, "abnormal": {}})
                for sample in data_info[product]:
                    sample_info = {
                        "img_path": os.path.join(data_path, sample["img_path"]),
                        "mask_path": os.path.join(data_path, sample["mask_path"]),
                        "anomaly": sample["anomaly"],
                    }
                    if sample["anomaly"]:
                        product_type = "abnormal"
                        if sample["specie_name"] == "":
                            sample["specie_name"] = "bad"
                    else:
                        product_type = "normal"
                        if sample["specie_name"] == "":
                            sample["specie_name"] = "good"
                    self.metadata[product][product_type].setdefault(
                        sample["specie_name"], []
                    ).append(sample_info)

        self.nclass = len(self.all_product)

    def __len__(self):
        if self.split == "test":
            self._ensure_all_test_schedules()
            if self.set_class is not None:
                return len(self.test_query_schedule[self.set_class])
            return sum(len(self.test_query_schedule[p]) for p in self.products)
        return self.choice

    def mask_list_transform(self, mask_list, img_size):
        mask_tensor = []
        for mask in mask_list:
            mask_tensor.append(
                F.interpolate(
                    mask.unsqueeze(0).unsqueeze(0).float(), img_size, mode="nearest"
                ).squeeze()
            )
        return torch.stack(mask_tensor)

    def image_list_transform(self, image_list):
        image_tensor = []
        for image in image_list:
            image_tensor.append(self.transform(image))
        return torch.stack(image_tensor)

    def __getitem__(self, idx):
        if self.split == "test":
            (
                query_tuple,
                support_normal_tuple,
                support_abnormal_tuple,
                sample_product,
                query_type,
                query_specie_name,
                ref_abnormal_specie_name,
            ) = self.load_frame_test(idx)
        else:
            (
                query_tuple,
                support_normal_tuple,
                support_abnormal_tuple,
                sample_product,
            ) = self.load_frame()
            # default_collate
            query_type = "na"
            query_specie_name = "na"
            ref_abnormal_specie_name = "na"

        query_image = self.image_list_transform(query_tuple[0])  # [1, C, H, W]
        query_mask = self.mask_list_transform(
            query_tuple[1], query_image.shape[-2:]
        )  # [1, H, W]
        query_data = [query_image, query_mask]

        support_normal_data = [0, 0]
        support_abnormal_data = [0, 0]

        if self.n_shot:
            support_normal_imgs = self.image_list_transform(support_normal_tuple[0])
            support_normal_masks = self.mask_list_transform(
                support_normal_tuple[1], support_normal_imgs.size()[-2:]
            )
            support_normal_data = [support_normal_imgs, support_normal_masks]

        if self.a_shot:
            support_abnormal_imgs = self.image_list_transform(support_abnormal_tuple[0])
            support_abnormal_masks = self.mask_list_transform(
                support_abnormal_tuple[1], support_abnormal_imgs.size()[-2:]
            )
            support_abnormal_data = [support_abnormal_imgs, support_abnormal_masks]

        group_dict = {
            "query": query_data,
            "image_level_label": query_tuple[2],
            "support_normal": support_normal_data,
            "support_abnormal": support_abnormal_data,
            "sample_product": sample_product,  # class_name
            "query_type": query_type,
            "query_specie_name": query_specie_name,
            "ref_abnormal_specie_name": ref_abnormal_specie_name,
            "data_path": getattr(self, "query_path_list", self.data_path_list),
        }
        return group_dict

    def build_class_ids(self):
        if self.target_to_target:
            class_ids_trn = range(0, self.nclass)
            class_ids_val = range(0, self.nclass)
            class_trn = [self.all_product[i] for i in class_ids_trn]
            class_val = [self.all_product[i] for i in class_ids_val]
            class_ids = class_ids_trn if self.split == "train" else class_ids_val
            msg = (
                f"Train classes: {class_trn}"
                if self.split == "train"
                else f"Val classes: {class_val} \n"
            )
            print(msg)
            self.products = class_trn if self.split == "train" else class_val
            self.test_support_normal = {x: None for x in class_val}
            self.test_support_abnormal = {x: None for x in class_val}
            self.test_support_abnormal_names = {x: None for x in class_val}
            self.test_support_abnormal_types = {x: set() for x in class_val}
            self.test_normal_idx = {x: None for x in class_val}
            self.test_abnormal_idx = {x: None for x in class_val}
            self.test_query_schedule = {}
            return class_ids

        if self.target_set == "MVTecAD":  # visa -> mvtec
            class_ids_val = range(0, 15)  # mvtec classes
            class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]
        else:  # mvtec -> others
            class_ids_trn = range(0, 15)  # mvtec classes
            class_ids_val = [x for x in range(self.nclass) if x not in class_ids_trn]

        class_trn = [self.all_product[i] for i in class_ids_trn]
        class_val = [self.all_product[i] for i in class_ids_val]
        class_ids = class_ids_trn if self.split == "train" else class_ids_val

        msg = (
            f"Train classes: {class_trn}"
            if self.split == "train"
            else f"Val classes: {class_val} \n"
        )
        print(msg)

        self.products = class_trn if self.split == "train" else class_val
        self.test_support_normal = {x: None for x in class_val}
        self.test_support_abnormal = {x: None for x in class_val}
        self.test_support_abnormal_names = {x: None for x in class_val}
        self.test_support_abnormal_types = {x: set() for x in class_val}
        self.test_normal_idx = {x: None for x in class_val}
        self.test_abnormal_idx = {x: None for x in class_val}
        self.test_query_schedule = {}

        return class_ids

    def random_sample(self, sample_list, num):
        n = len(sample_list)
        if n == 0:
            raise ValueError("random_sample: empty sample_list.")
        if num <= n:
            selected_idx = random.sample(range(n), num)
        else:
            selected_idx = random.choices(range(n), k=num)
        selected_sample = [sample_list[i] for i in selected_idx]
        return selected_idx, selected_sample

    def _validate_a_ref_types(self):
        a_ref_types = getattr(self, "a_ref_types", 1)
        if a_ref_types < 1:
            raise ValueError("a_ref_types must be at least 1.")
        if self.a_shot <= 0:
            if a_ref_types != 1:
                raise ValueError("a_ref_types must be 1 when a_shot=0.")
            return
        if a_ref_types > self.a_shot:
            raise ValueError(
                f"a_ref_types ({a_ref_types}) cannot exceed a_shot ({self.a_shot})."
            )
        if a_ref_types > 1 and self.select_ref_by_query:
            raise ValueError("a_ref_types>1 requires select_ref_by_query=False.")

    def _validate_explicit_test_reference(self):
        object_is_set = self.test_ref_object is not None
        type_is_set = self.test_ref_anomaly_type is not None
        if not object_is_set and not type_is_set:
            return
        if object_is_set != type_is_set:
            raise ValueError(
                "test_ref_object and test_ref_anomaly_type must be provided together."
            )
        if self.split != "test":
            raise ValueError("Explicit test reference selection requires split='test'.")
        if self.a_shot <= 0:
            raise ValueError("Explicit test reference selection requires a_shot > 0.")
        if self.select_ref_by_query:
            raise ValueError(
                "Explicit test reference selection requires select_ref_by_query=False."
            )
        if getattr(self, "a_ref_types", 1) != 1:
            raise ValueError(
                "Explicit test reference selection requires a_ref_types=1."
            )
        if self.test_ref_object not in self.products:
            raise ValueError(
                f"Unknown test_ref_object={self.test_ref_object!r}. "
                f"Available test objects: {sorted(self.products)}"
            )
        available_types = sorted(self.metadata[self.test_ref_object]["abnormal"].keys())
        if self.test_ref_anomaly_type not in available_types:
            raise ValueError(
                f"Unknown test_ref_anomaly_type={self.test_ref_anomaly_type!r} "
                f"for object {self.test_ref_object!r}. "
                f"Available anomaly types: {available_types}"
            )

    def get_sample_info(self, sample_dict):
        img_path = sample_dict["img_path"]
        mask_path = sample_dict["mask_path"]
        anomaly = sample_dict["anomaly"]
        return img_path, mask_path, anomaly

    def read_mask(self, anomaly, mask_path, imsize):
        if not anomaly:
            mask = torch.zeros(imsize)
        else:
            mask = torch.tensor(np.array(Image.open(mask_path).convert("L")))
            mask[mask > 0.5] = 1
        return mask

    def read_data(self, data_list):
        image_list = []
        mask_list = []
        anomaly_list = []
        data_path_list = []
        for data in data_list:
            data_path, mask_path, anomaly = self.get_sample_info(data)
            data_path_list.append(data_path)
            image_list.append(Image.open(data_path).convert("RGB"))
            mask_list.append(self.read_mask(anomaly, mask_path, image_list[-1].size))
            anomaly_list.append(anomaly)
        self.data_path_list = data_path_list
        return image_list, mask_list, anomaly_list

    def _sample_abnormal_support_same_type(self, product):
        abnormal_keys = list(self.metadata[product]["abnormal"].keys())
        if self.a_shot <= 0:
            return [], [], abnormal_keys[0] if abnormal_keys else "bad"
        if product == getattr(self, "test_ref_object", None):
            support_abnormal_specie_type = self.test_ref_anomaly_type
        else:
            ok_a = [
                k
                for k in abnormal_keys
                if len(self.metadata[product]["abnormal"][k]) >= self.a_shot
            ]
            support_abnormal_specie_type = (
                np.random.choice(ok_a, 1)[0]
                if ok_a
                else np.random.choice(abnormal_keys, 1)[0]
            )
        pool = self.metadata[product]["abnormal"][support_abnormal_specie_type]
        support_abnormal_idx, support_abnormal = self.random_sample(pool, self.a_shot)
        pool_paths = {x["img_path"] for x in pool}
        for s in support_abnormal:
            if s["img_path"] not in pool_paths:
                raise RuntimeError("abnormal support not the same abnormal-type")
        return support_abnormal_idx, support_abnormal, support_abnormal_specie_type

    def _sample_abnormal_support_for_query(
        self, product, query_specie_name, query_sample
    ):
        pool = self.metadata[product]["abnormal"][query_specie_name]
        candidate_pool = [
            sample for sample in pool if sample["img_path"] != query_sample["img_path"]
        ]
        if not candidate_pool:
            candidate_pool = pool
        _, support_abnormal = self.random_sample(candidate_pool, self.a_shot)
        return support_abnormal, query_specie_name

    def _sample_test_abnormal_support(self, product):
        if getattr(self, "a_ref_types", 1) == 1:
            indices, samples, anomaly_type = self._sample_abnormal_support_same_type(
                product
            )
            selected_types = [anomaly_type] if self.a_shot > 0 else []
            return indices, samples, selected_types

        abnormal_keys = list(self.metadata[product]["abnormal"].keys())
        selected_type_count = min(self.a_ref_types, len(abnormal_keys))
        selected_types = [
            str(anomaly_type)
            for anomaly_type in np.random.choice(
                abnormal_keys,
                selected_type_count,
                replace=False,
            )
        ]
        reference_types = selected_types.copy()
        for _ in range(self.a_shot - selected_type_count):
            reference_types.append(str(np.random.choice(selected_types, 1)[0]))

        support_abnormal_idx = []
        support_abnormal = []
        for anomaly_type in selected_types:
            type_count = reference_types.count(anomaly_type)
            type_indices, type_samples = self.random_sample(
                self.metadata[product]["abnormal"][anomaly_type],
                type_count,
            )
            support_abnormal_idx.extend(type_indices)
            support_abnormal.extend(type_samples)

        return support_abnormal_idx, support_abnormal, selected_types

    def _sample_abnormal_support_mixed_types(self, product):
        abnormal_keys = list(self.metadata[product]["abnormal"].keys())
        if self.a_shot <= 0:
            return [], [], abnormal_keys[0] if abnormal_keys else "bad"
        support_abnormal = []
        support_abnormal_idx = []
        for _ in range(self.a_shot):
            t = np.random.choice(abnormal_keys, 1)[0]
            one_idx, one_samp = self.random_sample(
                self.metadata[product]["abnormal"][t], 1
            )
            support_abnormal_idx.append(one_idx[0])
            support_abnormal.append(one_samp[0])
        return support_abnormal_idx, support_abnormal, None

    def _init_test_support_once(self, product):
        if self.test_support_normal[product] is not None:
            return
        normal_keys = list(self.metadata[product]["normal"].keys())
        if self.n_shot > 0:
            ok_n = [
                k
                for k in normal_keys
                if len(self.metadata[product]["normal"][k]) >= self.n_shot
            ]
            support_normal_specie_type = (
                np.random.choice(ok_n, 1)[0]
                if ok_n
                else np.random.choice(normal_keys, 1)[0]
            )
        else:
            support_normal_specie_type = normal_keys[0]
        support_normal_idx, support_normal = self.random_sample(
            self.metadata[product]["normal"][support_normal_specie_type],
            self.n_shot,
        )
        support_abnormal_idx, support_abnormal, support_abnormal_specie_types = (
            self._sample_test_abnormal_support(product)
        )
        self.test_support_normal[product] = support_normal
        self.test_normal_idx[product] = support_normal_idx
        self.test_support_abnormal[product] = support_abnormal
        self.test_support_abnormal_types[product] = set(support_abnormal_specie_types)
        self.test_support_abnormal_names[product] = "|".join(
            support_abnormal_specie_types
        )
        self.test_abnormal_idx[product] = support_abnormal_idx
        if self.split == "test" and self.n_shot > 0:
            n_paths = [s["img_path"] for s in support_normal]
            print(
                f"[FSDataset test] product={product} "
                f"normal_support_specie={support_normal_specie_type} "
                f"n_shot={self.n_shot} indices_in_specie={support_normal_idx} "
                f"paths={n_paths}"
            )
        if self.split == "test" and self.a_shot > 0:
            ab_paths = [s["img_path"] for s in support_abnormal]
            print(
                f"[FSDataset test] product={product} "
                f"abnormal_support_species={support_abnormal_specie_types} "
                f"a_shot={self.a_shot} indices_in_specie={support_abnormal_idx} "
                f"paths={ab_paths}"
            )

    def _build_test_query_schedule(self, product):
        meta = self.metadata[product]
        fixed_n = self.test_support_normal[product] or []
        fixed_a = self.test_support_abnormal[product] or []
        support_paths = {s["img_path"] for s in fixed_n + fixed_a}
        ref_ab_name = self.test_support_abnormal_names[product]
        ref_ab_types = getattr(self, "test_support_abnormal_types", {}).get(product)
        if not ref_ab_types:
            ref_ab_types = set(ref_ab_name.split("|")) if ref_ab_name else set()
        schedule = []
        for qtype in ("normal", "abnormal"):
            for specie in sorted(meta[qtype].keys()):
                if (
                    qtype == "abnormal"
                    and self.test_ano_setting == "hard"
                    and not self.select_ref_by_query
                    and specie in ref_ab_types
                ):
                    continue
                for i, sample in enumerate(meta[qtype][specie]):
                    if sample["img_path"] in support_paths:
                        continue
                    schedule.append((qtype, specie, i))
        return schedule

    def _ensure_test_support_and_schedule(self, product):
        if product in self.test_query_schedule:
            return
        self._init_test_support_once(product)
        self.test_query_schedule[product] = self._build_test_query_schedule(product)

    def _ensure_all_test_schedules(self):
        prods = [self.set_class] if self.set_class is not None else list(self.products)
        for p in prods:
            self._ensure_test_support_and_schedule(p)

    def load_frame_test(self, idx):
        self._ensure_all_test_schedules()
        if self.set_class is not None:
            product = self.set_class
            spec = self.test_query_schedule[product]
            if idx < 0 or idx >= len(spec):
                raise IndexError(
                    f"Test index {idx} out of range [0, {len(spec)}) for {product}"
                )
            qtype, qspecie, qi = spec[idx]
        else:
            offset = 0
            product = None
            qtype, qspecie, qi = None, None, None
            for p in self.products:
                spec = self.test_query_schedule[p]
                n = len(spec)
                if idx < offset + n:
                    product = p
                    qtype, qspecie, qi = spec[idx - offset]
                    break
                offset += n
            if product is None:
                raise IndexError(
                    f"Test index {idx} out of range (total {self.__len__()})"
                )
        query_data = [self.metadata[product][qtype][qspecie][qi]]
        new_fixed_normal = self.test_support_normal[product]
        new_fixed_abnormal = self.test_support_abnormal[product]
        ref_abnormal_specie_name = self.test_support_abnormal_names[product]
        if self.select_ref_by_query and qtype == "abnormal" and self.a_shot > 0:
            new_fixed_abnormal, ref_abnormal_specie_name = (
                self._sample_abnormal_support_for_query(product, qspecie, query_data[0])
            )
        query_tuple = self.read_data(query_data)
        self.query_path_list = self.data_path_list.copy()
        support_normal_tuple = (0, 0, 0)
        if self.n_shot > 0:
            support_normal_tuple = self.read_data(new_fixed_normal)
        support_abnormal_tuple = (0, 0, 0)
        if self.a_shot > 0:
            support_abnormal_tuple = self.read_data(new_fixed_abnormal)
        return (
            query_tuple,
            support_normal_tuple,
            support_abnormal_tuple,
            product,
            qtype,
            qspecie,
            ref_abnormal_specie_name,
        )

    def load_frame(self):
        assert self.split == "train", "load_frame only for training"
        if self.set_class is not None:
            sample_product = self.set_class
        else:
            sample_product_idx = np.random.choice(self.class_ids, 1)[
                0
            ]  # random sample a class
            sample_product = self.all_product[sample_product_idx]

        # sample normal support
        normal_keys = list(self.metadata[sample_product]["normal"].keys())
        if self.n_shot > 0:
            ok_n = [
                k
                for k in normal_keys
                if len(self.metadata[sample_product]["normal"][k]) >= self.n_shot
            ]
            support_normal_specie_type = (
                np.random.choice(ok_n, 1)[0]
                if ok_n
                else np.random.choice(normal_keys, 1)[0]
            )
        else:
            support_normal_specie_type = normal_keys[0]
        support_normal_idx, support_normal = self.random_sample(
            self.metadata[sample_product]["normal"][support_normal_specie_type],
            self.n_shot,
        )
        # Get normal support images
        support_normal_tuple = (0, 0, 0)
        if self.n_shot > 0:
            support_normal_tuple = self.read_data(support_normal)

        # sample abnormal support
        _, support_abnormal, _ = self._sample_abnormal_support_mixed_types(
            sample_product
        )
        support_abnormal_paths = {s["img_path"] for s in support_abnormal}
        abnormal_keys = list(self.metadata[sample_product]["abnormal"].keys())
        # Get abnormal support images
        support_abnormal_tuple = (0, 0, 0)
        if self.a_shot > 0:
            support_abnormal_tuple = self.read_data(support_abnormal)

        query_type = np.random.choice(["normal", "abnormal"], 1)[0]

        if query_type == "normal":
            query_specie_type = support_normal_specie_type
        else:
            query_specie_type = np.random.choice(abnormal_keys, 1)[0]
        query_idx, query_data = self.random_sample(
            self.metadata[sample_product][query_type][query_specie_type], 1
        )
        # check if query image is in support set
        q_path = query_data[0]["img_path"]
        if query_type == "normal" and query_idx[0] in support_normal_idx:
            self.load_frame()
        elif query_type == "abnormal" and q_path in support_abnormal_paths:
            self.load_frame()
        # Get query image
        query_tuple = self.read_data(query_data)
        self.query_path_list = self.data_path_list.copy()

        return (
            query_tuple,
            support_normal_tuple,
            support_abnormal_tuple,
            sample_product,
        )
