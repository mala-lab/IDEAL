# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.11
import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from scipy.ndimage import label as cc_label
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import roc_auc_score


def _safe_roc_auc_numpy(y_true, y_score):
    y_true = (np.asarray(y_true).reshape(-1) > 0.5).astype(np.int32)
    y_score = np.asarray(y_score).reshape(-1).astype(np.float64)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _downsample_maps_for_pro(score_maps, gt_maps, max_side=256):
    if score_maps.ndim != 3:
        return score_maps, gt_maps
    _, h, w = score_maps.shape
    m = max(h, w)
    if m <= max_side:
        return score_maps, gt_maps
    z = max_side / m
    zm = zoom(score_maps, (1, z, z), order=1)
    zg = zoom(np.asarray(gt_maps, dtype=np.float64), (1, z, z), order=0)
    return zm, zg


def _subsample_flat_pair(y_true, y_score, max_n=2_000_000, seed=0):
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)
    if y_true.size != y_score.size:
        raise ValueError(
            f"y_true and y_score length mismatch: {y_true.size} vs {y_score.size}"
        )
    n = y_true.size
    if n <= max_n:
        return y_true, y_score
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_n, replace=False)
    return y_true[idx], y_score[idx]


class FSMetric(object):
    def __init__(
        self,
        args,
        products,
        compute_pro=True,
        pro_num_thresholds=50,
        pro_max_fpr=0.3,
        apply_smoothing=True,
        smoothing_sigma=4.0,
    ):
        self.args = args
        self.products = products
        self.compute_pro = compute_pro
        self.pro_num_thresholds = pro_num_thresholds
        self.pro_max_fpr = pro_max_fpr
        self.apply_smoothing = apply_smoothing
        self.smoothing_sigma = smoothing_sigma
        self.metric_names = [
            "i_auroc",
            "i_ap",
            "i_f1_max",
            "p_auroc",
            "p_f1_max",
            "p_pro",
        ]
        self.stat_buffer = {}
        for product in products:
            self.stat_buffer[product] = {}
            self.stat_buffer[product]["i_score"] = []
            self.stat_buffer[product]["i_gt"] = []
            self.stat_buffer[product]["p_score_map"] = []
            self.stat_buffer[product]["p_gt"] = []

    def update(self, i_logits, i_label, p_logits, p_label, products_list):
        for i, product in enumerate(products_list):
            self.stat_buffer[product]["i_score"].append(i_logits[i, 1].item())
            self.stat_buffer[product]["i_gt"].append(i_label[i].item())
            score_map = (
                ((p_logits[i, 1, ...] + 1 - p_logits[i, 0, ...]) / 2)
                .detach()
                .cpu()
                .numpy()
            )
            if self.apply_smoothing:
                score_map = gaussian_filter(score_map, self.smoothing_sigma)
            self.stat_buffer[product]["p_score_map"].append(score_map)
            self.stat_buffer[product]["p_gt"].append(p_label[i].detach().cpu().numpy())

    def extend_from_raw_lists(
        self,
        product: str,
        i_scores: list,
        i_gts: list,
        p_score_maps: list,
        p_gts: list,
    ) -> None:
        for s in i_scores:
            self.stat_buffer[product]["i_score"].append(float(s))
        for g in i_gts:
            self.stat_buffer[product]["i_gt"].append(int(g))
        for sm, gm in zip(p_score_maps, p_gts):
            sm = np.asarray(sm, dtype=np.float32)
            if self.apply_smoothing:
                sm = gaussian_filter(sm, self.smoothing_sigma)
            self.stat_buffer[product]["p_score_map"].append(sm)
            self.stat_buffer[product]["p_gt"].append(np.asarray(gm, dtype=np.float32))

    def get_scores(self):
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self.calculate_scores, product): product
                for product in self.products
            }
            by_product = {futures[future]: future.result() for future in futures}
        results = [by_product[p] for p in self.products]

        self.results = np.array(results, dtype=np.float32)
        mean_results = np.nanmean(self.results, axis=0)
        self.mean_metrics = {
            name: float(mean_results[idx]) for idx, name in enumerate(self.metric_names)
        }
        return self.mean_metrics["i_auroc"], self.mean_metrics["p_auroc"]

    @staticmethod
    def _f1_components(scores, labels):
        scores = scores.astype(np.float64).reshape(-1)
        labels = (labels > 0.5).astype(np.int32).reshape(-1)

        if scores.size == 0:
            return np.array([]), np.array([])

        order = np.argsort(-scores, kind="mergesort")
        scores_sorted = scores[order]
        labels_sorted = labels[order]

        tp_cum = np.cumsum(labels_sorted)
        fp_cum = np.cumsum(1 - labels_sorted)

        change_idx = np.where(np.diff(scores_sorted))[0]
        thresh_idx = np.r_[change_idx, labels_sorted.size - 1]

        tp = tp_cum[thresh_idx].astype(np.float64)
        fp = fp_cum[thresh_idx].astype(np.float64)

        total_pos = float(labels.sum())
        if total_pos <= 0:
            return np.zeros_like(tp), np.zeros_like(tp)

        precision = tp / np.maximum(tp + fp, 1e-12)
        recall = tp / total_pos
        return precision, recall

    def _average_precision(self, scores, labels):
        precision, recall = self._f1_components(scores, labels)
        if precision.size == 0:
            return 0.0
        # Step integral over PR curve
        precision = np.r_[1.0, precision]
        recall = np.r_[0.0, recall]
        return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))

    def _f1_max(self, scores, labels):
        precision, recall = self._f1_components(scores, labels)
        if precision.size == 0:
            return 0.0
        f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
        return float(np.max(f1))

    def _pro_score(self, score_maps, gt_maps, max_fpr=0.3, num_thresholds=50):
        scores = score_maps.reshape(-1).astype(np.float64)
        if scores.size == 0:
            return 0.0

        t_min, t_max = float(scores.min()), float(scores.max())
        if np.isclose(t_min, t_max):
            return 0.0

        gt_maps = gt_maps.astype(bool)
        bg_masks = ~gt_maps
        bg_total = int(bg_masks.sum())
        if bg_total <= 0:
            return 0.0

        region_info = []
        for i in range(score_maps.shape[0]):
            labeled, n_regions = cc_label(gt_maps[i])
            rids = []
            for rid in range(1, n_regions + 1):
                m = labeled == rid
                region_size = int(m.sum())
                if region_size > 0:
                    rids.append((rid, region_size))
            region_info.append((labeled, rids))

        thresholds = np.linspace(t_max, t_min, num_thresholds)
        pro_list, fpr_list = [], []

        for thr in thresholds:
            pro_vals = []
            fp = 0

            for i in range(score_maps.shape[0]):
                pred_i = score_maps[i] >= thr
                fp += np.logical_and(pred_i, bg_masks[i]).sum()

                labeled, rids = region_info[i]
                for rid, region_size in rids:
                    overlap = np.logical_and(pred_i, labeled == rid).sum()
                    pro_vals.append(overlap / region_size)

            pro = float(np.mean(pro_vals)) if len(pro_vals) > 0 else 0.0
            fpr = float(fp / max(bg_total, 1))
            pro_list.append(pro)
            fpr_list.append(fpr)

        pro_arr = np.array(pro_list, dtype=np.float64)
        fpr_arr = np.array(fpr_list, dtype=np.float64)
        order = np.argsort(fpr_arr)
        fpr_arr = fpr_arr[order]
        pro_arr = pro_arr[order]

        valid = fpr_arr <= max_fpr
        if not np.any(valid):
            return 0.0

        fpr_valid = fpr_arr[valid]
        pro_valid = pro_arr[valid]
        if fpr_valid[-1] < max_fpr:
            fpr_valid = np.r_[fpr_valid, max_fpr]
            pro_valid = np.r_[pro_valid, pro_valid[-1]]

        aupro = np.trapz(pro_valid, fpr_valid) / max_fpr
        return float(aupro)

    def calculate_scores(self, product):
        i_score = np.array(self.stat_buffer[product]["i_score"])
        i_gt = np.array(self.stat_buffer[product]["i_gt"])
        if i_score.size == 0 or not self.stat_buffer[product]["p_score_map"]:
            nan6 = (float("nan"),) * 6
            return nan6

        i_auroc = _safe_roc_auc_numpy(i_gt, i_score)
        i_ap = self._average_precision(i_score, i_gt)
        i_f1_max = self._f1_max(i_score, i_gt)

        p_score_map = np.stack(self.stat_buffer[product]["p_score_map"])
        p_gt = np.stack(self.stat_buffer[product]["p_gt"])
        y_flat = p_gt.ravel()
        s_flat = p_score_map.ravel()
        y_s, s_s = _subsample_flat_pair(y_flat, s_flat)
        p_auroc = _safe_roc_auc_numpy(y_s, s_s)
        p_f1_max = self._f1_max(s_s, y_s)
        p_pro = float("nan")
        if self.compute_pro:
            sm, gm = _downsample_maps_for_pro(p_score_map, p_gt)
            p_pro = self._pro_score(
                sm,
                gm,
                max_fpr=self.pro_max_fpr,
                num_thresholds=self.pro_num_thresholds,
            )
        return i_auroc, i_ap, i_f1_max, p_auroc, p_f1_max, p_pro

    def print_metrics(self):
        metrics = self.results
        print(
            f'{"Product":<20} {"I-AUROC":<10} {"I-AP":<10} {"I-F1max":<10} {"P-AUROC":<10} {"P-F1max":<10} {"P-PRO":<10}'
        )
        for i, product in enumerate(self.products):
            print(
                f"{product:<20} {metrics[i][0]:<10.4f} {metrics[i][1]:<10.4f} {metrics[i][2]:<10.4f} {metrics[i][3]:<10.4f} {metrics[i][4]:<10.4f} {metrics[i][5]:<10.4f}"
            )
        metrics_mean = np.nanmean(metrics, axis=0)
        print(
            f'{"Mean":<20} {metrics_mean[0]:<10.4f} {metrics_mean[1]:<10.4f} {metrics_mean[2]:<10.4f} {metrics_mean[3]:<10.4f} {metrics_mean[4]:<10.4f} {metrics_mean[5]:<10.4f}'
        )

    def reset(self):
        self.stat = np.zeros((self.n_class + 1, 3))
