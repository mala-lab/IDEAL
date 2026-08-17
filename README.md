<div align="center">
  <h2><b>IDEAL: Intrinsic Deviation Learning for Discriminative FSAD</b></h2>
</div>

<div align="center">

![](https://img.shields.io/github/last-commit/mala-lab/IDEAL?color=blue)
![](https://img.shields.io/github/stars/mala-lab/IDEAL?color=yellow)
![](https://img.shields.io/github/forks/mala-lab/IDEAL?color=lightblue&label=Forks)
![](https://img.shields.io/badge/PRs-Welcome-green)
[![arXiv](https://img.shields.io/badge/arXiv-2605.23231-b31b1b)](https://arxiv.org/abs/2605.23231)

</div>

Official implementation of paper [Beyond Normal References: Discriminative Few-Shot Anomaly Detection](https://arxiv.org/abs/2605.23231).

## 🔎 Overview

This paper considers a practical few-shot anomaly detection (FSAD) setting, termed **discriminative FSAD**, where a limited number of both normal and anomalous examples are available as references during inference.
Existing FSAD methods rely on normal-only references through normality matching, ignoring the discriminative clues in anomalous references, while directly fitting both references can overfit to the seen anomalies.
We introduce **IDEAL**, an **i**ntrinsic **de**vi**a**tion **l**earning framework that leverages both reference types to learn intrinsic deviation patterns characterizing generalizable abnormality as deviations from normality.
IDEAL decomposes the learning process into two novel components: 1) a Normal Variation Eraser (NVE) to suppress nuisance normal variations that may lead to noisy deviations from normality, thereby highlighting anomaly-relevant deviation representations; 2) an Intrinsic Deviation Encoder (IDE) to decompose these denoised deviation representations into intrinsic deviation vectors capturing the most discriminative orthogonal deviation directions. At inference, IDEAL scores query-to-normal deviations preserved after projection onto the learned intrinsic deviation vectors, enabling generalization for both seen and unseen anomalies. The framework diagram of the proposed IDEAL is shown below:

![image](./figs/IDEAL_overview.png)

## ⚙️ Setup Libraries

- python >= 3.10.11
- torch == 2.4.1
- torchvision == 0.19.1
- scipy == 1.7.3
- scikit-image == 0.19.2
- numpy >= 1.24.3
- tqdm >= 4.64.0
- transformers == 4.31.0

## 🚀 Prepare Anomaly Detection Datasets and Weights

#### Step 1. Download Anomaly Detection Datasets

- **Industrial Anomaly Detection Datasets**: [MVTecAD](https://www.mvtec.com/company/research/datasets/mvtec-ad), [VisA](https://github.com/amazon-science/spot-diff), [AITEX](https://www.aitex.es/afid/), [BTAD](http://avires.dimi.uniud.it/papers/btad/btad.zip), [MPDD](https://github.com/stepanje/MPDD).
- **Medical Anomaly Detection Datasets**: [BraTS](https://www.kaggle.com/datasets/dschettler8854/brats-2021-task1), [Liver](https://drive.google.com/drive/folders/1AC-wWZl_K18CWL2eIxUScoSOoxT4IBuw?usp=sharing) (from [BMAD](https://arxiv.org/abs/2306.11876) benchmark), [RESC](https://drive.google.com/drive/folders/1AC-wWZl_K18CWL2eIxUScoSOoxT4IBuw?usp=sharing) (from [BMAD](https://arxiv.org/abs/2306.11876) benchmark).
- Please put these anomaly detection datasets in the `./dataset/data/` directory.

#### Step 2. Download Few-Shot Reference Samples

- **Normal Few-Shot Reference Samples**: The few-shot reference normal samples are used for normal-based FSAD methods at inference. Please download the few-shot normal reference samples from [Google Drive](https://drive.google.com/file/d/1H0vTqzZHeSOTMEnadKls22VnjLqF7V3h/view?usp=sharing) and put these data samples in the `./dataset/data/` directory.
- **Normal and Abnormal Few-Shot Reference Samples**: Please download the few-shot normal and abnormal reference samples from [Google Drive](https://drive.google.com/file/d/1hCqqh5Q4il5p4Um0zEIpEkQxEjwjsoeh/view?usp=sharing) and put these data samples in the `./dataset/data/` directory.

#### Step 3. Creating Json File for Each Dataset

- **Dataset Json File**: Please run the following code for generating json file for each dataset (taking the MVTecAD dataset as an example):
  ```python
  python3 dataset/gen_mvtec_json.py --data_path ./dataset/data/MVTecAD
  ```

#### Step 4. Download Pre-train Weights for Inference

- **Download Pre-train Weights**: Please download the pre-train models from [Google Drive](https://drive.google.com/drive/folders/1xSObxwEmxU7WBCJ7kgvNywOzT3h5bPp_?usp=sharing).

## 📊 Run Experiments

#### ✅ Quick Inference

- updating the checkpoint path and reference path (e.g., train on MVTecAD and test on VisA):
  ```python
  python3 -u main_infer.py \
    --data_root ./dataset/data \
    --data_target VisA \
    --test_ano_setting general \
    --trace_path ./trace_n1a1_mvtec_2_visa.pt \
    --ref_root ./dataset/data/fewshot_both_ref \
    --gpu_id 0 --n_shot 1 --a_shot 1 \
    > ./outputs/infer_VisA_n1a1_mvtec_2_visa.log 2>&1
  ```

#### ✅ Training Repository

- updating the dataset path (i.e., `data_root` in train.sh):

  ```python
  bash train_v2m.sh  # v2m = train on VisA and test on MVTecAD
  ```

- we thank for the code repository of [InCTRL](https://github.com/mala-lab/InCTRL), [NAGL](https://github.com/JasonKyng/NAGL), [WinCLIP](https://github.com/mala-lab/WinCLIP), and [Dinomaly](https://github.com/guojiajeremy/dinomaly).

## 📚 Citation

- If you would like to discuss any details about this work, please feel free to email me (huanwang1018@gmail.com) or open a GitHub issue (email is usually replied faster, sorry for any delay).

- If you find this paper and repository useful, please cite our paper:
  ```bibtex
  @article{wang2026beyond,
    title={Beyond Normal References: Discriminative Few-Shot Anomaly Detection},
    author={Wang, Huan and Shen, Jun and Yan, Jun and Pang, Guansong},
    journal={arXiv preprint arXiv:2605.23231},
    year={2026}
  }
  ```
