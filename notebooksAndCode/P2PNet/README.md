# P2PNet for Winter-Snow Crowd Counting Fine-Tuning

This repository extends the official **P2PNet** implementation for a specialized **winter-snow crowd counting scenario**. The original P2PNet project is a purely point-based framework for crowd counting and localization built on a VGG16 backbone with an FPN-style decoder and prediction heads. Our work adapts that pipeline to outdoor broomball / winter scenes, where the visual domain is quite different from standard crowd-counting benchmarks because of snow-covered backgrounds, strong brightness, small or sparse human targets, and unusual viewpoints.

Compared with the original repository, our project focuses on **custom dataset preparation, transfer learning, and evaluation for a domain-specific winter dataset**. In particular, we add:

- `train_finetune.py` for fine-tuning pretrained P2PNet weights on our custom winter-snow dataset
- `train_eval_only.py` for evaluation-only testing on the held-out split without entering the training loop

These additions are built on top of the original P2PNet training and testing workflow, which uses list files, point annotations in text format, and a VGG16-based point prediction architecture. 

---

## Project Goal

The goal of this project is to adapt P2PNet to **specialized winter-snow scenarios** for people counting. Instead of relying only on the original pretrained model, we fine-tune it on our own annotated images collected from snowy outdoor scenes.

This is important because winter-snow images present several domain-specific challenges:

- strong white snow background and reduced contrast
- different crowd density from public crowd datasets
- outdoor lighting variation and reflections on snow/ice
- small human heads due to distant camera viewpoints
- scene layout different from common benchmark datasets

To address this, we use **transfer learning**: we start from the original pretrained P2PNet model and fine-tune it on our custom annotations.

---

## What We Changed from the Original P2PNet Repository

The original repository provides:

- the official P2PNet implementation
- training via `train.py`
- visualization testing via `run_test.py`
- a dataset format based on image/annotation list files and point annotations in `.txt` format

Our extension adds a domain-adaptation workflow for winter scenes:

### Added scripts

#### `train_finetune.py`
A fine-tuning script for our custom winter-snow dataset.

Main purpose:
- load pretrained P2PNet weights
- support transfer learning for a custom dataset
- allow freezing part of the pretrained model and updating only selected modules
- train on our specialized winter scenario rather than the default benchmark setup

Typical transfer learning strategies we explored include:
- freeze backbone, train FPN + heads
- freeze backbone + FPN, train heads only
- optionally compare with fuller fine-tuning settings

#### `train_eval_only.py`
An evaluation-only script for held-out data.

Main purpose:
- load a saved checkpoint
- run only the evaluation stage
- compute performance on the validation or test split
- avoid entering the normal training loop in `train.py`

This is especially useful for fair comparison between different fine-tuned models.

---

## Model Overview

P2PNet is a **point-based crowd counting model**.

Its main workflow is:

1. **Backbone feature extraction** using pretrained VGG16
2. **Multi-scale feature fusion** using an FPN-style decoder
3. **Prediction heads**:
   - classification head for point confidence
   - regression head for point location
4. **Point-based output**, where the final count is obtained from predicted head points

Compared with density-map-based methods, P2PNet directly predicts person head points, which makes it suitable for both:
- **crowd counting**
- **approximate localization**

---

## Our Fine-Tuning Focus for Winter-Snow Scenarios

Our main contribution is not changing the P2PNet architecture itself, but rather **adapting it to a domain-specific outdoor winter dataset**.

### Why fine-tuning is needed

The original pretrained P2PNet model is trained on standard crowd-counting datasets. However, our winter-snow images differ substantially in appearance and scene structure. Therefore, direct use of the original model may not generalize well enough.

### Our transfer learning idea

We fine-tune the model with a focus on **specialized adaptation while reducing overfitting**.

A representative strategy is:
- **freeze the backbone** to preserve general visual features learned from pretraining
- **train the FPN and prediction heads** so the model can better adapt to winter scene layout, human scale, and annotation style

This strategy is a good compromise because:
- the backbone captures general CNN features
- the FPN helps adapt multi-scale features to our new scene distribution
- the classification and regression heads are the most task-specific layers for point prediction

---

## Dataset Preparation

We organize the custom dataset in a P2PNet-compatible format. The original repository uses list files and one `.txt` annotation file per image, where each line contains one point coordinate. 

### Annotation format

Each annotation file contains one point per line:

```text
x1 y1
x2 y2
x3 y3
...
```

### Dataset structure

```text
DATA_ROOT/
├── train/
│   └── scene01/
├── val/
│   └── scene01/
├── test/
│   └── scene01/
├── train.list
├── val.list
└── test.list
```

### Example list-file entry

```text
train/scene01/frame_0001.png train/scene01/frame_0001.txt
```

In our project, we also prepared utility scripts to:
- convert labeled coordinates to `.npy`
- convert `.npy` annotations to P2PNet `.txt` annotations
- generate `train.list`, `val.list`, and `test.list`
- support reproducible dataset splitting

---

## Training for Fine-Tuning

A typical fine-tuning command is:

```bash
CUDA_VISIBLE_DEVICES=0 python train_finetune.py \
    --data_root $DATA_ROOT \
    --dataset_file SHHA \
    --pretrained /path/to/pretrained_checkpoint.pth \
    --output_dir ./logs \
    --checkpoints_dir ./weights \
    --tensorboard_dir ./logs \
    --lr 1e-4 \
    --batch_size 4 \
    --eval_freq 1 \
    --gpu_id 0
```

Depending on the version of the script, optional arguments may include:
- `--freeze_backbone`
- `--freeze_fpn`
- `--train_heads_only`

These options allow different transfer learning settings for ablation and comparison.

---

## Evaluation Only

To evaluate a trained checkpoint on the held-out set only:

```bash
CUDA_VISIBLE_DEVICES=0 python train_eval_only.py \
    --data_root $DATA_ROOT \
    --dataset_file SHHA \
    --resume /path/to/checkpoint.pth \
    --split test \
    --output_dir ./eval_logs
```

This script is intended for:
- test-only evaluation
- comparing checkpoints fairly
- reporting MAE / RMSE on the held-out split
- optionally saving visualization results

---

## Suggested Workflow for This Project

1. Annotate head points on winter-snow images
2. Convert annotations into P2PNet-compatible `.txt` files
3. Split the dataset into `train / val / test`
4. Generate `train.list`, `val.list`, and `test.list`
5. Load pretrained P2PNet weights
6. Fine-tune with `train_finetune.py`
7. Evaluate with `train_eval_only.py`
8. Compare the original pretrained model with the fine-tuned model

---


## Acknowledgements

This project is based on the original official P2PNet repository and paper:

**Qingyu Song, Changan Wang, Zhengkai Jiang, Yabiao Wang, Ying Tai, Chengjie Wang, Jilin Li, Feiyue Huang, and Yang Wu.**  
**Rethinking Counting and Localization in Crowds: A Purely Point-Based Framework.**  
**ICCV 2021.**

Please also refer to the original repository README for the base implementation, installation assumptions, and citation information. 
