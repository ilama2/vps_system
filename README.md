# ACEG-Custom-Dataset-Experiments

Custom dataset experiments using ACE-G for visual relocalization and scene coordinate regression.

---

## Overview

This repository contains a full pipeline for:

- COLMAP sparse reconstruction
- Converting COLMAP outputs into ACE-G dataset format
- Dataset splitting
- ACE-G training
- Image registration
- Pose evaluation

The project focuses on experimenting with ACE-G on custom real-world datasets.

---

## Pipeline

Images
→ COLMAP Reconstruction
→ Sparse TXT Model
→ ACE-G Dataset Conversion
→ Train/Test Split
→ ACE-G Training
→ Registration
→ Pose Evaluation

---

## Repository Structure

```text
.
├── scripts/
│   ├── run_colmap.sh
│   ├── train_aceg.sh
│   ├── convert_to_aceg.py
│   └── split_dataset.py
│
│
├── configs/
│   └── aceg_custom.yaml
│
└── README.md
