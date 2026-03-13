# U6G XL-MIMO Radiomap Prediction: Multi-config Dataset & Beam Map Approach

A public benchmark and reproducibility package for **multi-configuration radiomap prediction** in **U6G / XL-MIMO** systems.

> **Public release:** The dataset, pretrained models, source code, and project website are now publicly available.
>
> - **Paper (arXiv):** https://arxiv.org/abs/2603.06401
> - **Dataset & Pretrained Models (Hugging Face):** https://huggingface.co/datasets/lxj321/Multi-config-Radiomap-Dataset
> - **Code Repository (GitHub):** https://github.com/Lxj321/MulticonfigRadiomapDataset
> - **Project Website:** https://lxj321.github.io/MulticonfigRadiomapDataset/

---

## Overview

This project provides a unified benchmark for radiomap prediction under multiple transmitter configurations in U6G / XL-MIMO systems. It is designed to support reproducible research on:

- **multi-configuration radiomap prediction**
- **cross-configuration generalization**
- **cross-environment generalization**
- **beam-aware radiomap modeling**
- **sparse radiomap reconstruction**

A key feature of the release is the joint organization of:

- **height maps**
- **configuration-aware beam maps**
- **ray-tracing radiomap labels**
- **ray-tracing scenes and related assets**
- **UNet / GAN baseline pipelines**

---

## Dataset Preview

Representative examples from the released U6G XL-MIMO multi-configuration dataset, including height maps, ray-tracing radiomaps, and configuration-only beam maps.

<p align="center">
  <img src="docs/assets/dataset/homepage_teaser.png" width="82%">
</p>

## Example Galleries

<p align="center">
  <img src="docs/assets/dataset/height_maps_grid.png" width="82%">
</p>

<p align="center">
  <img src="docs/assets/dataset/radiomaps_grid.png" width="82%">
</p>

<p align="center">
  <img src="docs/assets/dataset/beam_maps_grid.png" width="82%">
</p>

## Released Resources

The public release includes:

### 1. Dataset
- height maps
- radiomaps
- beam maps
- configuration files
- ray-tracing scenes and simulation-related assets

### 2. Baseline Code
- dataset generation pipeline
- dataset preprocessing and loading
- UNet training and evaluation
- GAN training and evaluation
- validation and visualization tools

### 3. Pretrained Models
Pretrained checkpoints for benchmark tasks are released in the Hugging Face repository.

### 4. Documentation
Detailed documentation is provided on the project website, including dataset structure, benchmark settings, quickstart instructions, and pretrained model organization.

---

## Quick Facts

- **Scenes:** 800
- **Frequencies:** 1.8 / 2.6 / 3.5 / 4.9 / 6.7 GHz
- **TX antennas:** up to 1024 TR
- **Beam counts:** 1 / 8 / 16 / 64
- **Beam pattern:** 3GPP TR 38.901

---

## Benchmark Tasks

The benchmark uses a unified task naming scheme:

- `random_dense_feature`
- `random_dense_encoding`
- `beam_dense_feature`
- `beam_dense_encoding`
- `scene_dense_feature`
- `scene_dense_encoding`
- `random_sparse_feature_samples819`
- `random_sparse_encoding_samples819`

These tasks vary in:
- split strategy: `random / beam / scene`
- supervision density: `dense / sparse`
- input mode: `feature / encoding`

---

## Getting Started

### Option A: Inspect the released dataset
Download the dataset packages and pretrained models from Hugging Face.

### Option B: Evaluate released checkpoints
Use the released pretrained models together with the evaluation scripts in this repository.

### Option C: Retrain benchmark models
Use the training scripts together with the documented benchmark task settings.

### Option D: Reproduce data generation
Use the dataset-generation scripts to inspect or reproduce the pipeline for scene processing, radiomap generation, validation, and beam-map construction.

For detailed usage instructions and benchmark documentation, please refer to the project website:
- **Project Website:** https://lxj321.github.io/MulticonfigRadiomapDataset/

---

## Core Dependencies

The released repository spans multiple workflows, and different parts of the codebase rely on different environments.

### For evaluation and baseline training
The released UNet / GAN training and evaluation scripts primarily rely on standard Python ML packages such as:
- `PyTorch`
- `NumPy`
- `Pandas`
- `Matplotlib`
- `scikit-image`

### For ray-tracing radiomap generation
The radiomap generation pipeline (`DatasetGeneration_Step4_RadiomapRT.py`) relies on:
- `TensorFlow`
- `Sionna 0.19.2`

### For scene construction from OSM
The scene-conversion step (`DatasetGeneration_Step2_OSMToSionna.py`) relies on:
- `Blender 4.0`
- `bpy` (Blender Python environment)

### Important note
Users who only want to evaluate the released pretrained checkpoints do **not** need to install the full dataset-generation environment.  
The `Sionna 0.19.2` and `Blender 4.0 + bpy` dependencies are mainly required for reproducing the released data-generation pipeline.

---

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{li2026u6gxlmimoradiomapprediction,
      title={U6G XL-MIMO Radiomap Prediction: Multi-Config Dataset and Beam Map Approach}, 
      author={Xiaojie Li and Yu Han and Zhizheng Lu and Shi Jin and Chao-Kai Wen},
      year={2026},
      eprint={2603.06401},
      archivePrefix={arXiv},
      primaryClass={eess.SP},
      url={https://arxiv.org/abs/2603.06401}
}
```
## Contact
Xiaojie Li (李宵杰）:xiaojieli@seu.edu.cn/xiaojieli@nuaa.edu.cn
