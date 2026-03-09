
# U6G XL-MIMO Radiomap Prediction: Multi-config Dataset & Beam Map Approach

A benchmark project for **multi-configuration radiomap prediction** in **U6G / XL-MIMO** systems.

> **Public release:** The dataset, pretrained models, source code, and project website are now publicly available.
>
> -**Paper:**https://arxiv.org/abs/2603.06401
> - **Dataset & Pretrained Models:** https://huggingface.co/datasets/lxj321/Multi-config-Radiomap-Dataset
> - **Code Repository:** https://github.com/Lxj321/MulticonfigRadiomapDataset
> - **Project Website:** https://lxj321.github.io/MulticonfigRadiomapDataset/

---

## Overview

This project is designed for studying:

- **multi-configuration radiomap prediction**
- **cross-configuration generalization**
- **cross-environment generalization**
- **beam-aware radiomap modeling**
- **sparse radiomap reconstruction**

A key feature of this project is the joint design of:

- **height maps**
- **configuration-aware beam maps**
- **ray-tracing radiomap labels**
- **optional mesh assets for ray-tracing reproduction**
- **UNet / GAN baseline pipelines**

---

## Released Resources

The current public release includes:

### 1. Dataset

The released dataset includes resources for large-scale U6G / XL-MIMO radiomap prediction, including:

- height maps
- radiomaps
- beam maps
- configuration files
- optional ray-tracing related assets

### 2. Baseline Code

This repository provides code for:

- UNet training and evaluation
- GAN training and evaluation
- dataset loading and benchmark execution
- dataset generation pipeline
- beam map generation
- radiomap generation and validation

### 3. Pretrained Models

Pretrained checkpoints for benchmark tasks are released in the Hugging Face repository.

### 4. Documentation

Detailed documentation is provided on the project website, including:

- dataset structure
- benchmark settings
- quickstart instructions
- pretrained model organization

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

These tasks cover variations in:

- split strategy: random / beam / scene
- supervision density: dense / sparse
- input mode: feature / encoding

---

## Repository Structure

This repository includes code and documentation related to:

- dataset generation
- model training
- model evaluation
- project documentation

Representative files include:

- `DatasetGeneration_Step1_OSMDownload.py`
- `DatasetGeneration_Step2_OSMToSionna.py`
- `DatasetGeneration_Step3_OSMToHeightMap.py`
- `DatasetGeneration_Step4_RadiomapRT.py`
- `DatasetGeneration_Step5_RadiomapValidation.py`
- `DatasetGeneration_Step6_BeammapGeneration.py`
- `ModelTraining_GAN.py`
- `ModelTraining_Unet.py`
- `ModelEvaluation_GAN.py`
- `ModelEvaluation_Unet.py`

---

## Getting Started

For dataset download, pretrained models, and detailed documentation, please refer to:

- **Hugging Face:** https://huggingface.co/datasets/lxj321/Multi-config-Radiomap-Dataset
- **Project Website:** https://lxj321.github.io/MulticonfigRadiomapDataset/

---

## Citation

If you use this project, please cite the corresponding paper or project page.

```bibtex
@misc{li2026u6gxlmimoradiomapprediction,
      title={U6G XL-MIMO Radiomap Prediction: Multi-Config Dataset and Beam Map Approach}, 
      author={Xiaojie Li and Yu Han and Zhizheng Lu and Shi Jin and Chao-Kai Wen},
      year={2026},
      eprint={2603.06401},
      archivePrefix={arXiv},
      primaryClass={eess.SP},
      url={https://arxiv.org/abs/2603.06401}, 
}
```

Formal citation information will be updated after the paper metadata is finalized.

---

## License

* **Code:** MIT License
* **Dataset / Pretrained Models:** see the Hugging Face repository for the corresponding release license

---

## Contact

**Xiaojie Li (李宵杰)**
Email: `xiaojieli@seu.edu.cn` / `xiaojieli@nuaa.edu.cn`
