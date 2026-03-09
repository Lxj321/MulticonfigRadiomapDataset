# U6G XL-MIMO Radiomap Prediction: Multi-config Dataset & Beam Map Approach

A benchmark project for **multi-configuration radiomap prediction** in **U6G / XL-MIMO** systems, featuring **800 scenes**, **multi-frequency**, **multi-antenna**, and **multi-beam** settings.

> **Public release:** The complete dataset, pretrained models, and code are now publicly available.
> 
> - **Paper:** https://arxiv.org/abs/2603.06401
> - **Dataset & Pretrained Models (Hugging Face):** https://huggingface.co/datasets/lxj321/Multi-config-Radiomap-Dataset
> - **Code Repository (GitHub):** https://github.com/Lxj321/MulticonfigRadiomapDataset
> - **Project Website:** https://lxj321.github.io/MulticonfigRadiomapDataset/

[Dataset](dataset.md) [Quickstart](quickstart.md) [Benchmark](benchmark.md) [Pretrained](pretrained.md)


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
- **configuration-only beam maps**
- **ray-tracing radiomap labels**
- **optional mesh assets for ray-tracing reproduction**
- **UNet / GAN baseline pipelines**

---


The current website is intended to:

- preview the dataset structure
- preview benchmark task definitions
- preview pretrained model organization
- help finalize documentation before public release

---

## What Is Included

The current public release includes:

- **Dataset** (`Dataset/`)
  - height maps
  - radiomaps
  - configuration-only beam maps
  - optional mesh assets
- **Baselines**
  - UNet training / evaluation scripts
  - GAN training / evaluation scripts
- **Pretrained models** (`Pretrained_Model/`)
  - GAN checkpoints for benchmark tasks
  - UNet checkpoints and related evaluation resources
- **Dataset generation pipeline**
  - OSM → Sionna meshes → height maps → ray-tracing radiomaps → beam maps

---

## Quick Facts

- Scenes: **800** (`u1..u800`)
- Frequencies: **1.8 / 2.6 / 3.5 / 4.9 / 6.7 GHz**
- TX antennas: up to **1024 TR**
- Beam counts: **1 / 8 / 16 / 64**
- Beam pattern: **3GPP TR 38.901**

---

## Recommended Entry Points

If you are browsing this preview site, start with:

- **Dataset** — dataset structure and naming rules
- **Benchmark** — task definitions and benchmark settings
- **Pretrained** — planned pretrained model organization
- **Quickstart** — preview of the intended evaluation workflow

> Please note that some pages currently describe the **planned public structure** and may be refined before release.

---

## Links

- **Project Website:** https://lxj321.github.io/MulticonfigRadiomapDataset/
- **Code Repository:** https://github.com/Lxj321/MulticonfigRadiomapDataset
- **Dataset & Pretrained Models:** https://huggingface.co/datasets/lxj321/Multi-config-Radiomap-Dataset
- **Paper/Preprint:** coming soon
---

## Future Updates

Future updates may include:

- paper / citation metadata
- refined tensor shape and unit documentation
- additional benchmark examples
- more detailed checkpoint-to-task mapping

---

## Citation

Citation information will be added after the paper metadata is finalized.

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

---

## License

* **Code:** **MIT License**
* **Dataset:** **CC-BY-4.0 License**

---

## Contributor

**Xiaojie Li (李宵杰)**
Email: `xiaojieli@seu.edu.cn/xiaojieli@nuaa.edu.cn`
