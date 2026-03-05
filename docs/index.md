# U6G XL-MIMO Radiomap Prediction: Multi-config Dataset & Beam Map Approach

A benchmark project for **multi-configuration radiomap prediction** in **U6G / XL-MIMO** systems, featuring **800 scenes**, **multi-frequency**, **multi-antenna**, and **multi-beam** settings.

> **Preview mode:** This website is currently in a pre-release state.  
> Public release of the dataset, code, pretrained models, and paper links will be announced later.

[Dataset](dataset.md) [Quickstart](quickstart.md){ .md-button } [Benchmark](benchmark.md){ .md-button } [Pretrained](pretrained.md){ .md-button }

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

## Current Status

The project website has been set up for documentation preview.

At this stage:

- dataset files are **not yet publicly released**
- source code is **not yet publicly released**
- pretrained checkpoints are **not yet publicly released**
- the paper is **not yet submitted / publicly available**

The current website is intended to:

- preview the dataset structure
- preview benchmark task definitions
- preview pretrained model organization
- help finalize documentation before public release

---

## What Will Be Released

The planned public release will include:

- **Dataset** (`Dataset/`)
  - height maps
  - radiomaps
  - configuration-only beam maps
  - optional mesh assets
- **Baselines**
  - UNet training / evaluation scripts
  - GAN training / evaluation scripts
- **Pretrained models** (`Pretrained_Model/`)
  - GAN checkpoints for 8 benchmark tasks
  - UNet checkpoints and visualizations
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

## Planned Links

- **Project Website:** this page
- **Code Repository:** coming soon
- **Dataset Download:** coming soon
- **Pretrained Models Download:** coming soon
- **Paper / Preprint:** coming soon

---

## Future Updates

Future public updates will include:

- dataset download links
- pretrained model download links
- code release
- exact tensor shape / unit documentation
- full preprocessing conventions
- UNet checkpoint-to-task mapping
- paper / citation details

---

## Citation

Citation information will be added after the paper metadata is finalized.

```bibtex
@article{to_be_added,
  title   = {U6G XL-MIMO Radiomap Prediction: Multi-config Dataset and Beam Map Approach},
  author  = {Xiaojie Li and collaborators},
  journal = {to be added},
  year    = {2026}
}
```

---

## License

* **Code:** planned to be released under the **MIT License**
* **Dataset:** dataset license will be specified separately at public release

---

## Contributor

**Xiaojie Li (李宵杰)**
Email: `xiaojieli@seu.edu.cn/xiaojieli@nuaa.edu.cn`
