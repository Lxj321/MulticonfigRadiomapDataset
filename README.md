# U6G XL-MIMO Radiomap Prediction: Multi-config Dataset & Beam Map Approach

A benchmark project for **multi-configuration radiomap prediction** in **U6G / XL-MIMO** systems.

This repository is currently in **pre-release** status.

> Public release of the dataset, source code, pretrained models, and paper links will be announced later.

---

## Project Website

The project website documents the planned structure of the release, including:

- dataset organization
- benchmark task definitions
- pretrained model layout
- intended evaluation workflow

**Website:** *(add your GitHub Pages link here once available)*

---

## Preview Scope

This repository currently serves as a **documentation preview** for the upcoming public release.

At this stage:

- dataset files are **not yet public**
- source code is **not yet public**
- pretrained weights are **not yet public**
- the paper is **not yet public**

The current contents are intended to help finalize:

- documentation structure
- benchmark naming
- dataset layout
- release organization

---

## Planned Release Contents

The planned public release will include:

### 1. Dataset (`Dataset/`)

- `height_maps/`
- `radiomaps/`
- `beam_maps/`
- `configs/`
- `sionna_maps/` (optional)

### 2. Baselines

- UNet training / evaluation scripts
- GAN training / evaluation scripts

### 3. Pretrained Models (`Pretrained_Model/`)

- GAN checkpoints for 8 benchmark tasks
- UNet checkpoints and visualizations

### 4. Dataset Generation Pipeline

- OSM download
- Sionna scene generation
- height map generation
- ray-tracing radiomap generation
- beam map generation

---

## Quick Facts

- Scenes: **800**
- Frequencies: **1.8 / 2.6 / 3.5 / 4.9 / 6.7 GHz**
- TX antennas: up to **1024 TR**
- Beam counts: **1 / 8 / 16 / 64**
- Beam pattern: **3GPP TR 38.901**

---

## Planned Benchmark Tasks

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

## Documentation Pages

The website documentation is organized as:

- **Dataset** — folder structure and naming rules
- **Quickstart** — intended evaluation workflow
- **Benchmark** — task definitions
- **Pretrained** — planned pretrained checkpoint organization

---

## Release Status

This project is currently under preparation.

Planned next steps include:

- finalizing documentation
- preparing code cleanup
- preparing dataset packaging
- preparing pretrained model packaging
- finalizing paper submission

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
````

---

## License

* **Code:** planned to be released under the **MIT License**
* **Dataset:** dataset license will be specified separately upon public release

---

## Contact

**Xiaojie Li (李宵杰)**
Email: `xiaojieli@seu.edu.cn/xiaojieli@nuaa.edu.cn`  
如果你愿意，我下一条可以直接给你一版 **“预发布状态”的 `quickstart.md`**。
```
