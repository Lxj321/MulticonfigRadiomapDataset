# U6G XL-MIMO Radiomap Prediction: Multi-config Dataset & Beam Map Approach

A benchmark dataset for **multi-configuration radiomap prediction** in U6G/XL-MIMO systems, featuring **800 scenes**, **multi-frequency**, **multi-antenna** and **multi-beam** settings.

[Dataset](dataset.md){ .md-button } [Quickstart](quickstart.md){ .md-button } [Benchmark](benchmark.md){ .md-button } [Pretrained](pretrained.md){ .md-button }

---

## What’s included

- **Dataset** (`Dataset/`): height maps, radiomaps, configuration-only beam maps, optional mesh assets  
- **Baselines**: UNet and GAN training/evaluation scripts  
- **Pretrained models** (`Pretrained_Model/`): GAN (8 tasks) + UNet checkpoints  
- **Dataset generation**: OSM → Sionna meshes → height maps → RT radiomaps → beam maps

---

## Quick facts

- Scenes: **800** (`u1..u800`)
- Frequencies: **1.8 / 2.6 / 3.5 / 4.9 / 6.7 GHz**
- TX antennas: up to **1024 TR**
- Beams: **1 / 8 / 16 / 64**
- Beam pattern: **TR38.901**

---

## Repo status

This website is being finalized. Please check **Quickstart** for running pretrained evaluation.
