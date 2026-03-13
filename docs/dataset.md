# Dataset

This dataset provides **height maps**, **configuration-only beam maps**, and **ray-tracing radiomaps** for studying **multi-configuration radiomap prediction** in **U6G / XL-MIMO** systems.

It is designed to support research on:

- cross-configuration generalization
- cross-environment generalization
- beam-aware radiomap prediction
- sparse radiomap reconstruction

The released benchmark separates three complementary data components:

- **height maps** for scene geometry representation
- **beam maps** for configuration-aware and environment-independent side information
- **radiomaps** for environment-dependent ray-tracing labels

---

## Visual Examples

Representative examples from the released dataset are shown below.

### Overall Preview

An overall preview of the released data components, including the height map, ray-tracing radiomap, and configuration-only beam map.

<p align="center">
  <img src="assets/dataset/homepage_teaser.png" width="88%">
</p>

### Radiomap Examples

Representative ray-tracing radiomap examples from different scenes and configurations.

<p align="center">
  <img src="assets/dataset/radiomaps_grid.png" width="88%">
</p>

### Height Map Examples

Representative height map examples from the released urban scenes.

<p align="center">
  <img src="assets/dataset/height_maps_grid.png" width="88%">
</p>

### Beam Map Examples

Representative configuration-only beam map examples.

<p align="center">
  <img src="assets/dataset/beam_maps_grid.png" width="88%">
</p>

### Paired Examples

The following examples illustrate the correspondence among the released data components for selected scenes and transmitter configurations.

<p align="center">
  <img src="assets/dataset/paired_examples/u9__freq_6.7GHz_64TR_8beams_pattern_tr38901__beam07.png" width="95%">
</p>

<p align="center">
  <img src="assets/dataset/paired_examples/u100__freq_6.7GHz_1024TR_64beams_pattern_tr38901__beam41.png" width="95%">
</p>

<p align="center">
  <img src="assets/dataset/paired_examples/u460__freq_6.7GHz_256TR_16beams_pattern_tr38901__beam00.png" width="95%">
</p>

<p align="center">
  <img src="assets/dataset/paired_examples/u100__freq_1.8GHz_16TR_1beams_pattern_tr38901__beam00.png" width="95%">
</p>

<p align="center">
  <img src="assets/dataset/paired_examples/u461__freq_6.7GHz_1024TR_64beams_pattern_tr38901__beam16.png" width="95%">
</p>

---

## Dataset Statistics

| Item | Value |
|-----|------|
| Scenes | 800 |
| Configurations | 98 |
| Frequencies | 1.8 / 2.6 / 3.5 / 4.9 / 6.7 GHz |
| Transmit antennas | 4 → 1024 TR |
| Beam counts | 1 / 8 / 16 / 64 |
| Beam pattern | 3GPP TR 38.901 |

---
---

# Folder Structure

```text
Dataset/
  beam_maps/
    <config_id>/
      u0/
        beam_XX_angle_*.npy
        u0_all_beams.npz
        beam_settings.txt
        *_plot.png (optional)
  configs/
    *.txt
  height_maps/
    u1..u800/
      u*_height_matrix.npy
      u*_height_matrix_coords.npz
  radiomaps/
    <config_id>_beamXX/
      u1_labeled_radiomap.npy
      ...
      u800_labeled_radiomap.npy
      beam_settings.txt
  sionna_maps/ (optional)
    u1..u800/
      meshes/*.ply
```

---

# Dataset Components

## 1. Height Maps

Height maps provide geometry-aware scene information for each environment.

Stored under:

```text
height_maps/u*/
```

Each scene contains:

* `u*_height_matrix.npy`
* `u*_height_matrix_coords.npz`

### Height matrix

```text
u*_height_matrix.npy
```

A 2.5D height map describing building and terrain height on a regular grid.

### Coordinate metadata

```text
u*_height_matrix_coords.npz
```

Stores the grid-coordinate metadata associated with the height matrix.

### Default generation settings

* target area: `1280 m × 1280 m`
* grid resolution: `5 m`
* default height per floor: `5 m`
* single-floor buildings are assigned a height of `20 m`

### Notes

* height maps are scene-dependent
* they are used as geometry-related inputs for radiomap learning
* if multiple buildings overlap on the same grid cell, the maximum height is used

---

## 2. Beam Maps (Configuration-Only)

Beam maps provide **configuration-aware** but **environment-independent** side information.

Stored under:

```text
beam_maps/<config_id>/u0/
```

### Important

`u0` does **not** correspond to a real scene.

It is a placeholder used to store **configuration-only beam map features** that depend on transmitter configuration and beam index, but not on a specific environment. The same beam maps can therefore be reused across all scenes.

### Typical files

#### Per-beam matrices

```text
beam_XX_angle_*.npy
```

Per-beam matrix representations of the beam pattern.

#### All-beam package

```text
u0_all_beams.npz
```

Compressed package containing all beam maps under one configuration.

#### Beam metadata

```text
beam_settings.txt
```

Beam and configuration metadata for the generated beam maps.

#### Visualization (optional)

```text
*_plot.png
```

Visualization of beam patterns.

### Modeling assumptions

The released beam maps are generated from a simplified geometric model:

* beam maps are **not** ray-tracing radiomaps
* only LOS geometry is considered
* no reflection, diffraction, blockage, or occlusion is modeled

The released generator supports:

* **plane-wave approximation**
* **spherical-wave model**

### Practical usage rule

When predicting radiomaps for configuration `<config_id>`, always load beam maps from:

```text
beam_maps/<config_id>/u0/
```

Beam maps should be paired with radiomaps using the **same configuration** and **same beam index**.

---

## 3. Radiomaps (Labels)

Radiomaps provide the environment-dependent labels generated by the ray-tracing pipeline.

Stored under:

```text
radiomaps/<config_id>_beamXX/
```

Each folder corresponds to **one configuration + one beam**.

### Files

#### Radiomap label

```text
u*_labeled_radiomap.npy
```

Radiomap corresponding to:

* scene `u*`
* configuration `<config_id>`
* beam `beamXX`

#### Beam metadata

```text
beam_settings.txt
```

Configuration metadata duplicated for convenience.

### Example access

Scene:

```text
u123
```

Configuration:

```text
freq_6.7GHz_256TR_16beams_pattern_tr38901
```

Beam:

```text
beam03
```

Radiomap file:

```text
Dataset/radiomaps/freq_6.7GHz_256TR_16beams_pattern_tr38901_beam03/u123_labeled_radiomap.npy
```

### Label semantics

The released labeled radiomap uses the following conventions:

* **building region**: `1000`
* **invalid / no-label region**: `-300`
* **valid propagation region**: values strictly between `-300` and `0` dB

This means:

* building cells are explicitly marked in the labeled radiomap
* invalid cells correspond to the ray-tracing floor value
* only valid propagation cells represent effective supervision targets in downstream preprocessing

### Notes

* radiomaps are environment-dependent ray-tracing labels
* under the default released generation settings, invalid cells correspond to `-300 dB`
* building regions are explicitly marked by `1000` in the labeled radiomap

---

# Indexing and Naming Rules

## Scene ID (`u1..u800`)

`u1..u800` denote different scenes / environments.

Each scene corresponds to a distinct geographic region with its own height map and ray-tracing environment.

---

## Configuration ID (`<config_id>`)

Configuration folders follow the naming format:

```text
freq_{f}GHz_{NTR}TR_{B}beams_pattern_tr38901
```

Examples:

```text
freq_1.8GHz_4TR_1beams_pattern_tr38901
freq_6.7GHz_1024TR_64beams_pattern_tr38901
```

### Interpretation

| Field             | Meaning                             |
| ----------------- | ----------------------------------- |
| `f`               | carrier frequency (GHz)             |
| `NTR`             | number of transmit antennas         |
| `B`               | number of beams in the codebook     |
| `pattern_tr38901` | beam pattern follows 3GPP TR 38.901 |

---

## Beam ID (`beamXX`)

Beam indices follow:

```text
beam00 .. beam{B-1}
```

where `B` is the beam count of the configuration.

### Examples

| Configuration | Beam IDs           |
| ------------- | ------------------ |
| 1 beam        | `beam00`           |
| 8 beams       | `beam00 .. beam07` |
| 16 beams      | `beam00 .. beam15` |
| 64 beams      | `beam00 .. beam63` |

---

# Mesh Assets (Optional)

The directory

```text
sionna_maps/
```

contains `.ply` meshes used to reproduce the ray-tracing scenes.

Example structure:

```text
sionna_maps/
  u1/
    meshes/*.ply
```

## Usage

Mesh assets are required **only if reproducing the ray-tracing pipeline**.

They are **not required** for training or evaluating machine learning models using the released height maps, beam maps, and radiomaps.

---

# Summary of Roles

The three released data components play different roles:

| Component       | Depends on scene? | Depends on configuration? | Main role                             |
| --------------- | ----------------- | ------------------------- | ------------------------------------- |
| Height map      | Yes               | No                        | geometry-aware scene input            |
| Beam map (`u0`) | No                | Yes                       | configuration-aware side information  |
| Radiomap        | Yes               | Yes                       | environment-dependent learning target |

This separation is one of the key design choices of the released benchmark.

---

# Used by the Baselines

The released baseline pipelines use these dataset components in different but coordinated ways:

* GAN and UNet preprocessing both use the **height map** as geometry-related input
* **feature-map mode** loads beam maps from the configuration-only reference folder `u0`
* radiomap preprocessing extracts the **building mask** and **valid mask** from the labeled radiomaps

In particular:

* the **building region** is identified from the labeled radiomap value `1000`
* the **invalid / no-label region** is identified from the default floor value `-300`
* only valid propagation regions are used as effective supervision targets in downstream preprocessing

