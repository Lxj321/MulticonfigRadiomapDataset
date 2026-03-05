# Dataset

This dataset provides **height maps**, **configuration-only beam maps**, and **ray-tracing radiomaps** for studying **multi-configuration radiomap prediction** in U6G / XL-MIMO systems.

The dataset is designed to support research on:

- cross-configuration generalization
- cross-environment generalization
- beam-aware radiomap prediction
- sparse radiomap reconstruction

---

# Dataset Statistics

| Item | Value |
|-----|------|
| Scenes | 800 |
| Configurations | 98 |
| Frequencies | 1.8 / 2.6 / 3.5 / 4.9 / 6.7 GHz |
| Transmit antennas | 4 → 1024 TR |
| Beam counts | 1 / 8 / 16 / 64 |
| Beam pattern | 3GPP TR 38.901 |

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
  configs/
    *.txt
  height_maps/
    u1..u800/
      u*_height_matrix.npy
      u*_height_matrix_coords.npz
  radiomaps/
    <config_id>_beamXX/
      u1..u800_labeled_radiomap.npy
      beam_settings.txt
  sionna_maps/ (optional)
    u1..u800/meshes/*.ply
```


---

# Indexing & Naming Rules

## Scene ID (`u1..u800`)

`u1..u800` denote different scenes / environments.

Each scene corresponds to a distinct geographic region with its own height map and ray-tracing environment.

---

## Configuration ID (`<config_id>`)

Configuration folders follow the naming format

```text
freq_{f}GHz_{NTR}TR_{B}beams_pattern_tr38901
```

Example configurations

```text
freq_1.8GHz_4TR_1beams_pattern_tr38901
freq_6.7GHz_1024TR_64beams_pattern_tr38901
```

Interpretation

| Field           | Meaning                             |
| --------------- | ----------------------------------- |
| f               | carrier frequency (GHz)             |
| NTR             | number of transmit antennas         |
| B               | number of beams in the codebook     |
| pattern_tr38901 | beam pattern follows 3GPP TR 38.901 |

---

## Beam ID (`beamXX`)

Beam indices follow

```text
beam00 .. beam{B-1}
```

where **B** is the beam count of the configuration.

Example

| Configuration | Beam IDs         |
| ------------- | ---------------- |
| 1 beam        | beam00           |
| 8 beams       | beam00 .. beam07 |
| 16 beams      | beam00 .. beam15 |
| 64 beams      | beam00 .. beam63 |

---

# Beam Maps (Configuration-Only)

Beam maps are stored under

```text
beam_maps/<config_id>/u0/
```

### Important

`u0` **does not correspond to a real scene**.

It is a placeholder used to store **configuration-only beam map features**, which are **environment-independent**.

This design allows beam maps to be reused across all scenes.

---

## Typical Files

### Per-beam matrices

```text
beam_XX_angle_*.npy
```

Matrix representation of the beam pattern.

---

### All beams package

```text
u0_all_beams.npz
```

Compressed file containing beam maps for all beams under the configuration.

---

### Beam configuration metadata

```text
beam_settings.txt
```

Contains beam parameters and configuration information.

---

### Visualization (optional)

```text
*_plot.png
```

Visualization of beam patterns.

---

## Practical Usage Rule

When predicting radiomaps for configuration `<config_id>`, always load beam maps from

```text
beam_maps/<config_id>/u0/
```

Beam maps should be paired with radiomaps according to the **same configuration and beam index**.

---

# Height Maps

Height maps are stored under

```text
height_maps/u*/
```

Each scene `u*` contains the following files.

---

## Height Matrix

```text
u*_height_matrix.npy
```

2.5D height map representing the terrain and building heights.

---

## Coordinate Metadata

```text
u*_height_matrix_coords.npz
```

Contains coordinate information for the height grid (e.g., x/y axes or grid coordinates).

---

## Notes

For full reproducibility, the following information will be documented in future updates:

* height map resolution (meters per pixel)
* coordinate origin and axis directions
* receiver plane height definition
* antenna array orientation reference

---

# Radiomaps (Labels)

Radiomap labels are stored under

```text
radiomaps/<config_id>_beamXX/
```

Each folder corresponds to **one configuration + one beam**.

---

## Files

### Radiomap label

```text
u*_labeled_radiomap.npy
```

Radiomap corresponding to

* scene `u*`
* configuration `<config_id>`
* beam `beamXX`

---

### Beam metadata

```text
beam_settings.txt
```

Metadata duplicated for convenience.

---

## Example Access

Scene

```text
u123
```

Configuration

```text
freq_6.7GHz_256TR_16beams_pattern_tr38901
```

Beam

```text
beam03
```

Corresponding radiomap file

```text
Dataset/radiomaps/freq_6.7GHz_256TR_16beams_pattern_tr38901_beam03/u123_labeled_radiomap.npy
```

---

## Notes

The following details will be documented for reproducibility:

* physical quantity (e.g., received power / pathloss / RSRP)
* unit (dB / dBm / linear)
* masking convention for buildings / invalid areas
* tensor shape ordering (H×W or C×H×W)

---

# Mesh Assets (Optional)

The directory

```text
sionna_maps/
```

contains `.ply` meshes used to reproduce the ray-tracing scenes.

Example structure

```text
sionna_maps/
  u1/
    meshes/*.ply
```

---

## Usage

Mesh assets are required **only if reproducing the ray-tracing pipeline**.

They are **not required** for training or evaluating machine learning models using the provided radiomaps and height maps.

```

如果你愿意，我下一步可以按同样方式把 **`index.md`、`quickstart.md`、`benchmark.md`、`pretrained.md`** 也一次性整理成完整可复制的 Markdown 文件。
```
