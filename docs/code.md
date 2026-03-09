# Code

This page summarizes the organization of the released codebase and explains the role of the main scripts in the repository.

The repository contains code for:

- dataset generation
- dataset preprocessing and loading
- model training
- model evaluation
- validation and visualization

---

# Repository Overview

The released repository is organized around three major workflows:

1. **dataset generation**
   - construct scenes, height maps, radiomaps, and beam maps

2. **benchmark data preparation**
   - load released data into unified dense / sparse learning tasks
   - prepare feature-map or continuous-encoding inputs
   - define building-mask, valid-mask, and sparse-sampling logic

3. **baseline training and evaluation**
   - train and evaluate released UNet and GAN baselines

---

# Main Script Groups

## 1. Dataset generation scripts

These scripts implement the released data-generation pipeline.

### `DatasetGeneration_Step1_OSMDownload.py`
Randomly samples candidate geographic regions, filters them by building count, and downloads OSM building data.

### `DatasetGeneration_Step2_OSMToSionna.py`
Converts downloaded OSM data into cropped 3D urban scene XML files for the downstream ray-tracing pipeline.

**Environment note:** this step is intended to run inside a **Blender 4.0 Python environment (`bpy`)** with Mitsuba/XML export support enabled. It is not intended to run in a standard Python interpreter.

### `DatasetGeneration_Step3_OSMToHeightMap.py`
Rasterizes OSM building footprints into height maps for geometry-aware learning.

### `DatasetGeneration_Step4_RadiomapRT.py`
Generates multi-beam radiomap labels by running ray tracing on scene XML files.

### `DatasetGeneration_Step5_RadiomapValidation.py`
Provides validation and visualization utilities for checking radiomap / height-map alignment, building masks, and invalid regions.

### `DatasetGeneration_Step6_BeammapGenerator.py`
Generates configuration-aware beam maps from a simplified LOS-based geometric model.

---

## 2. Dataset preprocessing and loading

These scripts convert the released raw dataset into benchmark-ready model inputs.

### `multiconfig_dataset_prepcocess_GAN.py`
Dataset class and preprocessing pipeline used by the released GAN baselines.

It supports:
- `random / beam / scene` split strategies
- `dense / sparse` supervision modes
- feature-map inputs
- continuous-encoding inputs

It also defines:
- building-mask extraction
- valid-region masking
- sparse sampling-mask generation
- target normalization

### `multiconfig_dataset_prepcocess_Unet.py`
Dataset class and preprocessing pipeline used by the released UNet baselines.

It supports:
- `random / beam / scene` split strategies
- `dense / sparse` supervision modes
- feature-map inputs
- continuous-encoding inputs
- optional use of height-map geometry channels

It also defines:
- building-mask extraction
- valid-region masking
- sparse sampling-mask generation
- target normalization

---

## 3. Training scripts

These scripts train the released baselines under the benchmark settings.

### `ModelTraining_GAN.py`
Trains GAN baselines for the released benchmark tasks.

Main supported settings:
- split strategy: `random / beam / scene`
- supervision mode: `dense / sparse`
- condition type: `feature / encoding`

The training pipeline uses:
- valid-region-aware adversarial loss
- valid-region-aware reconstruction loss
- sparse observation injection for sparse tasks

### `ModelTraining_Unet.py`
Trains the released UNet baselines.

The implemented UNet baseline follows a two-stage WNet-style design (`RadioWNet`) and supports:
- `first_only`
- `second_only`
- `both`

training modes.

The training pipeline supports:
- feature-map inputs
- continuous-encoding inputs
- dense and sparse settings
- optional second-stage refinement with the first-stage network frozen

---

## 4. Evaluation scripts

These scripts evaluate released pretrained checkpoints or re-trained baselines.

### `ModelEvaluation_GAN.py`
Evaluates released GAN checkpoints and reports benchmark metrics on the test split.

### `ModelEvaluation_Unet.py`
Evaluates released UNet checkpoints and reports benchmark metrics on the test split.

Both evaluators:
- use valid-region-aware metrics
- exclude building and invalid regions during the main error computation
- report results under the released benchmark settings

---

# Typical Workflow

## Option A: Evaluate released checkpoints

Recommended order:

1. prepare `Dataset/`
2. prepare `Pretrained_Model/`
3. run `ModelEvaluation_Unet.py`
4. run `ModelEvaluation_GAN.py`

This is the fastest way to verify that the released resources are correctly prepared.

---

## Option B: Retrain benchmark baselines

Recommended order:

1. prepare `Dataset/`
2. choose the target benchmark task setting
3. use the corresponding dataset preprocessing pipeline
4. run:
   - `ModelTraining_Unet.py`, or
   - `ModelTraining_GAN.py`
5. evaluate the trained model with the corresponding evaluation script

---

## Option C: Reproduce dataset assets

Recommended order:

1. run `DatasetGeneration_Step1_OSMDownload.py`
2. run `DatasetGeneration_Step2_OSMToSionna.py`
3. run `DatasetGeneration_Step3_OSMToHeightMap.py`
4. run `DatasetGeneration_Step4_RadiomapRT.py`
5. optionally inspect outputs with `DatasetGeneration_Step5_RadiomapValidation.py`
6. run `DatasetGeneration_Step6_BeammapGenerator.py`

---

# Key Design Roles

The released codebase separates different responsibilities clearly:

| Script group | Main role |
|---|---|
| `DatasetGeneration_*` | generate dataset assets |
| `multiconfig_dataset_prepcocess_*` | prepare benchmark-ready model inputs |
| `ModelTraining_*` | train released baselines |
| `ModelEvaluation_*` | evaluate released baselines |
| validation / visualization tools | inspect correctness and alignment |

---

# Notes

- `DatasetGeneration_Step5_RadiomapValidation.py` is a utility script rather than a core data-generation step
- `DatasetGeneration_Step6_BeammapGenerator.py` generates configuration-only beam maps, not ray-tracing radiomaps
- the dataset preprocessing scripts define how building regions, invalid regions, and valid supervision masks are handled
- sparse tasks are implemented by returning a sampling mask first, and later constructing sparse observations in the training or evaluation scripts
- only the scene-conversion step (`DatasetGeneration_Step2_OSMToSionna.py`) requires a Blender 4.0 `bpy` environment

---

# Related Pages

For more details, see:

- **Benchmark** — task definitions and benchmark settings
- **Dataset** — released data organization and label semantics
- **Pretrained Models** — checkpoint organization
- **Quickstart** — fastest evaluation-first workflow
