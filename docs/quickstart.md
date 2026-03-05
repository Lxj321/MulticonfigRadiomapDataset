
# Quickstart

This page shows how to prepare the dataset and pretrained checkpoints, and how to run the provided evaluation scripts.

---

# 1. Prepare Dataset and Pretrained Weights

Place the dataset and pretrained models under the project root as:

```text
<project_root>/
  Dataset/
    radiomaps/
    height_maps/
    beam_maps/
    configs/
    sionna_maps/          # optional
  Pretrained_Model/
    GAN/
      ...
    Unet/
      ...
````

The provided scripts assume these default paths.

---

# 2. Evaluate Pretrained UNet

The UNet evaluation script uses a built-in configuration class (`EvalConfig`) and does **not** require command-line arguments.

Run:

```bash
python ModelEvaluation_Unet.py
```

By default, the script uses:

* `Dataset/radiomaps`
* `Dataset/height_maps`
* `Dataset/beam_maps`

and evaluates a predefined list of UNet checkpoints under:

* `Pretrained_Model/Unet/`

The script also uses:

* `RANDOM_SEED = 42`
* `TRAIN_RATIO = 0.7`
* `VAL_RATIO = 0.1`
* `TEST_RATIO = 0.2`
* `BATCH_SIZE = 64`

Results are saved to:

```text
evaluation_results/
```

Generated outputs include:

* `evaluation_summary_dB.csv`
* `metrics_comparison_dB.png`
* `{model_name}_visualization.png`

> Notes:
>
> * Metrics are computed in the **dB domain**.
> * Buildings and invalid regions are excluded.
> * SSIM is computed only on valid regions.

---

# 3. Evaluate Pretrained GAN

The GAN evaluation script is currently provided in a **Jupyter / notebook-style** format.

Its default paths are:

* `Dataset/radiomaps`
* `Dataset/height_maps`
* `Dataset/beam_maps`

and the experiment directory is:

```text
Pretrained_Model/GAN
```

## Option A: Run as a Python script

Because the file ends with a direct call to:

```python
run_evaluation()
```

you can run:

```bash
python ModelEvaluation_GAN.py
```

This will evaluate all discovered GAN experiments under:

```text
Pretrained_Model/GAN/
```

and save summary files such as:

* `Pretrained_Model/GAN/evaluation_summary.json`
* `Pretrained_Model/GAN/evaluation_summary.csv`

## Option B: Use in Jupyter Notebook

You can also copy the script into a notebook cell and run:

```python
results = run_evaluation()
```

Useful variants:

```python
results = run_evaluation(single='random_dense_feature')
results = run_evaluation(visualize=True)
```

---

# 4. Supported GAN Tasks

The GAN evaluator is aligned with the following task folders:

* `beam_dense_encoding`
* `beam_dense_feature`
* `random_dense_encoding`
* `random_dense_feature`
* `random_sparse_encoding_samples819`
* `random_sparse_feature_samples819`
* `scene_dense_encoding`
* `scene_dense_feature`

These are automatically detected from subfolders containing `best_G.pth`.

---

# 5. Important Notes

## UNet script behavior

`ModelEvaluation_Unet.py` evaluates a predefined model list from `EvalConfig.MODELS`.

If you want to:

* evaluate only one checkpoint
* change paths
* change split strategy
* add/remove models

please edit the corresponding entries in `EvalConfig`.

## GAN script behavior

`ModelEvaluation_GAN.py` automatically scans `Pretrained_Model/GAN/` and evaluates experiment folders that contain:

```text
best_G.pth
```

It infers task settings from either:

* predefined `EXPERIMENT_CONFIGS`
* `config.json`
* or folder naming rules

---

# 6. Common Issues

## File not found

Check that the following folders exist exactly as expected:

* `Dataset/radiomaps`
* `Dataset/height_maps`
* `Dataset/beam_maps`
* `Pretrained_Model/GAN`
* `Pretrained_Model/Unet`

## GPU selection

The UNet evaluation script sets:

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

If you need another GPU, modify this value in the script.

## Missing Python packages

The scripts require packages such as:

* `torch`
* `numpy`
* `matplotlib`
* `pandas`
* `scikit-image`

The UNet script also imports:

* `seaborn`

---

# 7. Recommended First Run

For a first verification, it is recommended to:

* run `python ModelEvaluation_Unet.py` to evaluate the predefined UNet checkpoints
* run `python ModelEvaluation_GAN.py` to evaluate all GAN experiment folders

This provides a quick sanity check that:

* dataset paths are correct
* pretrained checkpoints are correctly placed
* the evaluation pipeline runs end-to-end
