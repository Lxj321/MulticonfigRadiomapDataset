# Pretrained Models

## GAN

Checkpoints are organized as:

```text
Pretrained_Model/GAN/<task_id>/
  best_G.pth
  best_D.pth
  config.json
  eval_results.json
  epoch_history.json
  batch_history.json

Available tasks:

beam_dense_encoding

beam_dense_feature

random_dense_encoding

random_dense_feature

random_sparse_encoding_samples819

random_sparse_feature_samples819

scene_dense_encoding

scene_dense_feature


## Unet

UNet checkpoints are under:
Pretrained_Model/Unet/
  *.pt
  *.png
  *.csv


We will provide a mapping table from checkpoint names to tasks.
