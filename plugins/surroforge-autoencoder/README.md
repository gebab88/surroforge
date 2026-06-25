# surroforge-autoencoder

A PyTorch autoencoder field surrogate plugin for SurroForge.

The plugin registers `autoencoder-field` through the `surroforge.trainers` entry
point. It learns a nonlinear latent representation of field snapshots and uses an
ordinary SurroForge inner trainer to map inputs such as `(aoa, speed)` to latent
coordinates.

```yaml
model:
  method: autoencoder-field
  latent_dim: 5
  encoder_hidden_layer_sizes: [256, 64]
  decoder_hidden_layer_sizes: [64, 256]
  activation: tanh
  max_iter: 2000
  learning_rate_init: 1.0e-3
  random_state: 0
  inner:
    method: torch-mlp
    hidden_layer_sizes: [48, 48]
    max_iter: 1000
```

Install from a SurroForge checkout:

```bash
pip install -e ./plugins/surroforge-autoencoder
```
