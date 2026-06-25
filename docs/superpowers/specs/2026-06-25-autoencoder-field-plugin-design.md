# Autoencoder Field Plugin Design

## Goal

Add a SurroForge plugin that provides an `autoencoder-field` trainer for field
surrogates. The trainer learns a nonlinear field compression with a PyTorch
autoencoder and learns the parameter-to-latent map with an existing SurroForge
inner trainer such as `mlp` or `torch-mlp`.

## User-Facing Configuration

The plugin registers one trainer through the `surroforge.trainers` entry point:

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

`inner` is the surrogate that maps design parameters to latent coordinates. It
uses the already supported SurroForge `InnerConfig` block.

## Architecture

`AutoencoderFieldTrainer` implements `BaseTrainer`. During `fit(x, y)` it
standardizes the field matrix `y`, trains an autoencoder on the standardized
fields, encodes the training fields to latent vectors, and fits the configured
inner trainer on `x -> latent`. During `predict(x)` it predicts latent vectors
with the inner trainer and decodes them back to full field values.

The plugin stays outside Core. It does not change `reduced-field`, because Core's
POD trainer intentionally supports only `basis: pod`.

## Scope

The first implementation supports pickle persistence through the normal
SurroForge model artifact path. It does not add ONNX or PT exports. It does not
add ROM basis artifacts, because an autoencoder does not expose an orthonormal
linear basis comparable to POD.

## Validation

The trainer validates unknown hyperparameters, positive dimensions, supported
activations, integer `random_state`, positive training limits, and fitted-state
checks. Tests cover registration metadata, construction validation, holdout
field prediction, pickling, and integration with the existing trainer API.
