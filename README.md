# Vollo model zoo

## Quick start

Pre-requisites:

- Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/).
- Install the [Vollo SDK](https://vollo.myrtle.ai/latest/installation.html).

Then:

1. Set the `UV_FIND_LINKS` environment variable to point at your Vollo SDK:

   ```fish
   set -x UV_FIND_LINKS /path/to/sdk/vollo-sdk-<version>/python/
   ```

2. Try a model out:

   ```fish
   uv run zoo wavenet
   ```

To see all available models (as well as other options), run:

```fish
uv run zoo --help
```

## Models in the zoo

TODO: index here

### Base multi-layer perceptrons (MLP)

MLP's are the memory-backbone of modern deep learning architectures, Vollo can
handle all of the things you might need in an MLP, including:

- Basic single layer: [`slp.py`](./vollo_model_zoo/models/slp.py)
- Basic multi-layer: [`mlp.py`](./vollo_model_zoo/models/mlp.py)
- MLP with residuals and RMSNorm: [`mlp-res-rms.py`](./vollo_model_zoo/models/mlp-res-rms.py)

The last of these showcases a variety of activation-functions available on
Vollo, including:

- ReLU
- Sigmoid
- Tanh
- Softplus
- SiLU
- ELU

### Transformer++'s SwiGLU FFN block

This is the feed-forward block, popularized by Llama/Mistral, that you'll find
in many modern transformer architectures. It consists of a linear layer,
followed by a gated activation function (SwiGLU), and then another linear
layer. This block is a key component of the transformer architecture and is
responsible for processing the output of the attention mechanism. You can find
an implementation of this block in
[`ffn-swiglu.py`](./vollo_model_zoo/models/ffn-swiglu.py), which demonstrates
how to implement a fused calculation of the gate/activation.

### ResMLP

TODO:

- ResMLP: <https://arxiv.org/pdf/2105.03404> (this is an MLP-mixer)
- Show softmax

### Mixture of experts (MOE)

### Basic convolutional neural networks (CNN)

## TODO

Before release:

- [ ] License (which one)
- [ ] Can we release the Python SDK as a package (simple execution)
  - [ ] Then we can add some github actions?
- [ ] Generating a latency report for all the models in the zoo
- [ ] Do we want the default config to run v80 and ia-840f?

- CNN:
  - Wavenet
  - Demo `if tracing`
- LSTM:
  - Multilayer
  - Residuals + normalization + FFN (a.k.a Transformer++)
    - Mixed precision (float 8) on the FFN
  - **Find a named model**
- Scan:
  - S3/S4:
    - Float 32 hidden state
  - Multi input/output/state
  - GRU/LSTM to demo full performance vs builtin
  - Mamba1 / Mamba2
  - Other models:
    - RetNet: <https://arxiv.org/abs/2307.08621>
    - xLSTM/mLSTM
    - RWKV 6/7
    - TTT: <https://arxiv.org/pdf/2407.04620>

Gig ideas:

- Strange ensembles?
- Vector quantizer?
- MOE (William shows it is with weight writing!)
