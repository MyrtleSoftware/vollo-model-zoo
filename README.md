# Vollo model zoo

This repo is a collection of example models (written in PyTorch) that you can
compile with the [Vollo SDK](https://vollo.myrtle.ai/latest/installation.html)
to perform low-latency inference on a variety of FPGA accelerators.

Models in the zoo include:

- [Multilayer perceptrons (MLP)](#basic-multilayer-perceptrons-mlp)
- [Transformer++'s SwiGLU feed-forward block](#transformers-swiglu-feed-forward-block)
- [ResMLP/MLP-mixer](#resmlp)
- [Mixture of experts (MoE) blocks](#mixture-of-experts-moe-block)
- [Convolutional neural networks (CNN)](#basic-convolutional-neural-networks-cnn)
- [WaveNet](#wavenet)
- [MobileNet](#mobilenet)

See the [quick-start](#-quick-start) section to find out how to run the VM and
calculate the compute-latency for any of these models.

## ⚡ Quick start

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

## 🐘 Models

### Basic multilayer perceptrons (MLP)

Code/models:

- [`slp.py`](./vollo_model_zoo/models/slp.py)
- [`mlp.py`](./vollo_model_zoo/models/mlp.py)
- [`mlp-res-rms.py`](./vollo_model_zoo/models/mlp-res-rms.py)

Multilayer perceptrons are the memory-backbone of modern deep learning
architectures, an MLP layer/block at its core is a combination of:

1. A linear layer (`Wx + b`)
2. A non-linear activation function.

Vollo can handle all the things you might need in an MLP, including
normalization and residual connections. In addition,
[mlp-res-rms](./vollo_model_zoo/models/mlp-res-rms.py) showcases a variety of
activation-functions available on Vollo, including:

- ReLU
- Sigmoid
- Tanh
- Softplus
- SiLU
- ELU

### Transformer++'s SwiGLU feed-forward block

Code/models: [ffn-swiglu.py](./vollo_model_zoo/models/ffn-swiglu.py)

This is the feed-forward block, popularized by
[Llama](https://arxiv.org/abs/2407.21783)/Mistral, that you'll find in many
modern transformer architectures. It consists of:

1. An up-projecting linear layer
2. A gated activation function (SwiGLU),
3. A final down-projecting linear layer.

This block is a key component of the transformer architecture and is
responsible for processing the output of the attention mechanism. In our
[implementation](./vollo_model_zoo/models/ffn-swiglu.py) of this block we
demonstrate how to implement a fused calculation of the gate/value activation.

### ResMLP

Code/model: [resmlp.py](./vollo_model_zoo/models/resmlp.py)

[ResMLP](https://arxiv.org/pdf/2105.03404) is a pure MLP-based architecture
inspired by the "MLP-Mixer" family of models. It removes convolutions and
self-attention entirely, replacing them with stacked residual MLP blocks that
mix information across tokens and channels using only linear layers and
non-linearities. ResMLP attains SoTA accuracy/complexity trade-offs on
fixed-input-length tasks like ImageNet.

A typical ResMLP block consists of two residual sublayers:

- Token mixing MLP:
  - Operates across the sequence (or patch) dimension.
  - Implemented as a linear projection over tokens.
  - Often followed by a softmax or affine scaling to stabilise mixing.
- Channel mixing MLP:
  - Standard per-token feed-forward network:
  - Linear up-projection.
  - Non-linearity.
  - Linear down-projection.

The Vollo implementation also showcases the GELU activation function implemented
via the common `tanh` approximation.

### Mixture of experts (MoE) block

Code/model: [moe.py](./vollo_model_zoo/models/moe.py)

Mixture-of-Experts replaces a single feed-forward block with multiple parallel
experts, and a learned gating network that routes tokens to a sparse subset of
them.

A MoE block typically consists of:

1. A gating linear layer that produces routing logits
2. A Top-k selection (usually k=1 or 2)
3. Several independent expert FFNs
4. A weighted combination of selected expert outputs

This architecture is widely used in SoTA large-scale models such as:

- Switch Transformers
- OpenAI's OSS models

Mixture-of-Experts increase model capacity (parameter count) without increasing
the computational cost per token, by activating only a subset of the experts
for each input. This allows for more efficient scaling of model capacity
compared to dense architectures.

TODO: the MoE block's latency in the zoo is a strong function of the number of
experts, we could improve vollo to fix this?

### Basic convolutional neural networks (CNN)

Code/models: [cnn.py](./vollo_model_zoo/models/cnn.py)

Convolutional Neural Networks are designed for spatially structured inputs
(images, spectrograms, feature maps).

A standard CNN block typically consists of:

1. Convolution layer
2. Activation
3. Normalization
4. Optional residual connection

Vollo has comprehensive support for 1D causal convolutions.

### WaveNet

Code/model: [`wavenet.py`](./vollo_model_zoo/models/wavenet.py)

[WaveNet](https://arxiv.org/pdf/1609.03499) was a seminal work from Google
for generating raw audio waveforms which advanced the SoTA in text-to-speech.
WaveNet is a deep convolutional neural network that uses dilated convolutions
to reduce parameter count while maintaining a large receptive field. This is
crucial for the high temporal sampling frequency (kHz) for raw audio.

### MobileNet

Code/model: ['mobilenet.py'](./vollo_model_zoo/models/mobilenet.py)

MobileNet is a family of efficient convolutional neural networks designed for
low-latency and resource-constrained environments. The core architectural idea
is the use of depthwise separable convolutions, which significantly reduce
compute and parameter count compared to standard convolutions. This is done by
factorizing a standard convolution into two separate layers:

- Depthwise convolution (i.e. `groups == in_channels == out_channels`)
- Pointwise 1×1 convolution (i.e. linear layers)

In the Vollo model zoo implementation we focus on the canonical
depthwise–pointwise factorisation pattern in 1D.

### LSTM

### GRU

### S3/S4/S5

Code/model: ['ssm.py'](./vollo_model_zoo/models/ssm.py)

State space (sequence) models (SSM) (later expanded to _simple_ and
_structured_) are discretizations of linear time-invariant systems:

```math
\begin{aligned}
h'(t) &= A h(t) + B x(t) \\
y(t) &= C h(t) + D x(t)
\end{aligned}
```

where `h` is the hidden state, `x` is the input, and `y` is the output. The

### Mamba

Code/model: ['mamba1.py'](./vollo_model_zoo/models/mamba1.py)

[Mamba](https://arxiv.org/pdf/2312.00752) is a modern selective structured
SSM that replaces self-attention with a learned, input-dependent recurrent
mechanism. Unlike transformers, which rely on quadratic-cost attention over the
full sequence, Mamba achieves linear-time complexity in sequence length while
maintaining strong long-range modeling capability.

At a high level, Mamba can be understood as a selective state space model: the
state update and output projection are dynamically modulated by the input at
each timestep, allowing content-based reasoning without explicit attention.

The Mamba block incorporates

## TODO

Before release:

- [ ] License (which one)
- [ ] Can we release the Python SDK as a package (simple execution)
  - [ ] Then we can add some github actions?
- [ ] Generating a latency report for all the models in the zoo
- [ ] Do we want the default config to run v80 and ia-840f?

- Show softmax

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

Big ideas:

- Strange ensembles?
- Vector quantizer?
- MOE (William shows it is with weight writing!)

```

```
