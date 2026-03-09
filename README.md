# Vollo model zoo

This repo is a collection of example models (written in PyTorch) that you can
compile with the [Vollo SDK](https://vollo.myrtle.ai/latest/installation.html)
to perform low-latency inference on a variety of FPGA accelerators.

Models in the zoo include:

- [Basic multilayer perceptrons (MLP)](#basic-multilayer-perceptrons-mlp)
- [Transformer++'s SwiGLU feed-forward block](#transformers-swiglu-feed-forward-block)
- [ResMLP (MLP-mixer)](#resmlp)
- [Mixture of experts (MoE)](#mixture-of-experts-moe-block)
- [Basic Convolutional neural networks (CNN)](#basic-convolutional-neural-networks-cnn)
- [WaveNet](#wavenet)
- [MobileNet](#mobilenet)
- [Long short-term memory (LSTM)](#lstm)
- [Gated recurrent units (GRU)](#gru)
- [State space models (S3/S4/S5)](#s3s4s5)
- [Mamba](#mamba)

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

   Or similarly for `bash`:

   ```bash
   export UV_FIND_LINKS=/path/to/sdk/vollo-sdk-<version>/python/
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

Some of these are _first-class_ (i.e. have hardware support) whilst others are
composed of simpler operations.

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
demonstrate how to implement a fused calculation of the gate/value activation
(as would often be done to optimize a GPU program) to highlight how this
premature optimization can actually slow a Vollo program.

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

Vollo has comprehensive first-class support for 1D causal convolutions.

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
depthwise-pointwise factorisation pattern in 1D.

### LSTM

Code/model: [`lstm.py`](./vollo_model_zoo/models/lstm.py)

Long Short-Term Memory (LSTM) is a recurrent neural network (RNN) architecture
that uses a series of gates to control the flow of information, allowing it to
capture long-term dependencies in sequential data while mitigating the
vanishing gradient problem common in vanilla RNNs.

A standard LSTM cell consists of:

1. **Forget gate**: Decides what information to discard from the cell state.
2. **Input gate**: Decides what new information to store in the cell state.
3. **Output gate**: Decides what part of the cell state to output.

Vollo has first-class support for `torch.nn.LSTM`, allowing for efficient
streaming inference of multi-layer, biased, and batch-first LSTM models.

### GRU

Code/model: [`gru.py`](./vollo_model_zoo/models/gru.py)

Gated Recurrent Unit (GRU) is a recurrent neural network (RNN) architecture
designed to capture dependencies at different time scales. It simplifies the
standard LSTM architecture by merging the cell state and hidden state, and
using fewer gates.

A GRU cell consists of:

1. **Reset gate**: Determines how much of the past information to forget.
2. **Update gate**: Controls how much of the previous state is carried over to
   the current state.

Through the scan API, Vollo can efficiently perform streaming inference for GRU
models, allowing for high-performance recurrent computations. In addition, the
GRU example demonstrates how to use select `fp32` operations to keep the hidden
state in full precision, as is often required to prevent numerical errors
accumulating over long sequences.

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

Whare `h` is a hidden state, `x` is the input, and `y` is the output. Through
the scan API Vollo can efficiently perform inference through the recurrent
formulation. The Vollo exemplar is fully general in the parameterization of
`A..D`.

### Mamba

Code/model: ['mamba1.py'](./vollo_model_zoo/models/mamba1.py)

[Mamba](https://arxiv.org/pdf/2312.00752) is a modern selective structured
SSM that replaces self-attention with a learned, input-dependent recurrent
mechanism. Unlike transformers, which rely on quadratic-cost attention over the
full sequence, Mamba achieves linear-time complexity in sequence length while
maintaining strong long-range modeling capability.

At a high level, Mamba can be understood as a selective state space model: the
state update and output projection are dynamically modulated by the input at
each time step, allowing content-based reasoning without explicit attention.

If you would like to see example code to convert an
[FLA](https://github.com/fla-org/flash-linear-attention) Mamba state-dict to a
Vollo Mamba state dict see the [the tests](./tests/test_mamba.py).

Note: when compiled to Vollo the example Mamba will use a `bf16` hidden state,
if this is not accurate enough for your use-case please see
[GRU](./vollo_model_zoo/models/gru.py) or
[Mamba-2](./vollo_model_zoo/models/mamba2.py) as examples of how to modify a
model to use an `fp32` hidden state. In addition you can reach out directly to
Myrtle for support.

### Mamba 2

## TODO

Before release:

- [ ] License (which one)
- [ ] Can we release the Python SDK as a package (simple execution)
  - [ ] Then we can add some github actions?
- [ ] Generating a latency report for all the models in the zoo
- [ ] Do we want the default config to be v80, LL or something else?
- [ ] Integrate testing with myrtlepkgs

- Features:
  - Show softmax
  - Demo `if tracing`
  - FP8 support

- LSTM:
  - **Find a named model**

- Scan:
  - Multi input/output/state
  - Other models:
    - RetNet: <https://arxiv.org/abs/2307.08621>
    - xLSTM/mLSTM
    - RWKV 6/7
    - TTT: <https://arxiv.org/pdf/2407.04620>

Big ideas:

- Strange ensembles?
- Vector quantizer?
