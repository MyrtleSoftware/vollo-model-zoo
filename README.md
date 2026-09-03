# Vollo model zoo

This repo is a collection of example models (written in PyTorch) that you can
compile with the [Vollo SDK](https://vollo.myrtle.ai/latest/installation.html)
to perform low-latency inference on a variety of FPGA accelerators.

Models in the zoo include:

| Category          | Model                                                 | Implementation                                                                                                                                        |
| :---------------- | :---------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Dense**         | [Basic MLPs](#basic-multilayer-perceptrons-mlp)       | [`slp.py`](./vollo_model_zoo/models/slp.py), [`mlp.py`](./vollo_model_zoo/models/mlp.py), [`mlp-res-rms.py`](./vollo_model_zoo/models/mlp-res-rms.py) |
|                   | [SwiGLU FFN](#transformer-swiglu-feed-forward-block)  | [`ffn-swiglu.py`](./vollo_model_zoo/models/ffn-swiglu.py)                                                                                             |
|                   | [ResMLP](#resmlp)                                     | [`resmlp.py`](./vollo_model_zoo/models/resmlp.py)                                                                                                     |
|                   | [Mixture of Experts](#mixture-of-experts-moe-block)   | [`moe.py`](./vollo_model_zoo/models/moe.py)                                                                                                           |
| **Convolutional** | [Basic CNN](#basic-convolutional-neural-networks-cnn) | [`cnn.py`](./vollo_model_zoo/models/cnn.py)                                                                                                           |
|                   | [TCN](#tcn)                                           | [`tcn.py`](./vollo_model_zoo/models/tcn.py)                                                                                                           |
|                   | [WaveNet](#wavenet)                                   | [`wavenet.py`](./vollo_model_zoo/models/wavenet.py)                                                                                                   |
|                   | [MobileNet](#mobilenet)                               | [`mobilenet.py`](./vollo_model_zoo/models/mobilenet.py)                                                                                               |
| **Recurrent**     | [LSTM](#lstm)                                         | [`lstm.py`](./vollo_model_zoo/models/lstm.py)                                                                                                         |
|                   | [GRU](#gru)                                           | [`gru.py`](./vollo_model_zoo/models/gru.py)                                                                                                           |
|                   | [S3/S4/S5 (SSM)](#s3s4s5-state-space-models)          | [`ssm.py`](./vollo_model_zoo/models/ssm.py)                                                                                                           |
|                   | [Mamba](#mamba)                                       | [`mamba1.py`](./vollo_model_zoo/models/mamba1.py)                                                                                                     |
|                   | [Mamba-2](#mamba-2)                                   | [`mamba2.py`](./vollo_model_zoo/models/mamba2.py)                                                                                                     |
| **Attention**     | [Sliding window attention](#sliding-window-attention) | [`swa.py`](./vollo_model_zoo/models/swa.py)                                                                                                           |

See the [quick-start](#-quick-start) section to find out how to run the VM
and calculate the compute-latency for any of these models. Alternatively, have
a read of the [benchmarks](./benchmarks/README.md) for a quick latency
reference.

## ⚡ Quick start

Pre-requisites:

- Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) for dependency management.

`uv run` fetches the Vollo SDK version pinned in [`uv.lock`](./uv.lock), which is
the version every model here is written and measured against — a weekly workflow
moves it to each new release. If you copy a model file into a project on an older
SDK, it may use compiler features that release does not have.

Then try a model out:

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
- Channel mixing MLP:
  - Standard per-token feed-forward network:
  - Linear up-projection.
  - Non-linearity.
  - Linear down-projection.

The Vollo implementation also showcases the GELU activation function implemented
via the common `tanh` approximation.

### Mixture of experts (MoE) block

Code/model: [moe.py](./vollo_model_zoo/models/moe.py)

<p align="center">
  **⚠️ This model is currently experimental ⚠️**
</p>

<p align="center">
  **If you are interested in MoE please contact Myrtle to find out about upcoming improvements**
</p>

Mixture-of-Experts replaces a single feed-forward block with multiple parallel
experts, and a learned gating network that routes tokens to a sparse subset of
them.

A MoE block typically consists of:

1. A gating linear layer that produces routing logits
2. A Top-k selection (usually `k=1` or `k=2`)
3. Several independent expert FFNs
4. A weighted combination of selected expert outputs

This architecture is widely used in SoTA large-scale models such as:

- Switch Transformers
- OpenAI's OSS models

Mixture-of-Experts increase model capacity (parameter count) without increasing
the computational cost per token, by activating only a subset of the experts
for each input. This allows for more efficient scaling of model capacity
compared to dense architectures.

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

### TCN

Code/model: [tcn.py](./vollo_model_zoo/models/tcn.py)

[Temporal Convolutional Networks (TCN)](https://arxiv.org/pdf/1803.01271) are a
1D convolutional architecture designed for sequence modeling. It uses:

1. **Causal Convolutions**: Ensuring that there is no information leakage from
   future to past.
2. **Dilated Convolutions**: Allowing the network to have a large receptive
   field with fewer layers.
3. **Residual Connections**: Helping to train deep networks.

TCNs often outperform RNNs (like LSTMs and GRUs) on a variety of sequence
modeling tasks while being more parallelizable.

### WaveNet

Code/model: [`wavenet.py`](./vollo_model_zoo/models/wavenet.py)

[WaveNet](https://arxiv.org/pdf/1609.03499) was a seminal work from Google
for generating raw audio waveforms which advanced the SoTA in text-to-speech.
WaveNet is a deep convolutional neural network that uses dilated convolutions
to reduce parameter count while maintaining a large receptive field. This is
crucial for the high temporal sampling frequency (kHz) for raw audio.

### MobileNet

Code/model: [`mobilenet.py`](./vollo_model_zoo/models/mobilenet.py)

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

Code/model: [ssm.py](./vollo_model_zoo/models/ssm.py)

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

Code/model: [`mamba1.py`](./vollo_model_zoo/models/mamba1.py)

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
Vollo Mamba state dict see the [the tests](./tests/test_mamba1.py).

Note: when compiled to Vollo the example Mamba will use a `bf16` hidden state,
if this is not accurate enough for your use-case please see
[GRU](./vollo_model_zoo/models/gru.py) or
[Mamba-2](./vollo_model_zoo/models/mamba2.py) as examples of how to modify a
model to use an `fp32` hidden state. In addition you can reach out directly to
Myrtle for support.

### Mamba 2

Code/model: [`mamba2.py`](./vollo_model_zoo/models/mamba2.py)

[Mamba-2](https://arxiv.org/abs/2405.21060) is the second generation of the Mamba
architecture and further develops the selective structured state space model
(SSM) framework introduced in Mamba. It retains the core idea of replacing
quadratic self-attention with a linear-time recurrent state update, while
introducing algorithmic and numerical improvements that make the model easier
to train, more stable, and more efficient on modern hardware.

Conceptually, Mamba-2 reformulates the selective SSM update to better align with
matrix multiplication primitives commonly used in deep learning accelerators.
This allows the recurrent computation to be expressed in a way that improves
parallelism and throughput without sacrificing the linear scaling with sequence
length that characterizes the Mamba family.

Like Mamba, Mamba-2 performs **input-dependent state updates**, enabling
content-aware sequence modeling without explicit attention.

If you would like to see example code to convert an
[FLA](https://github.com/fla-org/flash-linear-attention) Mamba-2 state-dict to a
Vollo Mamba-2 state dict, see [the tests](./tests/test_mamba2.py).

Note: the Vollo Mamba-2 implementation uses an `fp32` hidden state by default to
improve numerical stability during long sequence processing. This mirrors the
reference implementation and helps avoid precision issues that may arise when
using reduced-precision recurrent states.

This model demonstrates how to use the `vollo_torch.CorePartition` API to
explicitly partition the model across Vollo's cores. The heads are split into
`head_partitions` groups, one per accelerator core, with each group's
projections, depthwise convolution and scan wrapped in a
`vollo_torch.CorePartition` so that the cores work on independent heads rather
than sharing one wide recurrence. This reduces cross-core communication. The
final RMS norm and output projection are partitioned along with them, under
`distributed_norm` (on by default).

### Sliding window attention

Code/model: [`swa.py`](./vollo_model_zoo/models/swa.py)

Sliding window (or _local_) attention is a form of attention that streams.
Full self-attention keeps every past timestep resident and re-attends over all
of them, so both the state and the work per new timestep grow with the sequence;
a window bounds the context to the last `window_size` timesteps, which fixes
both. This is the attention primitive in models such as
[Longformer](https://arxiv.org/abs/2004.05150) and
[Mistral](https://arxiv.org/abs/2310.06825), usually interleaved with a few full
attention layers to carry longer-range information.

The Vollo implementation is a pre-norm transformer block: a residual windowed
attention sublayer followed by a residual SwiGLU feed-forward sublayer, each
behind an RMSNorm. The attention is wrapped in a `vollo_torch.nn.Scan` with the
rolling K and V windows held as the scan state; each step evicts the oldest
entry of each window and appends the arriving timestep's, so the compiled
program consumes one timestep per inference and does a fixed amount of work
however long the sequence runs. `SlidingWindowAttention` is the bare attention
layer and is usable on its own -- the norms and the residual adds belong to
`SlidingWindowBlock`, which is what the size sweep measures.

Three details in the file are worth reading for what they say about Vollo:

- **Both attention matmuls take two activations**, because the K and V windows
  are state rather than weights. Those are the
  [dynamic weights](https://vollo.myrtle.ai/latest/data-dimension.html) cases,
  which the file has to opt into, and each wants its matrix operand with the
  contracted dimension second-innermost. The scores contract over features while
  the output contracts over timesteps, which is why only the K window is
  transposed where it is used.
- **The warm-up mask is a placeholder in the scan state.** A query must not
  attend to the window slots no timestep has reached yet, and rather than detect
  emptiness at runtime a third state carries one additive score per slot: `-inf`
  while nothing has been written to that slot, `0` once a real key has slid into
  it, which the softmax turns into a zero weight. The tempting alternative is to
  fold that bias into one extra key feature, weighted by a constant `1` in the
  query, so that it rides through the score matmul that is happening anyway --
  but `dim_head` is normally a multiple of the block size, so the one odd feature
  buys a whole extra block of work across the whole window, and the separate
  state measures faster. `mask` turns it off, for a deployment that is only
  ever read after streaming in a warm-up sequence.
- **Core partitioning pays only if the output projection stays out of it.**
  `head_partitions` splits the heads into groups and pins group `p` -- its
  projections, its window and its attention -- to core `p`. The output
  projection is the one thing that spans the groups, and leaving it _outside_
  the partitioning is what makes the whole thing worthwhile: splitting it too
  gives one partial projection per group plus a sum crossing every core, and
  that sum costs more than the parallelism buys. Hoisting it out is worth 3-7%
  of a partitioned block and turns partitioning from a small spaced-latency
  cost into a 3-6% saving, with 18-25% off `latency_contiguous` as well, on
  both core counts. What crosses cores now is one concatenation of the groups'
  outputs, which the single projection then contracts.

  What decides whether to partition at all is how many heads land on one core:
  one head per core is the happy case, but at three heads per core the spaced
  latency goes up 15-25% instead, so partition as finely as the config allows.
  Every size in the sweep runs six heads for that reason -- one group per core
  on the six-core configs, two heads per core on the three-core one -- and
  `_vm` takes the group count from the config, sweeping every size both ways.
  Weights are unaffected: a checkpoint loads at any partitioning.

The [tests](./tests/test_swa.py) check the streamed window against a dense
band-masked attention over the whole sequence, which is the readable statement
of what the scan state is supposed to be doing, and check that the block wires
its two sublayers the way its docstring draws them.

Note this model needs Vollo SDK 28.0.0 or newer: that is the release where the
compiler gained the dynamic-weight matmuls it is built out of.

## Other utilities

Alongside the primary `zoo` command a few utilities are available to run in
this repo:

- To generate a JSON file containing all the model/configuration combinations
  in the zoo run:

  ```fish
  uv run benchmark --json_output ./benchmarks/my-benchmark.json
  ```

  Left to itself it writes `./benchmarks/vollo_<sdk>+zoo.<zoo>.json`, naming
  both the Vollo SDK it measured and the version of this repo that defined the
  models, since either one moves the numbers.

- To plot multiple benchmark JSON files run:

  ```fish
  uv run plot ./benchmarks/*.json
  ```

- To generate a markdown report from the benchmarks and (optionally) plots run:

  ```fish
  uv run report ./benchmarks/my-benchmark.json --plots ./plots/*.svg
  ```

  This is used to generate the [benchmark README](./benchmarks/README.md).
