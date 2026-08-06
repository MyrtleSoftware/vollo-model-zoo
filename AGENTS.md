# AGENTS.md — Vollo Model Zoo (VMZ)

Guidance for AI agents (and humans) working in this repo.

## What this repo is

A collection of **self-contained PyTorch example models** that compile with the
[Vollo SDK](https://vollo.myrtle.ai/latest/installation.html) to run
low-latency inference on FPGA accelerators. Each model is a reference
implementation _plus_ a set of size configurations whose compute latency is
measured on Vollo's VM (no hardware needed).

There are two audiences; keep both in mind when changing anything:

1. **Customers** experimenting with Vollo — they copy-paste a single model file
   into their own code, or run `uv run zoo <model>` to see what latency a shape
   would achieve. Their needs: files that stand alone, honest numbers, README
   prose that explains the architecture.
2. **Myrtle devs** adding/optimizing models — they add model files, tune them
   against Vollo's execution model, and re-run the zoo against unreleased SDK
   builds to catch regressions and exploit new compiler features.

Everything here runs against the **compiler + VM only**. No FPGA, no
`vollo-rt`, no `.vollo` program artifacts are produced or checked in.

## Setup and commands

Requires [`uv`](https://docs.astral.sh/uv/) and Python 3.13 (`.python-version`).
`uv run` bootstraps the venv from `uv.lock` — never `pip install` into it by hand.

```fish
uv run zoo <model>                 # latency table for one model (default config: V80)
uv run zoo <model> --config V80LL  # pick a hardware config
uv run zoo <model> -j              # machine-readable JSON
uv run zoo --help                  # list all models + configs
uv run zoo moe --experimental      # experimental models need this flag (see below)

uv run pytest                      # full suite; slow — dominated by model x config
uv run pytest -k "test_models and mamba2"   # scope it while iterating
uv run pytest tests/test_gru.py

uv run pre-commit install          # once; hooks = isort, black, codespell
uv run pre-commit run --all-files  # lint
```

`ci.yml` runs `uv run --all-extras pre-commit run --all-files` then
`uv run --all-extras pytest -x`. Note the `-x`: a red CI reports only the
_first_ failure, so never read "everything else passed" into it — re-run the
suite locally without `-x` to get the full picture.

A model file can also be run directly (`uv run python vollo_model_zoo/models/mlp.py`)
thanks to its `__main__` block — useful when iterating on one model.

## Running against a different Vollo SDK

`vollo-compiler` / `vollo-torch` come from PyPI, pinned by `uv.lock`. Two
things override that: a released `.run` bundle, or a locally built (unreleased)
SDK. Both work the same way — point uv at a directory of wheels.

### Building an unreleased SDK from a myrtlepkgs checkout (Myrtle devs)

```fish
nix-build ~/myrtlepkgs -A vollo2.release.vollo-sdk-release --out-link /tmp/vollo-sdk-release
```

The result contains an already-extracted `vollo-sdk-<version>/` tree (plus the
`vollo-sdk-<version>.run` self-extractor). Wheels live in
`vollo-sdk-<version>/python/`: `vollo_compiler`, `vollo_torch` (both needed
here) and `vollo_rt` (not used by the zoo).

For a _released_ SDK instead, download `vollo-sdk-<version>.run` from
<https://github.com/MyrtleSoftware/vollo-sdk/releases> and extract it
(`./vollo-sdk-<version>.run --target <dir> --noexec --accept`); it has the same
`python/` layout.

### Point the zoo at those wheels

Preferred, because it leaves the tracked `uv.lock` untouched:

```fish
set -x SDK /tmp/vollo-sdk-release/vollo-sdk-29.0.0   # adjust version

uv pip install --find-links $SDK/python \
  --reinstall-package vollo-compiler --reinstall-package vollo-torch \
  vollo-compiler==29.0.0 vollo-torch==29.0.0

uv run --no-sync zoo mlp
uv run --no-sync pytest -k "test_models and mlp"
```

**`--no-sync` is required.** A plain `uv run` re-syncs the venv from `uv.lock`
and silently reinstalls the PyPI-pinned version, so results would come from the
old compiler with no warning. Confirm what you are actually measuring:

```fish
uv run --no-sync python -c "import importlib.metadata as m; print(m.version('vollo-compiler'))"
```

### How CI does it instead

`update_benchmarks.yml` uses a different mechanism, worth knowing when you are
reproducing or debugging one of its PRs: it sets `UV_FIND_LINKS` to the
extracted SDK's `python/` directory for the whole job and then lets a normal
`uv sync` resolve against it:

```fish
set -x UV_FIND_LINKS (pwd)/build/vollo-sdk/python/

uv sync --all-extras --dev \
  --upgrade-package vollo-compiler --upgrade-package vollo-torch \
  --reinstall-package vollo-compiler --reinstall-package vollo-torch
```

The trade-off is the mirror image of the recipe above: `--upgrade-package`
re-resolves and **rewrites `uv.lock`** to the new version, which is why later
`uv run`s don't need `--no-sync` — the lock itself now names the new SDK, and
`UV_FIND_LINKS` keeps the wheels findable. Use this when you want CI's exact
behaviour and don't mind a dirty lock file (CI's `add-paths` never commits it);
use `uv pip install` above when the lock must stay untouched.

### What to expect from a newer SDK

- `$SDK/CHANGELOG.md` is the first place to look when a model's latency or
  compile behaviour changes.
- `$SDK/docs/mdbook/src/` holds that build's full documentation (see
  "Where to look things up").

## Bump the zoo version when you change what is measured

A latency number is a function of two things: the compiler that produced it and
the zoo code that defined the model. So a benchmark run is identified by both,
as a PEP 440 version `<vollo>+zoo.<zoo>` (`vollo_model_zoo/version.py`), which
is the JSON filename _and_ its top-level key:

```
benchmarks/vollo_28.1.0+zoo.0.2.0.json   # {"28.1.0+zoo.0.2.0": {model: {config: [...]}}}
```

**The zoo version is `[project] version` in `pyproject.toml`, and nothing bumps
it for you.** `update_benchmarks.yml` decides whether to work by asking whether
that exact filename already exists, so an unbumped version means the weekly job
sees a run it thinks it already has and skips — your model change is never
measured. Bump it in the same PR as:

- any change under `models/` that moves a number: the architecture, a sweep in
  `main()`, a constructor default;
- a change to `vm.py`, which sets the compile options every model is measured
  with;
- a dependency change that alters compilation.

Don't bump for README prose, tests, or the reporting tooling (`plot.py`,
`report.py`, `parse.py`) — those can't move a latency, and a bump would spend a
full benchmark run to reproduce numbers we already have. A minor bump is the
normal case; the number carries no compatibility promise, it just has to
increase.

Two consequences worth knowing:

- The project's own version is recorded in `uv.lock`, so a bump re-locks it —
  commit that hunk alongside the `pyproject.toml` one.
- Only the latest SDK gets re-measured, so the new point lands beside the old
  ones rather than replacing them. That is deliberate (the history is a record
  of what was measured, not a claim that every point used today's models), and
  it is why plot labels name both versions. To rebuild the whole curve against
  current models, run `scripts/backtest_github_versions.sh` over the released
  SDKs by hand.

## Layout

```
vollo_model_zoo/
  models/*.py     one file per model — the actual content of the zoo
  vm.py           compile + measure harness (vollo_info, CONFIGS, Result types, vollo_fn)
  zoo.py          `zoo` CLI: run one model, print markdown table or JSON
  benchmark.py \
  parse.py       | CI latency-history tooling (`benchmark`, `plot`, `report`
  plot.py        | commands) feeding benchmarks/ and plots/. Generated by a
  report.py    /  weekly workflow; not part of model development.
  version.py      what identifies a benchmark run (see "Bump the zoo version")
tests/            see "Tests"
benchmarks/, plots/  generated latency history (JSON + SVG + report); do not hand-edit
scripts/          hand-written helper: backtest_github_versions.sh runs the
                  benchmarks against a list of released SDK versions
.github/workflows ci.yml (lint + tests), update_benchmarks.yml (weekly benchmark PR)
```

## The model file contract

Read `CONTRIBUTING.md` first: **we follow the Hugging Face single-file model
style.** Each model file must be self-contained and copy-pasteable, and must
not import `vollo_model_zoo` components at module level. This is a hard rule —
it is what makes the zoo useful to customers.

Every file in `vollo_model_zoo/models/` follows the same shape (see
`mlp.py` for the minimal case, `mamba2.py` for the involved one):

```python
class MyModel(nn.Module):        # public, plain PyTorch, @beartype'd __init__
    ...                          # helper submodules are _private

@beartype
def _vm(<hyperparams>, config: str):
    from vollo_model_zoo.vm import vollo_info   # deferred import: keeps the file standalone
    input = torch.randn(...)                    # a representative single-batch input
    model = MyModel(...)
    return vollo_info(
        model, input,
        config=config,
        time_axis=<int | None>,
        meta=dict(...),        # hyperparams shown in the CLI table / benchmark metadata
    )

@beartype
def main(config: str = "V80") -> Generator:    # discovered and called by vm.get_results
    for x in [dict(...), dict(...)]:           # size sweep, smallest first
        yield _vm(**x, config=config)

if __name__ == "__main__":
    print(f"Model '{Path(__file__).stem}':")
    for result in main():
        print(f"\t{result}")
```

Discovery is by filename: `vm.get_models()` lists `models/*.py` not starting
with `__`, and `vm.get_results()` `importlib`-imports the module and calls
`main(config=...)`. Consequences:

- The **filename is the CLI model name** (`mlp-res-rms.py` → `uv run zoo mlp-res-rms`).
- Hyphenated filenames are importable only via `importlib`, so tests and other
  code must use `importlib.import_module` for those (`test_gru.py` etc. import
  the underscore-free ones directly).
- `main` must be a generator yielding `Result`s and must accept `config` as a
  keyword argument.

### Invariants `tests/test_models.py` enforces on every model

- At least one yielded `Ok` result per model.
- The sweep is monotonic: results sorted by `param_count` **or** by
  `latency_spaced` (increasing). Order your `main()` list accordingly.
- At least one configuration lands in **0.95M–1.05M parameters** — the
  "baseline" size models are compared at. `lstm.py` shows the arithmetic being
  done in a comment; do the same.
- No `AllocationError`/`SaveError` on the default, `V80` or `V80LL` configs.
  Not fitting on the smaller boards (`IA-420f`, etc.) is tolerated and simply
  omitted from results.
- **A `ValueError` result fails the test on _every_ config**, small boards
  included — the `case _` arm re-raises. `vollo_info` catches `ValueError`
  alongside the allocation errors, so "the compiler rejected this model" is a
  returned value like any other, but unlike "too big" it is never tolerated.
  Don't read the previous bullet as blanket permission for non-`Ok` results on
  `IA-420f`.

## Vollo's execution model

The compiler constraints in this section are what make a model fast, slow, or
uncompilable.

> **The SDK docs are the authority; this section is a map, not a copy.**
> <https://vollo.myrtle.ai/latest/> tracks the installed compiler. Op support,
> the data-dimension algebra and measured latencies all change between
> releases, so this file states **which page answers a question** and only
> keeps detail that is specific to _this repo_ (which model sets which flag,
> which idiom we follow). Resist the urge to paste a support table or a latency
> figure back in — an out-of-date copy here is worse than no copy, because it
> gets trusted. If you catch this file contradicting the SDK docs, fix this
> file.

### Streaming

`vollo_info(..., time_axis=N)` runs `nnir.streaming_transform(N)`, converting a
model written over a whole sequence into a **stateful program that consumes one
timestep per inference**. Pass the axis of `input` that is time. This covers
every model in the zoo, so use it to place a new one:

- `1` for `[B, T, F]`: `slp`, `mlp`, `mlp-res-rms`, `lstm`, `ffn-swiglu`, `moe`
- `2` for conv models with `[B, C, T]`: `cnn`, `tcn`, `wavenet`, `mobilenet`
- `0` for `[T, D]` step-style models: `gru`, `ssm`, `mamba1`, `mamba2`
- `None` for genuinely non-streaming models: `resmlp`, which annotates why

Two `vollo_torch` building blocks make a model streaming-transformable;
[example-2-cnn](https://vollo.myrtle.ai/latest/example-2-cnn.html) and
[example-3-lstm](https://vollo.myrtle.ai/latest/example-3-lstm.html) are the
canonical write-ups. What the zoo adds on top:

- **`vollo_torch.nn.PaddedConv1d`** replaces `torch.nn.Conv1d`, which the
  streaming transform cannot handle. It wants time as the rightmost dimension —
  hence the `transpose` sandwiches in `mamba1.py` / `mamba2.py`.
- **`vollo_torch.nn.Scan`** wraps a `step` module the compiler can recognise as
  a repeated application. Zoo convention: initial state is an
  `nn.Buffer(..., persistent=False)` so `.to()` works without polluting the
  state dict, and multiple inputs/states go in as lists (`mamba2.py`).
  `torch.nn.LSTM` is supported natively and needs no `Scan` (`lstm.py`).

### The data dimension (the `!` in shape comments)

Vollo's compute units operate on contiguous vectors, so every activation tensor
has exactly one **data dimension**; compile-time constants (weights) have none.
When a compile fails on a shape, this is usually why.

**Read [data-dimension](https://vollo.myrtle.ai/latest/data-dimension.html) —
don't guess, and don't expect this file to list the rules.** That page is
organised per-op (pointwise, slicing, unsqueeze, broadcasting, concatenation,
reduction, matrix multiplication incl. dynamic weights, transpose, reshape), so
go straight to the op that failed. The cases that most often bite zoo models
are matmul contraction and reshape, both of which have worked examples there.

The repo-side convention: `[a b! c]` marks `b` as the data dimension, and
`mamba2.py` tracks `!` in a comment on every intermediate. Keep that up in
shape-juggling code — it is the fastest way to see why a compile failed, and it
survives SDK changes in a way a prose summary doesn't.

`allow_dynamic_weights=True` unlocks the matmul cases the default contraction
rule rejects (see that page's "Dynamic weights" section for exactly which).
It is why the scan-based and large-linear models set it. Treat it as advanced:
expect higher latency and more tensor RAM than the contracted-data-dimension
form, so don't enable it reflexively — reach for it when a matmul won't compile,
not to make one faster.

### Precision

Default compute is **bf16**, with wider accumulation inside dot products.
Mixed precision has one authoritative page each:

- **Which ops can run in fp32** — the "Fp32 Support" column of
  [supported-models](https://vollo.myrtle.ai/latest/supported-models.html).
  Check it there every time; the set grows between releases and the answer is
  release-specific.
- **How to apply it** — `Fp32Activations`, `Fp8Weights`, and input/output
  precisions are all in
  [example-4-mixed-precision](https://vollo.myrtle.ai/latest/example-4-mixed-precision.html).

What that means here:

- `vollo_torch.Fp32Activations()` is the lever for keeping recurrent state
  accurate over long sequences (`gru.py`'s `fp32`, `mamba2.py`'s `ssm_fp32`).
  Zoo convention: expose it as a constructor flag and sweep both settings in
  `main()` so the latency cost is visible in the table.
- Model inputs/outputs are bf16 unless `to_nnir` is given `inputs_precisions` /
  `outputs_precisions` — `vm.vollo_fn` is the one place in the repo that does.
- `vollo_torch.Fp8Weights()` halves weight storage on the Versal boards (V80,
  V80LL).
- `gru.py` shows an accuracy trick (`sigmoid(-z)` instead of `1 - sigmoid(z)`);
  `tests/test_approx.py` shows how to build higher-precision primitives
  (Newton–Raphson reciprocal) and measure their bit accuracy against ideal
  bf`N`. Both are repo-grown, not from the SDK docs.

### Cost model

- Configs are `<cores>×<block size>`: `c6b32` (V80, V80LL, IA-420f, NT400D11)
  or `c3b64` (IA-840f). Feature dims that are multiples of the block size map
  cleanly onto the compute units, which is why sweeps use sizes like
  `32 * 6 * 4`.
- The VM is an instruction-level simulation — the zoo describes it as **near
  cycle-accurate** — and it models **no host↔accelerator IO**. A round trip
  costs single-digit microseconds, varying with transport and tensor sizes; get
  the current numbers from
  [benchmark-io](https://vollo.myrtle.ai/latest/benchmark-io.html) rather than
  from any figure quoted here. Note the implication: the smallest zoo models
  compile to well under 1 µs of compute, so for those **IO dominates end-to-end
  latency**. Zoo numbers are compute-only estimates for comparing models and
  shapes — never end-to-end guarantees. Every surface that prints them repeats
  that caveat (`zoo.py`, `report.py`, `benchmarks/README.md`) in the same
  wording; keep it there and keep it consistent.
- On-hardware latencies for comparable models live in
  [benchmark-mlp](https://vollo.myrtle.ai/latest/benchmark-mlp.html) /
  [-cnn](https://vollo.myrtle.ai/latest/benchmark-cnn.html) /
  [-lstm](https://vollo.myrtle.ai/latest/benchmark-lstm.html) — the sanity check
  when a VM estimate looks too good.
- **Fusing is not always faster.** `ffn-swiglu.py` deliberately sweeps
  `fuse=True/False` to show that a GPU-style fused gate/value projection can
  _slow down_ a Vollo program. Preserve that kind of pedagogical pairing.
- `quick_compile=True` skips optimization passes to cut compile time (`mamba1`,
  `mamba2` use it); drop it when chasing the last few percent of latency.

## `vm.py` API

- `CONFIGS`: `{"V80", "V80LL", "IA-420f", "IA-840f", "NT400D11"}` → `vc.Config`,
  built with `hasattr` probes so the repo still works against older SDKs that
  lack a config. Add new configs the same defensive way.
- `vollo_info(model, x, *, time_axis, config, meta=None, allow_dynamic_weights=False, quick_compile=False) -> Result`
  Calls `model(x)` first (nicer errors), then `prepare_shape` → `to_nnir` →
  optional `streaming_transform` → `to_program` → `pack()`. Returns `Ok` or
  the caught `AllocationError | SaveError | ValueError` — compile failures are
  _values_, not exceptions, so a sweep can report "too big" per size.
- `Ok`: `config`, `param_count`, `cycle_count`, `latency_spaced`,
  `latency_contiguous` (both `Microseconds`), `meta`. `latency_spaced` =
  isolated inferences (`compute_duration_per_inference_us(spaced=True)`);
  `latency_contiguous` = back-to-back. `meta` keys prefixed with `_` are hidden
  from the `zoo` table.

Features the harness deliberately doesn't expose (reach for the compiler API
directly if you need them): `program.metrics()` for static resource usage,
`ProgramBuilder` for multi-model programs, `generate_state_reset=True` for
resettable state, and `program.save()`/`vollo-onnx` for producing `.vollo`
files.

## Debugging a model: observing the NNIR

Mostly for Myrtle devs. `vollo_info` is a black box that returns a latency or an
error, which is the wrong granularity when a model won't compile or is
mysteriously slow. **Drop below the harness and print the NNIR** — it is the
compiler's own view of your model, and it answers most questions in one look.

First, know that **`zoo` hides failures**: `run_model` skips every non-`Ok`
result, so a size that fails to compile just _silently vanishes from the table_.
Use `-j` to see it (`to_dict` renders `{"ValueError": "<message>"}`), or go
manual:

This runs as-is (`uv run python <file>`); swap in the model you're debugging.
Hyphenated model files need `importlib.import_module`.

```python
import torch, vollo_compiler as vc, vollo_torch as vt
from vollo_model_zoo.models.mlp import MLP

model = MLP(num_layers=2, in_features=64, out_features=64, hidden_features=64)
x = torch.randn(1, 5, 64)

model, _ = vt.fx.prepare_shape(model, x)   # torch.fx trace + shape propagation
nnir = vt.fx.nnir.to_nnir(model)
print(nnir)                                # <-- the whole point
streamed, out_axis = nnir.streaming_transform(1)   # same time_axis as _vm passes
print(streamed)                            # <-- diff against the above
program = streamed.to_program(vc.Config.v80_c6b32())
program.pack()                             # AllocationError surfaces here, not earlier
```

`print(nnir)` (its `__str__`/`__repr__`) lists the inputs, the outputs, and one
line per node:

```
Id(1v2) - (linear) Linear(weight: (128, 64), weight_precision: Bf16, bias: 128, input: Id(0)) - [1, 4, 128]
Id(2v2) - (relu) Clamp(min: 0, max: None, input: Id(1), activation_precision: Bf16) - [1, 4, 128]
```

i.e. node id, the originating **fx name in parentheses** (`(linear)`, `(relu)` —
this is your link back to the Python source), the NNIR op with its constant
shapes and precision, and the output shape. What to use it for:

- **Verify a precision context actually applied.** Every node carries
  `activation_precision:` and every weighted node a `weight_precision:`. Wrapping
  a region in `Fp32Activations()` flips its nodes to `F32`
  (`Pointwise(op: Mul { activation_precision: F32 }, ...)`) and leaves everything
  else `Bf16`. This is the only reliable way to check a `with` block covers the
  ops you meant — and when a node inside the block stays `Bf16`, that op has no
  fp32 support, so go read the support table rather than arguing with the
  compiler.
- **Verify the streaming transform.** Print before and after: the time axis
  should disappear from every shape (`[1, 4, 64] → [1, 64]`). If a shape keeps
  its time extent, that node didn't stream.
- **See what your PyTorch actually lowered to.** Ops fuse and rename — `relu`
  becomes `Clamp`, and `nn.Linear` becomes `Linear` with the bias inlined. When
  the data dimension complains about an op you didn't think you wrote, this is
  where you find it.

Two caveats: the `vN` suffix in `Id(0v2)` is a graph version, and it changes as
the graph is transformed (the same node is `Id(0v2)` before the streaming
transform and `Id(0v9)` after), so diff on structure and shapes, never on ids;
and `to_nnir` aggregates every unsupported op into one plain
**`ValueError`** (not `UnsupportedOperationError`) whose message is already the
answer:

```
Unsupported operations encountered translating to Vollo NNIR:
  unsupported function 'fft_fft': found 1
  unsupported attribute 'real': found 1
```

Cross-reference those names against
[supported-models](https://vollo.myrtle.ai/latest/supported-models.html). This is
the error `vollo_info` returns as a value and that `test_models.py` re-raises on
every config.

### One level further out: the fx graph

If `to_nnir` fails before producing anything, the torch.fx graph is what it was
reading:

```python
print(model.graph)  # after prepare_shape; print_tabular() needs `tabulate`, not a dep here
for n in model.graph.nodes:
    print(n.op, n.target, n.meta["tensor_meta"].shape, n.meta["vollo_fp32_activations"])
```

`prepare_shape` hangs a `tensor_meta` on every node (shape, dtype, stride) plus
the `vollo_fp8_weights` / `vollo_fp32_activations` / `vollo_core_partition` keys
that the precision context managers set — so you can confirm a context reached a
node even before NNIR exists. `vt.fx.save(...)` / `vt.fx.load(...)` archive a
traced module, which is the tidiest way to hand a repro to someone else.

### Resource usage and where the cycles go

`program.metrics()` is the tool for an `AllocationError`. Its `repr` is useless;
read the named fields, most of which are **per-core lists** (six entries on
`c6b32`, three on `c3b64`):

```python
m = program.metrics()
m.tensor_ram_used, m.tensor_ram_depth       # [384, 0, 0, 0, 0, 0], 262144
m.weight_store_used, m.weight_store_depth   # per-core used vs total available
m.num_instrs, m.num_micro_instructions      # per-core instruction counts
m.input_bytes, m.output_bytes               # per-model IO, per inference
```

`_used` against `_depth` tells you which store you blew and by how much; all the
load sitting on core 0 with the rest at zero tells you the model isn't spread
across cores.

## Tests

- `test_models.py` — the bulk test: every model × (default + all 5 configs),
  asserting the invariants listed above. Slow; scope with `-k`.
- `test_gru.py` — numerical equivalence of the zoo GRU against `torch.nn.GRU`,
  including the **state-dict conversion** from PyTorch layout to the zoo's
  split-linear layout.
- `test_mamba1.py` / `test_mamba2.py` — equivalence against
  [FLA](https://github.com/fla-org/flash-linear-attention) reference layers,
  again with a `convert_state_dict` helper. Skipped without CUDA (FLA needs
  triton). The README points customers at these files as the canonical
  "how do I load my trained weights into the zoo model" example — keep them
  readable and keep them working.

When you add a model that has an established reference implementation, add an
equivalence test with a `convert_state_dict` in the same style.

## Conventions and gotchas

- `@beartype` on public constructors and module-level functions; type hints
  everywhere. `beartype.typing` and `collections.abc` are both used for
  `Generator`/`Optional` — match the file you're in.
- Formatting is `black` + `isort`; `codespell` runs over all text with
  `vollo` allow-listed.
- Experimental gating is **hardcoded in `zoo.py`** (`if args.model.lower() == "moe"`).
  The notice prints unconditionally; only the early `return 1` is conditional on
  `--experimental`. If you mark another model experimental, update that check
  _and_ the README banner; there is no per-model metadata for it yet.
- Docstrings state tensor shapes on `forward`. Do this; it is how readers
  navigate the streaming and data-dimension rules.
- README table + per-model prose section: adding a model means adding both, with
  links to the file. The prose explains the architecture _and_ what Vollo
  feature the file demonstrates.

## Checklist: adding a model

1. `vollo_model_zoo/models/<name>.py` following the contract above; no
   `vollo_model_zoo` imports outside `_vm`.
2. Choose a sweep for `main()`: monotonic in params or latency, including one
   ~1M-parameter size; expose interesting Vollo trade-offs (fp32 vs bf16, fused
   vs unfused) as paired entries.
3. `uv run zoo <name>` and `uv run zoo <name> --config IA-420f` — check it
   compiles and the metadata column reads well.
4. `uv run pytest -k "test_models and <name>"`.
5. Add an equivalence test if a reference implementation exists.
6. Update `README.md` (category table + section).
7. Bump `[project] version` in `pyproject.toml` (see "Bump the zoo version"),
   committing the `uv.lock` re-lock with it — the same applies when you _change_
   an existing model.
8. `uv run pre-commit run --all-files`.

## Where to look things up

**Anything about what the compiler supports or how fast the hardware is belongs
to the SDK docs, not to this file.** Go there first and quote nothing back.

Hosted (tracks latest): <https://vollo.myrtle.ai/latest/>. Offline, matching the
compiler you actually have installed: `$SDK/docs/mdbook/src/*.md` in any
extracted SDK (in a myrtlepkgs checkout,
`pkgs/vollo2/release-dir/docs/mdbook/src/`). Prefer the offline copy when
running against an unreleased build — the hosted site describes the latest
_release_, which may be behind your wheels. Pages are the same names with
`.html` hosted / `.md` offline.

| Question                                                         | Page                                                                                                                                                                                       |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Is this op supported? Can it run in fp32?                        | [supported-models](https://vollo.myrtle.ai/latest/supported-models.html) — check first when a model won't compile                                                                          |
| Why did my shape/matmul/reshape fail?                            | [data-dimension](https://vollo.myrtle.ai/latest/data-dimension.html) — per-op, incl. "Dynamic weights"                                                                                     |
| How does `streaming_transform` behave?                           | [example-2-cnn](https://vollo.myrtle.ai/latest/example-2-cnn.html), [example-3-lstm](https://vollo.myrtle.ai/latest/example-3-lstm.html)                                                   |
| How do I apply `Fp32Activations` / `Fp8Weights` / IO precisions? | [example-4-mixed-precision](https://vollo.myrtle.ai/latest/example-4-mixed-precision.html)                                                                                                 |
| What is the compile → VM → latency flow `vm.py` wraps?           | [example-1-mlp](https://vollo.myrtle.ai/latest/example-1-mlp.html)                                                                                                                         |
| Multi-model programs (`ProgramBuilder`)                          | [example-5-multi-model](https://vollo.myrtle.ai/latest/example-5-multi-model.html)                                                                                                         |
| What IO latency do I add to a zoo number?                        | [benchmark-io](https://vollo.myrtle.ai/latest/benchmark-io.html)                                                                                                                           |
| What does real hardware measure?                                 | [benchmark-mlp](https://vollo.myrtle.ai/latest/benchmark-mlp.html), [-cnn](https://vollo.myrtle.ai/latest/benchmark-cnn.html), [-lstm](https://vollo.myrtle.ai/latest/benchmark-lstm.html) |
| Compiler/runtime API signatures                                  | [api](https://vollo.myrtle.ai/latest/api.html)                                                                                                                                             |
| Did behaviour change in this release?                            | [release-notes](https://vollo.myrtle.ai/latest/release-notes.html), or `$SDK/CHANGELOG.md`                                                                                                 |
