# AGENTS.md — Vollo Model Zoo (VMZ)

Guidance for AI agents (and humans) working in this repo.

## What this repo is

**Self-contained PyTorch example models** that compile with the
[Vollo SDK](https://vollo.myrtle.ai/latest/installation.html) for low-latency
FPGA inference. Each model is a reference implementation _plus_ size
configurations whose compute latency is measured on Vollo's VM. Compiler + VM
only — no FPGA, no `vollo-rt`, no `.vollo` artifacts.

Two audiences constrain every change:

- **customers** copy-paste a single model file or run `uv run zoo <model>`, so
  files must stand alone.
- **Myrtle devs** add and tune models, and re-run the zoo against unreleased
  SDKs to catch regressions and exploit new compiler features.

## Setup and commands

Requires [`uv`](https://docs.astral.sh/uv/) and Python 3.13 (`.python-version`).
`uv run` bootstraps the venv from `uv.lock` — never `pip install` into it by hand.

```fish
uv run zoo <model>                 # latency table for one model (default config: V80)
uv run zoo <model> --config V80LL  # pick a hardware config
uv run zoo <model> -j              # machine-readable JSON
uv run zoo --help                  # list all models + configs
uv run zoo moe --experimental      # experimental models need this flag

uv run pytest                      # full suite; slow — dominated by model x config
uv run pytest -k "test_models and mamba2"   # scope it while iterating
uv run pytest tests/test_gru.py

uv run pre-commit install          # once; hooks = isort, black, codespell
uv run pre-commit run --all-files  # lint
```

A model file also runs directly (`uv run python vollo_model_zoo/models/mlp.py`)
via its `__main__` block.

`ci.yml` runs `uv lock --check`, `pre-commit run --all-files`, then `pytest -x`.

- The `-x` means a red CI reports only the _first_ failure — never read
  "everything else passed" into it; re-run locally without `-x`.
- `uv lock --check` fails when `pyproject.toml` was committed without its
  `uv.lock` hunk (usually a zoo version bump, recorded in both). Every other uv
  command re-locks silently, so this is the only thing that catches it. Fix with
  `uv lock` and commit the result.

## Running against a different Vollo SDK

`vollo-compiler` / `vollo-torch` come from PyPI, pinned by `uv.lock` — the single
source of truth for the version the zoo is written and tested against, since
`pyproject.toml` declares no version bound. To override it, get a wheel
directory from a released `.run` bundle
([releases](https://github.com/MyrtleSoftware/vollo-sdk/releases)):

```fish
./vollo-sdk-<version>.run --target <dir> --noexec --accept
```

or build an unreleased SDK from a myrtlepkgs checkout (Myrtle devs):

```fish
nix-build ~/myrtlepkgs -A vollo2.release.vollo-sdk-release --out-link /tmp/vollo-sdk-release
```

Either tree holds:

- `python/` — the `vollo_compiler` / `vollo_torch` wheels this repo needs, plus
  an unused `vollo_rt`.
- `CHANGELOG.md` — first stop when a latency or compile behaviour changes.
- `docs/mdbook/src/*.md` — that build's docs.

Install from it:

```fish
set -x SDK /tmp/vollo-sdk-release/vollo-sdk-29.0.0   # adjust path and version

uv pip install --find-links $SDK/python \
  --reinstall-package vollo-compiler --reinstall-package vollo-torch \
  vollo-compiler==29.0.0 vollo-torch==29.0.0

uv run --no-sync zoo mlp
uv run --no-sync pytest -k "test_models and mlp"

uv run --no-sync python -c "import importlib.metadata as m; print(m.version('vollo-compiler'))"
```

Two things there are load-bearing:

- **`--no-sync`**: a plain `uv run` re-syncs from `uv.lock` and silently
  reinstalls the pinned version, so results would come from the old compiler with
  no warning. Hence the last line, to confirm what you measured.
- **The `==<version>` pin**: `UV_FIND_LINKS` / `--find-links` are _additive_ to
  PyPI, so an unpinned re-resolve still prefers the newest PyPI release over the
  wheels you pointed at.

### How CI does it instead

`update_benchmarks.yml` needs none of that, because released wheels are on PyPI:
it moves the lock (`uv lock --upgrade-package vollo-compiler --upgrade-package
vollo-torch`) and lets a normal `uv sync` follow.

- The **lock file is part of the PR**, so merging it is what gives `uv run zoo`
  users the new compiler — and `ci.yml` runs the suite against the new SDK on
  that PR, so a red run there is the regression signal.
- A PR appears for either of two independent reasons: a new SDK on PyPI (the
  lock moves) or a zoo version bump (the benchmark file name is new). The
  benchmark run is gated on the file name, so a lock-only PR carries no numbers.
- Don't add a floor to `pyproject.toml` to record which SDK features a model
  needs: only the locked version is ever tested, so a floor would be an unchecked
  support claim about older SDKs. Backwards compatibility would have to be a CI
  leg (`uv lock --resolution lowest-direct`, needing floors on _every_ direct
  dependency), not a number in a file.

## Bump the zoo version when you change what is measured

A latency is a function of the compiler _and_ the zoo code, so a run is
identified by both, as a PEP 440 version `<vollo>+zoo.<zoo>`
(`vollo_model_zoo/version.py`) — the JSON filename and its top-level key:

```
benchmarks/vollo_28.1.0+zoo.0.2.0.json   # {"28.1.0+zoo.0.2.0": {model: {config: [...]}}}
```

**The zoo version is `[project] version` in `pyproject.toml`, and nothing bumps
it for you** — `update_benchmarks.yml` skips when that filename exists, so an
unbumped version means your change is never measured. A minor bump; the number
carries no compatibility promise.

- **Bump** for any change under `models/` that moves a number (architecture, a
  sweep in `main()`, a constructor default), or to `vm.py`, which sets the compile
  options every model is measured with. Commit the `uv.lock` re-lock with it.
- **Don't** for README prose, tests or the reporting tooling — that spends a
  benchmark run reproducing numbers we have.
- **Don't** for the weekly SDK lock bump: the SDK version is already the other
  half of the identity.

Two consequences:

- Only the latest SDK is re-measured, so a new point lands beside the old ones
  rather than replacing them — hence plot labels naming both versions.
- `scripts/backtest_github_versions.sh` is meant to rebuild the curve over older
  releases, but it overrides the SDK with `UV_FIND_LINKS` + `uv sync`, which no
  longer works (see the pin above) — fix it to pin with `uv pip install` before
  trusting it.

## Layout

```
vollo_model_zoo/
  models/*.py     one file per model — the actual content of the zoo
  vm.py           compile + measure harness (vollo_info, CONFIGS, Result types, vollo_fn)
  zoo.py          `zoo` CLI: run one model, print markdown table or JSON
  benchmark.py \
  parse.py       | CI latency-history tooling feeding benchmarks/ and plots/;
  plot.py        | generated by the weekly workflow, not part of model
  report.py    /  development.
  version.py      what identifies a benchmark run (see above)
benchmarks/, plots/  generated latency history; do not hand-edit
scripts/          backtest_github_versions.sh (see caveat above)
.github/workflows ci.yml (lock check + lint + tests), update_benchmarks.yml (weekly PR)
```

## The model file contract

**We follow the Hugging Face single-file model style** (see `CONTRIBUTING.md`).
Each file must be self-contained, copy-pasteable, and must not import
`vollo_model_zoo` at module level. Hard rule — it is what makes the zoo useful
to customers. `mlp.py` is the minimal example, `mamba2.py` the involved one:

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
with `__`, and `vm.get_results()` `importlib`-imports it and calls
`main(config=...)`. Hence:

- The **filename is the CLI model name** (`mlp-res-rms.py` → `uv run zoo mlp-res-rms`).
- `main` must be a generator, taking `config` as a keyword argument.
- Hyphenated files are importable only via `importlib.import_module`, which tests
  must use for them.

`tests/test_models.py` enforces on every model:

- At least one yielded `Ok` result.
- A sweep monotonic in `param_count` **or** `latency_spaced` — order `main()`
  accordingly.
- At least one configuration in **0.95M–1.05M parameters**, the "baseline" size
  models are compared at. Show the arithmetic in a comment, as `lstm.py` does.
- No `AllocationError`/`SaveError` on the default, `V80` or `V80LL` configs. Not
  fitting the smaller boards (`IA-420f`, etc.) is tolerated and omitted.
- **A `ValueError` fails on _every_ config**, small boards included (the `case _`
  arm re-raises). `vollo_info` returns it as a value like the allocation errors,
  but unlike "too big" it is never tolerated.

## Vollo's execution model

> **The SDK docs are the authority; this is a map, not a copy.** Op support, the
> data-dimension algebra and latencies all change between releases, so this file
> says _which page answers a question_ and keeps only repo-specific detail. Never
> paste a support table or latency figure back in — a stale copy gets trusted.
> If this file contradicts the SDK docs, fix this file.

### Streaming

`vollo_info(..., time_axis=N)` runs `nnir.streaming_transform(N)`, turning a
model written over a whole sequence into a **stateful program consuming one
timestep per inference**. Pass the axis of `input` that is time; every zoo model
does this, so use them to place a new one:

- `1` for `[B, T, F]`: `slp`, `mlp`, `mlp-res-rms`, `lstm`, `ffn-swiglu`, `moe`
- `2` for conv models with `[B, C, T]`: `cnn`, `tcn`, `wavenet`, `mobilenet`
- `0` for `[T, D]` step-style models: `gru`, `ssm`, `mamba1`, `mamba2`
- `None` for genuinely non-streaming models: `resmlp`, which annotates why

Two `vollo_torch` building blocks make a model streaming-transformable
([example-2-cnn](https://vollo.myrtle.ai/latest/example-2-cnn.html),
[example-3-lstm](https://vollo.myrtle.ai/latest/example-3-lstm.html) are the
canonical write-ups):

- **`PaddedConv1d`** replaces `torch.nn.Conv1d`, which the streaming transform
  cannot handle. It wants time rightmost — hence the `transpose` sandwiches in
  `mamba1.py` / `mamba2.py`.
- **`Scan`** wraps a `step` module the compiler recognises as a repeated
  application (`gru`, `ssm`, `mamba1`, `mamba2`). Zoo convention: initial state is
  an `nn.Buffer(..., persistent=False)` so `.to()` works without polluting the
  state dict; multiple inputs/states go in as lists (`mamba2.py`).
  `torch.nn.LSTM` is supported natively and needs no `Scan`.

### The data dimension (the `!` in shape comments)

Vollo's compute units operate on contiguous vectors, so every activation tensor
has exactly one **data dimension**; weights have none. A compile that fails on a
shape usually failed here — read
[data-dimension](https://vollo.myrtle.ai/latest/data-dimension.html), organised
per-op, rather than guessing. Matmul contraction and reshape bite zoo models
most often and both have worked examples there.

Repo convention: `[a b! c]` marks `b` as the data dimension, and `mamba2.py`
tracks `!` on every intermediate. Keep that up in shape-juggling code — it is the
fastest way to see why a compile failed.

`allow_dynamic_weights=True` unlocks the matmul cases the default contraction
rule rejects, which is why the scan-based (`gru`, `ssm`, `mamba2`) and
large-linear (`resmlp`) models set it. Advanced: expect higher latency and more
tensor RAM, so reach for it when a matmul won't compile, not to make one faster.

### Precision

Default compute is **bf16**, with wider accumulation inside dot products. One
authoritative page each:

- **Which ops can run in fp32** — the "Fp32 Support" column of
  [supported-models](https://vollo.myrtle.ai/latest/supported-models.html). Check
  every time; the set grows per release.
- **How to apply it** —
  [example-4-mixed-precision](https://vollo.myrtle.ai/latest/example-4-mixed-precision.html).

What that means here:

- `vollo_torch.Fp32Activations()` keeps recurrent state accurate over long
  sequences (`gru.py`'s `fp32`, `mamba2.py`'s `ssm_fp32`). Zoo convention: expose
  it as a constructor flag and sweep both settings in `main()`, so the latency
  cost is visible in the table.
- `vollo_torch.Fp8Weights()` halves weight storage on the Versal boards.
- Inputs/outputs are bf16 unless `to_nnir` gets `inputs_precisions` /
  `outputs_precisions` — `vm.vollo_fn` is the one place here that does.
- `gru.py` shows an accuracy trick (`sigmoid(-z)` over `1 - sigmoid(z)`);
  `tests/test_approx.py` builds higher-precision primitives (Newton–Raphson
  reciprocal) and measures their bit accuracy. Both repo-grown.

### Cost model

- Configs are `<cores>×<block size>`: `c6b32` (V80, V80LL, IA-420f, NT400D11) or
  `c3b64` (IA-840f). Feature dims that are multiples of the block size map
  cleanly onto the compute units — hence sweeps using sizes like `32 * 6 * 4`.
- The VM is a **near cycle-accurate** instruction-level simulation modelling
  **no host↔accelerator IO**. The smallest zoo models compile to well under 1 µs
  of compute, so for those **IO dominates end-to-end latency**; get IO figures
  from [benchmark-io](https://vollo.myrtle.ai/latest/benchmark-io.html). Zoo
  numbers are compute-only estimates for comparing models and shapes, never
  end-to-end guarantees — `zoo.py`, `report.py` and `benchmarks/README.md` all
  repeat that caveat in the same wording; keep it consistent.
- Sanity-check a too-good VM estimate against real hardware:
  [benchmark-mlp](https://vollo.myrtle.ai/latest/benchmark-mlp.html) /
  [-cnn](https://vollo.myrtle.ai/latest/benchmark-cnn.html) /
  [-lstm](https://vollo.myrtle.ai/latest/benchmark-lstm.html).
- **Fusing is not always faster**: `ffn-swiglu.py` sweeps `fuse=True/False` to
  show a GPU-style fused gate/value projection _slowing down_ a Vollo program.
  Preserve that kind of pedagogical pairing.
- `quick_compile=True` skips optimization passes to cut compile time (`mamba1`,
  `mamba2`); drop it when chasing the last few percent.

## `vm.py` API

- `CONFIGS`: `{"V80", "V80LL", "IA-420f", "IA-840f", "NT400D11"}` → `vc.Config`,
  built with `hasattr` probes so the repo still works against older SDKs missing
  a config. Add new configs the same defensive way.
- `vollo_info(model, x, *, time_axis, config, meta=None, allow_dynamic_weights=False, quick_compile=False) -> Result`
  calls `model(x)` first (nicer errors), then `prepare_shape` → `to_nnir` →
  optional `streaming_transform` → `to_program` → `pack()`. Returns `Ok` or a
  caught `AllocationError | SaveError | ValueError`: compile failures are
  _values_, so a sweep can report "too big" per size.
- `Ok`: `config`, `param_count`, `cycle_count`, `latency_spaced`,
  `latency_contiguous` (both `Microseconds`), `meta`. `latency_spaced` = isolated
  inferences (`compute_duration_per_inference_us(spaced=True)`);
  `latency_contiguous` = back-to-back. `meta` keys prefixed `_` are hidden from
  the `zoo` table.

Deliberately not exposed (use the compiler API directly): `program.metrics()`,
`ProgramBuilder` for multi-model programs, `generate_state_reset=True`, and
`program.save()`/`vollo-onnx`.

## Debugging a model: observing the NNIR

Mostly for Myrtle devs. `vollo_info` returns a latency or an error, which is the
wrong granularity when a model won't compile or is mysteriously slow. **Drop
below the harness and print the NNIR** — the compiler's own view of your model.

Note that **`zoo` hides failures**: `run_model` skips every non-`Ok` result, so a
size that fails to compile silently vanishes from the table. Use `-j` to see it
(`{"ValueError": "<message>"}`), or go manual. This runs as-is with
`uv run python <file>`; swap in your model (hyphenated files need
`importlib.import_module`):

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

`print(nnir)` lists inputs, outputs, and one line per node — node id, the
originating **fx name in parentheses** (your link back to the Python source), the
NNIR op with its constant shapes and precision, then the output shape:

```
Id(1v2) - (linear) Linear(weight: (64, 64), weight_precision: Bf16, bias: 64, input: Id(0)) - [1, 5, 64]
Id(2v2) - (relu) Clamp(min: 0, max: None, input: Id(1), activation_precision: Bf16) - [1, 5, 64]
```

Use it to:

- **Verify a precision context applied.** Wrapping a region in
  `Fp32Activations()` flips its nodes' `activation_precision` to `F32` and leaves
  the rest `Bf16` — the only reliable check that a `with` block covers the ops you
  meant. A node inside the block that stays `Bf16` has no fp32 support, so read
  the support table rather than arguing with the compiler.
- **Verify the streaming transform.** The time axis should disappear from every
  shape (here `[1, 5, 64]` → `[1, 64]`); a shape keeping its time extent didn't
  stream.
- **See what your PyTorch lowered to.** Ops fuse and rename — `relu` becomes
  `Clamp`, `nn.Linear` becomes `Linear` with the bias inlined. When the data
  dimension complains about an op you didn't write, this is where you find it.

Two caveats:

- The `vN` in `Id(0v2)` is a graph version that changes as the graph is
  transformed (the same node is `Id(0v2)` before the streaming transform,
  `Id(0v9)` after), so diff on structure and shapes, never ids.
- `to_nnir` aggregates unsupported ops into one plain **`ValueError`** (not
  `UnsupportedOperationError`) whose message names them (`unsupported function
  'fft_fft': found 1`) — cross-reference against
  [supported-models](https://vollo.myrtle.ai/latest/supported-models.html). This
  is the error `vollo_info` returns as a value and `test_models.py` re-raises.

Two levels further out:

- **The fx graph**, if `to_nnir` fails before producing anything. After
  `prepare_shape`, `print(model.graph)` shows what it read, and every node carries
  `tensor_meta` (shape, dtype, stride) plus the `vollo_fp32_activations` /
  `vollo_fp8_weights` / `vollo_core_partition` keys the precision contexts set —
  so you can confirm a context reached a node before NNIR exists. Use `.get()`:
  the `output` node has none of the `vollo_*` keys.

  ```python
  for n in model.graph.nodes:
      print(n.op, n.target, n.meta["tensor_meta"].shape, n.meta.get("vollo_fp32_activations"))
  ```

  `vt.fx.save(...)` / `vt.fx.load(...)` archive a traced module — the tidiest way
  to hand someone a repro.

- **`program.metrics()`** for an `AllocationError`. Its `repr` is useless; read
  the named fields, most of which are **per-core lists** (six on `c6b32`, three on
  `c3b64`): `tensor_ram_used` / `tensor_ram_depth`, `weight_store_used` /
  `weight_store_depth`, `num_instrs`, `num_micro_instructions`, `input_bytes`,
  `output_bytes`. `_used` against `_depth` says which store you blew and by how
  much; all the load on core 0 with the rest at zero says the model isn't spread
  across cores.

## Tests

- `test_models.py` — the bulk test: every model × (default + all 5 configs),
  asserting the invariants above. Slow; scope with `-k`.
- `test_gru.py` — numerical equivalence against `torch.nn.GRU`, including the
  **state-dict conversion** from PyTorch layout to the zoo's split-linear layout.
- `test_mamba1.py` / `test_mamba2.py` — equivalence against
  [FLA](https://github.com/fla-org/flash-linear-attention) reference layers, again
  with `convert_state_dict`. Skipped without CUDA (FLA needs triton). The README
  points customers at these as the canonical "how do I load my trained weights
  into the zoo model" example — keep them working.

Add an equivalence test in the same style whenever a new model has an established
reference implementation.

## Conventions and gotchas

- `@beartype` on public constructors and module-level functions; type hints
  everywhere. `beartype.typing` and `collections.abc` are both used for
  `Generator`/`Optional` — match the file you're in.
- Formatting is `black` + `isort`; `codespell` runs over all text with `vollo`
  allow-listed.
- Docstrings state tensor shapes on `forward`. Do this; it is how readers
  navigate the streaming and data-dimension rules.
- Experimental gating is **hardcoded in `zoo.py`**
  (`if args.model.lower() == "moe"`). The notice prints unconditionally; only the
  early `return 1` is conditional on `--experimental`. Marking another model
  experimental means updating that check _and_ the README banner; there is no
  per-model metadata for it yet.

## Checklist: adding a model

1. `vollo_model_zoo/models/<name>.py` following the contract above; no
   `vollo_model_zoo` imports outside `_vm`.
2. A sweep for `main()`: monotonic in params or latency, including one
   ~1M-parameter size; expose Vollo trade-offs (fp32 vs bf16, fused vs unfused)
   as paired entries.
3. `uv run zoo <name>` and `uv run zoo <name> --config IA-420f` — check it
   compiles and the metadata column reads well.
4. `uv run pytest -k "test_models and <name>"`.
5. An equivalence test, if a reference implementation exists.
6. `README.md`: category-table row _and_ a prose section explaining the
   architecture and which Vollo feature the file demonstrates.
7. Bump `[project] version` in `pyproject.toml`, committing the `uv.lock` re-lock
   with it — same when you _change_ an existing model.
8. `uv run pre-commit run --all-files`.

## Where to look things up

**What the compiler supports and how fast the hardware is belongs to the SDK
docs, not here.** Go there first and quote nothing back. Two copies:

- **Hosted**, tracking the latest _release_: <https://vollo.myrtle.ai/latest/>.
- **Offline**, matching the compiler you have installed:
  `$SDK/docs/mdbook/src/*.md` in any extracted SDK (in a myrtlepkgs checkout,
  `pkgs/vollo2/release-dir/docs/mdbook/src/`). Prefer this against an unreleased
  build.

The table names pages, not URLs: `api` is
<https://vollo.myrtle.ai/latest/api.html> hosted, `api.md` offline.

| Question                                          | Page                                                      |
| ------------------------------------------------- | --------------------------------------------------------- |
| Is this op supported? Can it run in fp32?         | supported-models — check first when a model won't compile |
| Why did my shape/matmul/reshape fail?             | data-dimension — per-op, incl. "Dynamic weights"          |
| How does `streaming_transform` behave?            | example-2-cnn, example-3-lstm                             |
| `Fp32Activations` / `Fp8Weights` / IO precisions? | example-4-mixed-precision                                 |
| The compile → VM → latency flow `vm.py` wraps?    | example-1-mlp                                             |
| Multi-model programs (`ProgramBuilder`)           | example-5-multi-model                                     |
| What IO latency do I add to a zoo number?         | benchmark-io                                              |
| What does real hardware measure?                  | benchmark-mlp, benchmark-cnn, benchmark-lstm              |
| Compiler/runtime API signatures                   | api                                                       |
| Did behaviour change in this release?             | release-notes, or `$SDK/CHANGELOG.md`                     |
