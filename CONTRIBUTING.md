# Contributing

## Repo structure

We follow the Hugging Face model, each example model should be self-contained
in a single file. This enables the easiest copy-paste reproducibility for users
and (more importantly) makes it easier to understand a model without jumping
between files in an editor. Hence, each model file should not directly import
any `vollo_model_zoo` components.

## Pre-commit Hooks

We use `pre-commit` to ensure code quality and consistency. The hooks include:

- `isort`: Organizes imports.
- `black`: Formats Python code.
- `codespell`: Catches typos in all text files.

To install the pre-commit hooks, run:

```fish
uv run pre-commit install
```

The hooks will then run automatically on every commit. You can also run them
manually on all files:

```fish
uv run pre-commit run --all-files
```

## Testing

You can run all the models in the zoo with:

```fish
uv run pytest
```

## The Vollo SDK version

`uv.lock` pins the `vollo-compiler` / `vollo-torch` the zoo is developed and
tested against; `pyproject.toml` declares them without a version bound on
purpose, because no older SDK is tested. A weekly workflow moves the lock to
each new release and re-measures the benchmarks, so writing a model against the
current compiler is the expected thing to do — don't add a floor to
`pyproject.toml` to record which features a model needs.

If you need to compile against a different SDK (an unreleased build, or an
older release), point uv at a directory of wheels rather than editing the
dependency; see the recipes in [AGENTS.md](./AGENTS.md).

## Versioning

The [benchmarks](./benchmarks/README.md) are re-measured by a weekly workflow,
which identifies each run by the Vollo SDK version _and_ the version of this
repo, then skips the work if it already has that combination on file:

```
benchmarks/vollo_28.1.0+zoo.0.2.0.json
                 ^ SDK       ^ zoo, from `[project] version` in pyproject.toml
```

So if your change moves a latency — a model file, its size sweep, or the `vm.py`
harness they are all compiled with — **bump `version` in `pyproject.toml` in the
same PR**, and commit the `uv.lock` re-lock that comes with it. Without the bump
the workflow mistakes your change for a run it has already measured, and the new
numbers never appear.

Changes that cannot move a latency (documentation, tests, the plotting and
reporting tooling) don't need a bump.
