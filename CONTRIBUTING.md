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
