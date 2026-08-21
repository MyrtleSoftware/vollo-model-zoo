#!/usr/bin/env bash

# This script downloads specific versions of the Vollo SDK from GitHub,
# extracts them, and runs the benchmarks for each version. Each run writes
# benchmarks/vollo_<sdk>+zoo.<zoo>.json, and `benchmark` refuses to overwrite an
# existing file, so re-measuring a combination we already have is a no-op error.
#
# An already-extracted build/vollo-sdk-<version>/ is reused as-is, which is also
# how to point this at a locally built (unreleased) SDK.
#
# Usage: ./scripts/backtest_github_versions.sh v26.1.1 v26.1.0 ...

set -e

VERSIONS=$@

if [ -z "$VERSIONS" ]; then
  echo "Usage: $0 <version1> <version2> ..."
  echo "Example: $0 v26.1.1 v26.1.0"
  exit 1
fi

TMP_DIR="build"
mkdir -p "$TMP_DIR"

# `uv pip install` below needs an environment to install into, and every command
# after it runs with --no-sync, so nothing else creates one.
uv sync

for VERSION in $VERSIONS; do
  # Remove 'v' prefix if present for the filename
  VER_NUM=${VERSION#v}
  SDK_FILE="vollo-sdk-${VER_NUM}.run"
  SDK_DIR="${TMP_DIR}/vollo-sdk-${VER_NUM}"

  echo "================================================================================"
  echo "Processing Vollo SDK $VERSION"
  echo "================================================================================"

  if [ ! -d "$SDK_DIR" ]; then
    if [ ! -f "${TMP_DIR}/${SDK_FILE}" ]; then
      URL="https://github.com/MyrtleSoftware/vollo-sdk/releases/download/${VERSION}/${SDK_FILE}"
      echo "Downloading $URL ..."
      curl -L -o "${TMP_DIR}/${SDK_FILE}" "${URL}"
    fi

    echo "Extracting $SDK_FILE ..."
    # The .run file is a self-extracting archive.
    chmod +x "${TMP_DIR}/${SDK_FILE}"

    # We try to extract it into its own directory.
    ./"${TMP_DIR}/${SDK_FILE}" --target "$SDK_DIR" --noexec --accept >/dev/null
  fi

  if [ -d "$SDK_DIR/python" ]; then
    echo "Installing Vollo $VER_NUM from $SDK_DIR/python ..."

    # The `==` pins are what make this work: --find-links (like UV_FIND_LINKS) is
    # *additive* to PyPI, so an unpinned install resolves to whatever is newest
    # on PyPI rather than the bundle we just extracted. `uv pip install` leaves
    # uv.lock alone, which is why everything below needs --no-sync -- a plain
    # `uv run` would re-sync and silently measure the locked compiler instead.
    uv pip install --find-links "$SDK_DIR/python" \
      --reinstall-package vollo-compiler --reinstall-package vollo-torch \
      "vollo-compiler==${VER_NUM}" "vollo-torch==${VER_NUM}"

    INSTALLED=$(uv run --no-sync python -c \
      "import importlib.metadata as m; print(m.version('vollo-compiler'))")

    if [ "$INSTALLED" != "$VER_NUM" ]; then
      echo "Error: asked for vollo-compiler $VER_NUM but $INSTALLED is installed"
      exit 1
    fi

    echo "Running benchmark for $VERSION ..."
    uv run --no-sync benchmark
  else
    echo "Error: Could not find python/ directory in $SDK_DIR"
    exit 1
  fi

  echo "Completed $VERSION"
  echo ""
done

# The venv is left holding the last SDK installed above; put the locked version
# back so later work isn't quietly done against an old compiler.
uv sync

echo "Backtesting completed"
