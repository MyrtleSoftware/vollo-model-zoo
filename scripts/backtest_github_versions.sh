#!/usr/bin/env bash

# This script downloads specific versions of the Vollo SDK from GitHub,
# extracts them, and runs the benchmarks for each version.
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
    echo "Running benchmark for $VERSION ..."
    export UV_FIND_LINKS="$(pwd)/$SDK_DIR/python/"

    # Re-install dependencies to use the new SDK version
    uv sync --reinstall-package vollo-compiler --reinstall-package vollo-torch

    # Run the benchmark
    uv run benchmark
  else
    echo "Error: Could not find python/ directory in $SDK_DIR"
    exit 1
  fi

  echo "Completed $VERSION"
  echo ""
done

echo "Backtesting completed"
