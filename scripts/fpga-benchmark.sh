#!/usr/bin/env bash
# Runs the zoo on a Vollo card and writes a results JSON in the shape of
# benchmarks/*.json. The only input is a sourced SDK (`source $VOLLO_SDK/setup.sh`);
# the zoo config is read off the card. Tools: `uv` (or Python >= 3.13 with venv),
# `make`, `timeout` and a C compiler (`cc`). The checkout may be read-only: the
# project is copied into the work directory and everything is built there.
#
# Usage: scripts/fpga-benchmark.sh [--out results.json]
#          [--config <zoo config>]   (default: detected from the card)
#          [--device <bdf|index>] [--skip <model>|<model>-<size>]...
#          [--workdir DIR] [--num-inferences N] [--spacing-ns N]
# --workdir defaults to ${XDG_CACHE_HOME:-~/.cache}/vollo-model-zoo/fpga-benchmark,
# outside the checkout; --out defaults to <workdir>/results.json.
#
# Exit codes:
#   0  no compiled program failed on the card (sizes that don't compile or fit,
#      and runs skipped for an old vollo-example, are reported in the JSON but
#      don't fail the run)
#   1  a compiled program failed on the card (error, invalid JSON or timeout)
#   2  nothing compiled, so nothing was tested
#   64 setup error
# The JSON is written even when the exit code is 1 or 2.
set -uo pipefail

die() {
  echo "error: $*" >&2
  exit 64
}

CONFIG=""
OUT=""
DEVICE="${VOLLO_CARD_BDF:-0}"
WORKDIR="${XDG_CACHE_HOME:-$HOME/.cache}/vollo-model-zoo/fpga-benchmark"
NUM_INFERENCES=100000
SPACING_NS=20000
BENCH_TIMEOUT_S=600
SKIPS=()

while (($#)); do
  case $1 in
    --config) CONFIG=${2:?}; shift 2 ;;
    --out) OUT=${2:?}; shift 2 ;;
    --device) DEVICE=${2:?}; shift 2 ;;
    --skip) SKIPS+=(--skip "${2:?}"); shift 2 ;;
    --workdir) WORKDIR=${2:?}; shift 2 ;;
    --num-inferences) NUM_INFERENCES=${2:?}; shift 2 ;;
    --spacing-ns) SPACING_NS=${2:?}; shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "${VOLLO_SDK:-}" ]] || die "VOLLO_SDK is not set; run 'source setup.sh' from the SDK"
for tool in make timeout cc; do
  command -v "$tool" > /dev/null || die "$tool is not installed"
done

SRC=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$WORKDIR" || die "can't create $WORKDIR"
WORKDIR=$(cd "$WORKDIR" && pwd)
[[ -n "$OUT" ]] || OUT="$WORKDIR/results.json"
mkdir -p "$(dirname -- "$OUT")"
OUT=$(cd "$(dirname -- "$OUT")" && pwd)/$(basename -- "$OUT")
PROGRAMS="$WORKDIR/programs"
RUNS="$WORKDIR/runs"
HW_CONFIG="$WORKDIR/hw_config.json"
rm -rf "$PROGRAMS" "$RUNS"
mkdir -p "$PROGRAMS" "$RUNS"

# 1. Environment: a copy of the project, with the SDK's compiler installed over the locked one.
wheels=("$VOLLO_SDK"/python/vollo_compiler-*.whl)
[[ -e "${wheels[0]}" ]] || die "no vollo_compiler wheel in $VOLLO_SDK/python"
((${#wheels[@]} == 1)) || die "several vollo_compiler wheels in $VOLLO_SDK/python: ${wheels[*]}"
VER=$(basename -- "${wheels[0]}" | cut -d- -f2)
torch_wheels=("$VOLLO_SDK"/python/vollo_torch-"$VER"-*.whl)
[[ -e "${torch_wheels[0]}" ]] || die "no vollo_torch $VER wheel in $VOLLO_SDK/python"

rm -rf "$WORKDIR/zoo"
mkdir -p "$WORKDIR/zoo"
cp -r --no-preserve=mode "$SRC/pyproject.toml" "$SRC/uv.lock" "$SRC/.python-version" "$SRC/README.md" \
  "$SRC/vollo_model_zoo" "$WORKDIR/zoo/" \
  || die "can't copy the project"
cd "$WORKDIR/zoo" || die "can't enter $WORKDIR/zoo"

# Never touch an activated venv (e.g. direnv's `layout uv` in the checkout):
# everything goes into the work dir's own environment.
unset VIRTUAL_ENV UV_PROJECT_ENVIRONMENT

# run CMD...: a command of the project environment (never re-syncs it).
if command -v uv > /dev/null; then
  run() { uv run --no-sync "$@"; }
  uv sync --frozen --no-dev || die "uv sync failed (Python >= 3.13 is provided by uv)"
  uv pip install --find-links "$VOLLO_SDK/python" \
    --reinstall-package vollo-compiler --reinstall-package vollo-torch \
    "vollo-compiler==$VER" "vollo-torch==$VER" \
    || die "the wheels of SDK $VER don't install; fpga-compile can't run for it"
else
  # No uv: a plain venv, which needs a Python >= 3.13 of its own.
  python3 -c 'import sys; sys.exit(sys.version_info < (3, 13))' \
    || die "neither uv nor Python >= 3.13 is available"
  python3 -m venv "$WORKDIR/venv" || die "can't create a venv"
  run() { PATH="$WORKDIR/venv/bin:$PATH" "$@"; }
  # Every other dependency at its uv.lock version, as `uv sync` would install it,
  # rather than whatever PyPI resolves to today.
  python3 - "$WORKDIR/zoo/uv.lock" > "$WORKDIR/constraints.txt" << 'EOF' \
    || die "can't read the locked versions from uv.lock"
import sys, tomllib

with open(sys.argv[1], "rb") as f:
    lock = tomllib.load(f)
versions = {}
for package in lock["package"]:
    if "registry" in package.get("source", {}):
        versions.setdefault(package["name"], set()).add(package["version"])
for name, found in sorted(versions.items()):
    # The SDK's wheels replace the locked compiler; a name locked at several
    # versions (one per platform) is left to pip.
    if name not in ("vollo-compiler", "vollo-torch") and len(found) == 1:
        print(f"{name}=={found.pop()}")
EOF
  "$WORKDIR/venv/bin/pip" install --find-links "$VOLLO_SDK/python" \
    --constraint "$WORKDIR/constraints.txt" \
    "vollo-compiler==$VER" "vollo-torch==$VER" . \
    || die "pip install failed for SDK $VER"
fi
for package in vollo-compiler vollo-torch; do
  INSTALLED=$(run python -c \
    "import importlib.metadata as m; print(m.version('$package'))") || die "can't import $package"
  [[ "$INSTALLED" == "$VER" ]] || die "asked for $package $VER but $INSTALLED is installed"
done

# 2. vollo-example, as the SDK's benchmark.sh builds it.
mkdir -p "$WORKDIR/example"
cp "$VOLLO_SDK"/example/{*.h,*.c,Makefile} "$WORKDIR/example/" || die "can't copy the SDK's example"
(cd "$WORKDIR/example" && make vollo-example) || die "building vollo-example failed"
EXAMPLE="$WORKDIR/example/vollo-example"

# Options an older vollo-example may lack; a run needing one is skipped.
HELP=$("$EXAMPLE" --help 2>&1 || true)
missing=()
for option in --raw-buffer-api --model-index --json --inference-spacing-ns; do
  [[ "$HELP" == *"$option"* ]] || missing+=("$option")
done

# 3. The card's hw config.
"$VOLLO_SDK/bin/vollo-tool" read-hw-config "$DEVICE" | run fpga-hw-config --out "$HW_CONFIG" \
  || die "can't read the hw config of device $DEVICE"
[[ -n "$CONFIG" ]] || CONFIG=$(run fpga-config --hw-config "$HW_CONFIG") \
  || die "can't tell which zoo config device $DEVICE is; pass --config"
echo "Zoo config: $CONFIG"

# 4. Compile.
run fpga-compile --config "$CONFIG" --hw-config "$HW_CONFIG" --out "$PROGRAMS" ${SKIPS[@]+"${SKIPS[@]}"} \
  || die "fpga-compile failed"

# 5. Run.
timed_out=false
failures=()
ran=0

# bench NAME INDEX KIND [vollo-example args]: one run, JSON in runs/NAME.INDEX.KIND.json
# and, if it failed, its message in runs/NAME.INDEX.KIND.run-error.
bench() {
  local name=$1 index=$2 kind=$3 json="$RUNS/$1.$2.$3.json" status
  local error="$RUNS/$1.$2.$3.run-error"
  shift 3
  echo "Running inference ($kind) for $name model $index"
  timeout --kill-after=30 "$BENCH_TIMEOUT_S" "$EXAMPLE" --device "$DEVICE" --model-index "$index" \
    --num-inferences="$NUM_INFERENCES" --raw-buffer-api "$@" --json "$PROGRAMS/$name.vollo" > "$json"
  status=$?
  cat "$json"
  # 124 is timeout's TERM, 137 its KILL
  if [[ $status == 124 || $status == 137 ]]; then
    echo "$name: timed out after ${BENCH_TIMEOUT_S}s, skipping the remaining runs"
    timed_out=true
    echo "timed out after ${BENCH_TIMEOUT_S}s" > "$error"
  elif [[ $status != 0 ]]; then
    echo "vollo-example exited with $status" > "$error"
  elif ! run fpga-check-json "$json"; then
    echo "vollo-example did not print valid JSON" > "$error"
  fi
  [[ ! -e "$error" ]]
}

while read -r model size <&3; do
  name="$model-$size"
  [[ -e "$PROGRAMS/$name.vollo" ]] || continue
  num_models=$(cat "$PROGRAMS/$name.num-models" 2> /dev/null || echo 1)
  for ((index = 0; index < num_models; index++)); do
    marker="$RUNS/$name.$index"
    if $timed_out; then
      echo "skipped after an earlier run timed out" > "$marker.skipped"
    elif ((${#missing[@]})); then
      echo "this SDK's vollo-example lacks ${missing[*]}" > "$marker.skipped"
    else
      ran=$((ran + 1))
      bench "$name" "$index" raw || failures+=("$name.$index raw")
      # A raw run that timed out has its run-error; the spaced one isn't tried.
      if ! $timed_out && ! bench "$name" "$index" spaced --inference-spacing-ns "$SPACING_NS"; then
        failures+=("$name.$index spaced")
      fi
    fi
  done
done 3< "$PROGRAMS/sizes.txt"

# 6. Results, written even when a run failed.
run fpga-collect --config "$CONFIG" --hw-config "$HW_CONFIG" \
  --programs "$PROGRAMS" --runs "$RUNS" --out "$OUT" || die "fpga-collect failed"
echo "Results written to $OUT"

if ((${#failures[@]})); then
  echo "${#failures[@]} failures:"
  printf ' %s\n' "${failures[@]}"
  exit 1
fi
if ! compgen -G "$PROGRAMS/*.vollo" > /dev/null; then
  echo "No size compiled for this hw config: nothing was tested"
  exit 2
fi
if ((ran == 0)); then
  # Not a failure: a timeout has already failed above, so these are runs an
  # old vollo-example can't do, each reported as Skipped in the JSON.
  echo "Every run was skipped: nothing was measured; see the Skipped entries"
  exit 0
fi
echo "All zoo runs passed"
