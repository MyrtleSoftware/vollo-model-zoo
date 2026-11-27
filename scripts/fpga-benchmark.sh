#!/usr/bin/env bash
# Runs the zoo on a Vollo card and writes a results JSON in the shape of
# benchmarks/*.json. The only input is a sourced SDK (`source $VOLLO_SDK/setup.sh`);
# the zoo config is read off the card. Tools: `uv`, `make`, `timeout`, `jq` and
# a C compiler (`cc`). The checkout may be read-only: the project is copied into the
# work directory and everything is built there.
#
# As the VM benchmarks, every model runs, experimental ones included, at every
# size but the experimental ones.
#
# Usage: scripts/fpga-benchmark.sh [--out results.json]
#          [--config <zoo config>]   (default: detected from the card)
#          [--device <bdf|index>] [--skip <model>|<model>-<size>]...
#          [--workdir DIR] [--num-inferences N] [--spacing-ns N]
# --workdir must be new, empty or one this script made; it defaults to
# ${XDG_CACHE_HOME:-~/.cache}/vollo-model-zoo/fpga-benchmark,
# outside the checkout; --out defaults to <workdir>/results.json. The card's hw
# config is written next to it, as <out without .json>.hw_config.json.
#
# Results are under the zoo config's name suffixed with -FPGA (V80-FPGA), so
# `plot` and `report` keep them apart from the VM's.
#
# Exit codes:
#   0  no compiled program failed on the card (sizes that don't compile or fit
#      are reported in the JSON but don't fail the run)
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
for tool in uv make timeout jq cc; do
  command -v "$tool" > /dev/null || die "$tool is not installed"
done

SRC=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$WORKDIR" || die "can't create $WORKDIR"
WORKDIR=$(cd "$WORKDIR" && pwd)
# The script deletes and rewrites zoo/, programs/ and runs/ in the work dir, so
# it only uses one that is empty or that it made.
MARKER="$WORKDIR/.fpga-benchmark-workdir"
if [[ ! -e "$MARKER" && -n "$(ls -A -- "$WORKDIR")" ]]; then
  die "$WORKDIR is not empty and wasn't made by this script; pass a new or empty --workdir"
fi
touch "$MARKER" || die "can't write to $WORKDIR"
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
run() { uv run --no-sync "$@"; }
uv sync --frozen --no-dev || die "uv sync failed (Python >= 3.13 is provided by uv)"
uv pip install --find-links "$VOLLO_SDK/python" \
  --reinstall-package vollo-compiler --reinstall-package vollo-torch \
  "vollo-compiler==$VER" "vollo-torch==$VER" \
  || die "the wheels of SDK $VER don't install; the zoo can't compile for it"
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

# fpga CMD...: a command of vollo_model_zoo/fpga.py.
fpga() { run python -m vollo_model_zoo.fpga "$@"; }

# 3. The card's hw config: `vollo-tool` lists the cards it read, one here.
"$VOLLO_SDK/bin/vollo-tool" read-hw-config "$DEVICE" | jq -e '.[0].hw_config' > "$HW_CONFIG" \
  || die "can't read the hw config of device $DEVICE"
[[ -n "$CONFIG" ]] || CONFIG=$(fpga config --hw-config "$HW_CONFIG") \
  || die "can't tell which zoo config device $DEVICE is; pass --config"
echo "Zoo config: $CONFIG"

# 4. Compile.
fpga compile --config "$CONFIG" --hw-config "$HW_CONFIG" --out "$PROGRAMS" ${SKIPS[@]+"${SKIPS[@]}"} \
  || die "compiling the zoo failed"

# 5. Run.
timed_out=false
failures=()

# bench NAME INDEX KIND [vollo-example args]: one run, JSON in runs/NAME.INDEX.KIND.json
# and, if it failed, its message in runs/NAME.INDEX.KIND.run-error. Its output must
# have the numbers `fpga collect` reads (fpga.py's _RUN_KEYS), each finite and > 0.
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
  elif ! jq -e 'type == "object"' "$json" > /dev/null 2>&1; then
    echo "vollo-example did not print valid JSON" > "$error"
  elif ! jq -e '
      [.metrics.latency_us.mean, .metrics.latency_us.p99]
      | all(type == "number" and . > 0 and (isinfinite | not) and (isnan | not))
    ' "$json" > /dev/null 2>&1; then
    echo "vollo-example output lacks a latency, or has one <= 0" > "$error"
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
    else
      bench "$name" "$index" contiguous || failures+=("$name.$index contiguous")
      # A contiguous run that timed out has its run-error; the spaced one isn't tried.
      if ! $timed_out && ! bench "$name" "$index" spaced --inference-spacing-ns "$SPACING_NS"; then
        failures+=("$name.$index spaced")
      fi
    fi
  done
done 3< "$PROGRAMS/sizes.txt"

# 6. Results, written even when a run failed.
fpga collect --config "$CONFIG" \
  --programs "$PROGRAMS" --runs "$RUNS" --out "$OUT" || die "collecting the results failed"
# Once per run, rather than in every entry of the results.
HW_OUT="${OUT%.json}.hw_config.json"
cp "$HW_CONFIG" "$HW_OUT" || die "can't write $HW_OUT"
echo "Results written to $OUT, the card's hw config to $HW_OUT"

# The JSON decides too, so a run `collect` rejected can never pass.
run_errors=$(jq '[.. | objects | select(has("RunError"))] | length' "$OUT") \
  || die "can't read $OUT"
if ((${#failures[@]} || run_errors)); then
  if ((${#failures[@]})); then
    echo "${#failures[@]} failures:"
    printf ' %s\n' "${failures[@]}"
  fi
  echo "$run_errors RunError entries in $OUT"
  exit 1
fi
if ! compgen -G "$PROGRAMS/*.vollo" > /dev/null; then
  echo "No size compiled for this hw config: nothing was tested"
  exit 2
fi
echo "All zoo runs passed"
