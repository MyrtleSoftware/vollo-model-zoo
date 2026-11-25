import json
from pathlib import Path

from beartype import beartype

# Appended to the zoo config's name in card results (`fpga.py`), so `plot` and
# `report` keep them apart from the VM's (`V80` vs `V80-FPGA`).
FPGA_SUFFIX = "-FPGA"


@beartype
def parse_records_from_json(json_file: str | Path) -> list[dict]:
    """
    Extract benchmark records from a single JSON file.
    """
    records = []
    try:
        with open(json_file, "r") as f:
            data = json.load(f)

        for version, model, config, res in _iter_results(data):
            if "Ok" not in res:
                continue

            ok = res["Ok"]
            meta = ok.get("meta", {})
            meta_str = ", ".join(f"{k}={v}" for k, v in meta.items())
            records.append(
                {
                    "version": version,
                    "model": model,
                    "config": config,
                    "latency_spaced": _microseconds(ok, "latency_spaced"),
                    "latency_contiguous": _microseconds(ok, "latency_contiguous"),
                    "param_count": ok["param_count"],
                    "meta": meta_str,
                }
            )
    except Exception as e:  # noqa: BLE001 -- skip a bad file, keep the rest
        print(f"Warning: Failed to process '{json_file}': {e}")

    return records


@beartype
def _microseconds(ok: dict, key: str) -> float | None:
    """
    A latency of an `Ok` entry, or None where it is null: a card result
    (`fpga.py`) whose run of that kind failed or was skipped.
    """
    latency = ok.get(key)
    return None if latency is None else latency["microseconds"]


@beartype
def _iter_results(data: dict):
    """
    Generator to flatten the nested benchmark JSON structure.
    """
    for version, models in data.items():
        for model, configs in models.items():
            for config, results in configs.items():
                for res in results:
                    yield version, model, config, res
