import json
from pathlib import Path

from beartype import beartype


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
                    "latency_spaced": ok["latency_spaced"]["microseconds"],
                    "latency_contiguous": ok["latency_contiguous"]["microseconds"],
                    "param_count": ok["param_count"],
                    "meta": meta_str,
                }
            )
    except Exception as e:
        print(f"Warning: Failed to process '{json_file}': {e}")

    return records


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
