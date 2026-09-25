"""Normalize sectioned configs to the flat runtime/checkpoint contract."""
from copy import deepcopy


def flatten_config(config):
    result = deepcopy(config)
    version = result.pop("config_version", 1)
    if type(version) is not int or version not in {1, 2}:
        raise ValueError(f"Unsupported config_version: {version}")
    for section in ("model", "training", "objective"):
        values = result.pop(section, {})
        if not isinstance(values, dict):
            raise ValueError(f"{section} must be an object")
        for key, value in values.items():
            if key in result and result[key] != value:
                raise ValueError(f"Conflicting configuration field: {key}")
            result[key] = value
    if isinstance(result.get("data"), dict):
        values = result.pop("data")
        if "path" not in values:
            raise ValueError("data.path is required")
        result["data"] = values["path"]
        for key, value in values.items():
            if key == "path":
                continue
            if key in result and result[key] != value:
                raise ValueError(f"Conflicting configuration field: {key}")
            result[key] = value
    return result
