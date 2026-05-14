"""
config_reader.py — Reads config/config.yaml and returns a simple namespace.
Every module uses this to avoid hardcoding paths or params.
"""

import yaml
from pathlib import Path
from types import SimpleNamespace


def load_config(config_path: str = "config/config.yaml") -> SimpleNamespace:
    """Load YAML config and return as a nested SimpleNamespace for dot-access."""
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)
    return _dict_to_ns(raw)


def _dict_to_ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_ns(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_dict_to_ns(i) for i in d]
    return d


# Usage example:
# cfg = load_config()
# print(cfg.data.raw_data_path)       → "data/raw/Loan_Data_Clean_cld.xlsx"
# print(cfg.model.random_state)       → 42
# print(cfg.aws.region)               → "ap-south-1"
