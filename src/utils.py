"""Small helpers shared by all scripts: config loading, seeding, hardware info."""
import copy
import os
import platform
import random

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _deep_update(base, new):
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path):
    """Load a YAML config. Supports `_base_: relative/path.yaml` inheritance."""
    path = os.path.abspath(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("_base_", None)
    if base:
        base_cfg = load_config(os.path.join(os.path.dirname(path), base))
        cfg = _deep_update(base_cfg, cfg)
    return cfg


def apply_overrides(cfg, overrides):
    """overrides: dict with dotted keys, e.g. {'train.lr': 1e-3}."""
    cfg = copy.deepcopy(cfg)
    for key, val in (overrides or {}).items():
        d = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = val
    return cfg


def set_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def hardware_string():
    import torch
    cpu = platform.processor() or platform.machine()
    if torch.cuda.is_available():
        return f"GPU: {torch.cuda.get_device_name(0)} | CPU: {cpu}"
    return f"CPU only: {cpu}"


def exp_name(cfg, seed):
    return f"{cfg['dataset']}__{cfg['name']}__seed{seed}"
