"""Project configuration, seeding and environment capture.

Notebook 00 writes config.json; every other notebook reads it with load_config()."""
from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from copy import deepcopy

import numpy as np

DEFAULT_CONFIG = {
    # ---- where things live (EDIT in notebook 00) --------------------------------------------
    "datasets": {
        # name: {"path": folder with the CSVs, "loader": which parser, options}
        "InSDN": {"path": "", "loader": "insdn", "train_cap": None, "max_rows_per_class": None},
        # use the CORRECTED release (Engelen et al., 2021) — the original has labelling/timestamp faults
        "CICIDS2017": {"path": "", "loader": "cic", "train_cap": 400_000, "max_rows_per_class": 600_000,
                       "attempted": "benign"},
        # optional third dataset (CSE-CIC-IDS2018, ideally the corrected release); very large -> sampled
        "CSE-CIC-IDS2018": {"path": "", "loader": "cic", "train_cap": 400_000, "max_rows_per_class": 400_000,
                            "attempted": "benign"},
    },
    "out_dir": "results",
    "primary_dataset": "InSDN",
    # ---- run mode -------------------------------------------------------------------------
    # "full"    = design specification (10 seeds, 40 trials)   -> numbers for the paper
    # "reduced" = publishable CPU-laptop setting (10 seeds, 20 trials, L <= 16)
    # "quick" = smoke run on a laptop (3 seeds, 8 trials, capped train size) -> check everything runs
    "mode": "reduced",          # the setting used for every number in the paper
    "full": {"seeds": list(range(10)), "sweep_seeds": [0, 1, 2, 3, 4], "trials": 40,
             "L_grid": [1, 4, 8, 16, 32], "max_epochs": 50, "patience": 5, "val_cap": None},
    # "reduced" = publishable on a CPU laptop: 10 seeds kept (statistics), all InSDN data kept;
    # 20 tuning trials, L up to 16, random-split arms of B1 with 5 seeds, CICIDS2017 training capped at 150k
    "reduced": {"seeds": list(range(10)), "sweep_seeds": [0, 1, 2, 3, 4], "sr_seeds": [0, 1, 2, 3, 4],
                "trials": 20, "L_grid": [1, 4, 8, 16], "max_epochs": 40, "patience": 4, "val_cap": None,
                "train_caps": {"CICIDS2017": 150_000, "CSE-CIC-IDS2018": 150_000}},
    "quick": {"seeds": [0, 1, 2], "sweep_seeds": [0, 1], "trials": 8,
              "L_grid": [1, 4, 8, 16], "max_epochs": 20, "patience": 3, "val_cap": 30_000,
              "train_cap_override": 60_000},
    # ---- protocol -------------------------------------------------------------------------
    "tune_L": 8,                 # window length used for equal-budget tuning
    "best_L": None,              # set after notebook 03 (B3); None -> chosen automatically from B3
    "window_key": "Dst IP",      # context key for causal windows (sensitivity: "pair")
    "split_frac": [0.70, 0.15, 0.15],
    "tasks": ["multi", "binary"],
    "models": ["RF", "XGBoost", "MLP", "CNN1D", "LSTM", "GRU", "BiLSTM", "CNN-LSTM", "Transformer"],
    "b4_models": None,           # None -> top-4 by B2 macro-F1 on the primary dataset
    "b4_strategies": ["none", "weighted_ce", "focal", "logit_adjusted", "balanced_sampling", "smote"],
    # which experiments run on which datasets (empty path -> dataset skipped automatically)
    "experiments": {"B1": ["InSDN", "CICIDS2017", "CSE-CIC-IDS2018"], "B3": ["InSDN"],
                    "B4": ["InSDN"], "B5": ["InSDN"]},
    "tost_margin": 0.005,        # +/-0.5 macro-F1 points
    "bootstrap_B": 1000,
    "batch_size": 512,
    "device": "auto",            # "auto" | "cpu" | "cuda" | "mps"
    "n_jobs": -1,                # CPU workers for RF/XGBoost
    "save_preds": True,
}


def deep_update(base: dict, upd: dict) -> dict:
    out = deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def config_path(root: str = ".") -> str:
    return os.path.join(root, "config.json")


def save_config(cfg: dict, root: str = ".") -> str:
    p = config_path(root)
    with open(p, "w") as f:
        json.dump(cfg, f, indent=2)
    return p


def load_config(root: str = "..") -> dict:
    """Notebooks live in notebooks/, config.json in the project root (root='..')."""
    p = config_path(root)
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} not found - run notebook 00 first")
    cfg = deep_update(DEFAULT_CONFIG, json.load(open(p)))
    if not os.path.isabs(cfg["out_dir"]):
        cfg["out_dir"] = os.path.abspath(os.path.join(root, cfg["out_dir"]))
    return cfg


def mode_params(cfg: dict) -> dict:
    return cfg[cfg["mode"]]


def active_datasets(cfg: dict, experiment: str | None = None) -> list[str]:
    names = cfg["experiments"].get(experiment, list(cfg["datasets"])) if experiment else list(cfg["datasets"])
    return [n for n in names if n in cfg["datasets"] and cfg["datasets"][n].get("path")
            and os.path.isdir(cfg["datasets"][n]["path"])]


def train_cap(cfg: dict, dataset: str):
    mp = mode_params(cfg)
    cap = cfg["datasets"][dataset].get("train_cap")
    if dataset in mp.get("train_caps", {}):
        cap = mp["train_caps"][dataset]
    over = mp.get("train_cap_override")
    if over:
        cap = over if cap is None else min(cap, over)
    return cap


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def enable_determinism():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", "0")
    import torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_device(cfg: dict):
    import torch
    d = cfg.get("device", "auto")
    if d == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(d)


def capture_environment(out_dir: str) -> dict:
    """Hardware + library versions for the paper's 'Experimental environment' section."""
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "processor": platform.processor()}
    info["cpu_model"] = platform.processor()
    try:
        if sys.platform.startswith("win"):          # Windows: exact CPU name from the registry
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            info["cpu_model"] = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        elif sys.platform == "darwin":
            info["cpu_model"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                               capture_output=True, text=True).stdout.strip() or info["cpu_model"]
        else:
            cpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
            info["cpu_model"] = next((l.split(":", 1)[1].strip() for l in cpu.splitlines() if "Model name" in l),
                                     info["cpu_model"])
    except Exception:
        pass
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
        info["cpu_cores_logical"] = psutil.cpu_count()
    except ImportError:
        info["cpu_cores_logical"] = os.cpu_count()
    for lib in ("numpy", "pandas", "scipy", "sklearn", "torch", "xgboost", "optuna", "imblearn", "matplotlib"):
        try:
            m = __import__(lib)
            info[lib] = m.__version__
        except Exception:
            info[lib] = None
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
    except Exception:
        pass
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "environment.json"), "w") as f:
        json.dump(info, f, indent=2)
    return info
