"""Experiment engine: data preparation (cached), equal-budget tuning, resumable experiment grids.

Results layout (per dataset):
  <out>/<dataset>/cache/            cleaned data (raw and deduplicated)
  <out>/<dataset>/load_info.json    per-class counts, timestamp parsing, features dropped
  <out>/<dataset>/clean_*.json      duplicates / conflicts removed
  <out>/<dataset>/best_hp.json      tuned hyper-parameters per model and task
  <out>/<dataset>/tuning_log.jsonl  every Optuna trial
  <out>/<dataset>/results.jsonl     one line per run (resumable)
  <out>/<dataset>/preds/<key>.npz   test probabilities + labels (bootstrap CIs, confusion matrices)
  <out>/<dataset>/histories/<key>.json  training curves (neural models)
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import warnings

import numpy as np
import pandas as pd

from . import datasets as D
from . import training as T
from .config import get_device, mode_params, train_cap, set_seed
from .models import TABULAR

warnings.filterwarnings("ignore", category=UserWarning)
PROTOCOLS = ["SR_raw", "SR_dedup", "SC"]
# sensitivity protocols (S1): per-class chronological split on raw (non-deduplicated) data, and on
# data deduplicated by feature vector only (label-free; earliest occurrence kept)
SENSITIVITY_PROTOCOLS = ["SC_raw", "SC_feat"]
PROTOCOL_LABEL = {"SR_raw": "Random split, raw", "SR_dedup": "Random split, deduplicated",
                  "SC": "Per-class chronological (S-C)", "SC_raw": "Per-class chronological, raw",
                  "SC_feat": "Per-class chronological, label-free dedup."}
DEDUP_MODE = {"SR_raw": False, "SC_raw": False, "SR_dedup": True, "SC": True, "SC_feat": "features"}


# ================================================================ paths and cache
def ds_dir(cfg, ds):
    p = os.path.join(cfg["out_dir"], ds)
    os.makedirs(p, exist_ok=True)
    return p


_MEM = {}


def release_except(ds_keep=None):
    """Free cached DataFrames of other datasets (keeps RAM bounded when looping over datasets)."""
    for k in [k for k in _MEM if k[0] != ds_keep]:
        del _MEM[k]


def load_clean(cfg, ds, dedup, verbose=True):
    """Cleaned dataset (cached on disk). dedup: False | True | "features". Returns (df, classes)."""
    k = (ds, dedup)
    if k in _MEM:
        return _MEM[k]
    base = ds_dir(cfg, ds)
    cache = os.path.join(base, "cache")
    os.makedirs(cache, exist_ok=True)
    tag = "dedupfeat" if dedup == "features" else ("dedup" if dedup else "raw")
    f = os.path.join(cache, f"clean_{tag}.pkl")
    if not os.path.exists(f):
        raw_f = os.path.join(cache, "loaded.pkl")
        if os.path.exists(raw_f):
            df0 = pd.read_pickle(raw_f)
        else:
            if verbose:
                print(f"Loading {ds} ...")
            df0, info = D.load_dataset(ds, cfg["datasets"][ds], verbose=verbose)
            df0.to_pickle(raw_f)
            json.dump(info, open(os.path.join(base, "load_info.json"), "w"), indent=2, default=str)
        if not os.path.exists(os.path.join(base, "classes.json")):
            json.dump(D.class_list(df0), open(os.path.join(base, "classes.json"), "w"))
        df, cinfo = D.clean(df0, dedup=dedup)
        df.to_pickle(f)
        json.dump(cinfo, open(os.path.join(base, f"clean_{tag}.json"), "w"), indent=2, default=int)
        del df0
    df = pd.read_pickle(f)
    # class index is fixed from the loaded data, so indices agree across protocols even if
    # deduplication removes every flow of a class
    classes = json.load(open(os.path.join(base, "classes.json")))
    _MEM[k] = (df, classes)
    return df, classes


def stratified_cap(y, cap, seed, keep_small=2000):
    """Proportional stratified subsample to <= cap rows; classes with <= keep_small rows kept whole."""
    if cap is None or len(y) <= cap:
        return np.arange(len(y))
    rng = np.random.default_rng(seed + 991)
    classes, counts = np.unique(y, return_counts=True)
    small = counts <= keep_small
    budget = max(cap - counts[small].sum(), 1)
    big_total = counts[~small].sum()
    idx = []
    for c, n, s in zip(classes, counts, small):
        ci = np.flatnonzero(y == c)
        take = n if s else max(1, int(round(budget * n / big_total)))
        idx.append(ci if take >= n else rng.choice(ci, take, replace=False))
    return np.sort(np.concatenate(idx))


def prepare(cfg, ds, protocol, seed, L, task):
    """Split, fit preprocessing on train, build causal windows. Returns dict of parts tr/va/te, each
    {X [n_part, d], widx [m, L], y [m]} (train/val rows capped per config; test never capped)."""
    df, classes = load_clean(cfg, ds, dedup=DEDUP_MODE[protocol])
    frac = tuple(cfg["split_frac"])
    no_embargo = []
    if protocol.startswith("SC"):
        part, no_embargo = D.split_chrono_per_class(df, embargo=L, frac=frac)
    else:
        part = D.split_random(df, seed, frac=frac)
    feats = D.feature_columns(df)
    cmap = {c: i for i, c in enumerate(classes)}
    y_all = df[D.LABEL].map(cmap).values.astype(np.int64)
    if task == "binary":
        y_all = (y_all != 0).astype(np.int64)
    C = 2 if task == "binary" else len(classes)
    pre = D.Preprocessor().fit(df.loc[part == 0, feats].to_numpy(), feats)
    mp = mode_params(cfg)
    caps = {"tr": train_cap(cfg, ds), "va": mp.get("val_cap"), "te": None}
    out = {"C": C, "classes": classes if task == "multi" else ["Normal", "Attack"], "features": pre.names,
           "no_embargo_classes": no_embargo}
    for name, p in (("tr", 0), ("va", 1), ("te", 2)):
        mask = part == p
        dp = df.loc[mask, [c for c in ("Src IP", "Dst IP") if c in df.columns] + ["_t", "_row", D.LABEL]] \
            .reset_index(drop=True)
        X = pre.transform(df.loc[mask, feats].to_numpy())
        widx = D.window_index(dp, L, cfg["window_key"])
        y = y_all[mask]
        keep = stratified_cap(y, caps[name], seed)
        out[name] = {"X": X, "widx": widx[keep], "y": y[keep]}
        if name == "te":
            out["purity_te"] = D.host_label_purity(dp, cfg["window_key"])
    return out


# ================================================================ hyper-parameters
SEARCH_SPACE_TEXT = {
    "RF": "n_estimators {100, 200, 300}; max_depth {None, 20, 30}; min_samples_leaf {1, 2, 5}",
    "XGBoost": "n_estimators 100-400 (step 50); max_depth 4-10; learning_rate 0.03-0.3 (log)",
    "neural": "hidden {32, 64, 128}; dropout 0.0-0.3; learning rate 3e-4-3e-3 (log); "
              "layers {1, 2} for LSTM/GRU/BiLSTM/Transformer; AdamW, batch 512, early stopping on val. loss",
}


def default_hp(model):
    if model == "RF":
        return {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1}
    if model == "XGBoost":
        return {"n_estimators": 300, "max_depth": 8, "learning_rate": 0.1}
    hp = {"hidden": 64, "dropout": 0.2, "lr": 1e-3}
    if model in ("LSTM", "GRU", "BiLSTM"):
        hp["layers"] = 1
    if model == "Transformer":
        hp.update({"layers": 2, "heads": 4, "dropout": 0.1})
    return hp


def suggest(trial, model):
    if model == "RF":
        return {"n_estimators": trial.suggest_categorical("n_estimators", [100, 200, 300]),
                "max_depth": trial.suggest_categorical("max_depth", [None, 20, 30]),
                "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", [1, 2, 5])}
    if model == "XGBoost":
        return {"n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
                "max_depth": trial.suggest_int("max_depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.3, log=True)}
    hp = {"hidden": trial.suggest_categorical("hidden", [32, 64, 128]),
          "dropout": trial.suggest_float("dropout", 0.0, 0.3),
          "lr": trial.suggest_float("lr", 3e-4, 3e-3, log=True)}
    if model in ("LSTM", "GRU", "BiLSTM", "Transformer"):
        hp["layers"] = trial.suggest_categorical("layers", [1, 2])
    if model == "Transformer":
        hp["heads"] = 4
    return hp


def load_hp(cfg, ds, model, task):
    p = os.path.join(ds_dir(cfg, ds), "best_hp.json")
    if os.path.exists(p):
        h = json.load(open(p)).get(f"{model}|{task}")
        if h:
            return h
    warnings.warn(f"No tuned hyper-parameters for {ds}/{model}/{task}; using defaults (run notebook 02).")
    return default_hp(model)


# ================================================================ single run
def fit_eval(cfg, model, hp, d, seed, imbalance="none", with_cost=False, device=None):
    device = device or get_device(cfg)
    mp = mode_params(cfg)
    C = d["C"]
    set_seed(seed)
    if model in TABULAR:
        m, info = T.fit_tabular(model, hp, d["tr"], C, seed, imbalance, n_jobs=cfg["n_jobs"],
                                use_gpu=(device.type == "cuda"))
        Pva, Pte = T.predict_tabular(m, d["va"], C), T.predict_tabular(m, d["te"], C)
        cost = T.cost_tabular(m, d["tr"]["X"].shape[1], d["tr"]["widx"].shape[1]) if with_cost else {}
    else:
        m, info = T.fit_neural(model, hp, d["tr"], d["va"], C, seed, device, imbalance,
                               max_epochs=mp["max_epochs"], patience=mp["patience"], batch_size=cfg["batch_size"])
        Pva, Pte = T.predict_neural(m, d["va"], device), T.predict_neural(m, d["te"], device)
        cost = T.cost_neural(m, d["tr"]["X"].shape[1], d["tr"]["widx"].shape[1]) if with_cost else {}
    res = {"val": T.metrics(d["va"]["y"], Pva, C), "test": T.metrics(d["te"]["y"], Pte, C), **info, **cost}
    return res, Pte


def run_key(rec):
    s = json.dumps({k: rec[k] for k in ("exp", "dataset", "model", "task", "protocol", "L", "seed", "imbalance")},
                   sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()


def results_path(cfg, ds):
    return os.path.join(ds_dir(cfg, ds), "results.jsonl")


def done_keys(cfg, ds):
    p = results_path(cfg, ds)
    return {json.loads(l)["key"] for l in open(p) if l.strip()} if os.path.exists(p) else set()


def _save(cfg, ds, rec, Pte, yte):
    hist = rec.pop("history", None)
    with open(results_path(cfg, ds), "a") as f:
        f.write(json.dumps(rec, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
    base = ds_dir(cfg, ds)
    if hist:
        os.makedirs(os.path.join(base, "histories"), exist_ok=True)
        json.dump(hist, open(os.path.join(base, "histories", rec["key"] + ".json"), "w"))
    if cfg.get("save_preds"):
        os.makedirs(os.path.join(base, "preds"), exist_ok=True)
        np.savez_compressed(os.path.join(base, "preds", rec["key"] + ".npz"),
                            P=Pte.astype(np.float16), y=yte.astype(np.int16))


def run_grid(cfg, exp, datasets, tasks, protocols, Ls, seeds, models, imbalances=("none",), with_cost=False):
    """Run every (dataset, task, protocol, L, seed, model, imbalance) cell not yet in results.jsonl."""
    device = get_device(cfg)
    print(f"device = {device}")
    for ds in datasets:
        release_except(ds)
        done = done_keys(cfg, ds)
        for task in tasks:
            for protocol in protocols:
                for L in Ls:
                    for seed in seeds:
                        cells = []
                        for model in models:
                            for imb in imbalances:
                                if imb == "smote" and model not in TABULAR:
                                    continue
                                # for trees, every loss-based strategy reduces to the same class weights:
                                # run it once ("weighted_ce", reported as "class weighting")
                                if model in TABULAR and imb in ("focal", "logit_adjusted", "balanced_sampling"):
                                    continue
                                rec = {"exp": exp, "dataset": ds, "model": model, "task": task,
                                       "protocol": protocol, "L": int(L), "seed": int(seed), "imbalance": imb}
                                rec["key"] = run_key(rec)
                                if rec["key"] not in done:
                                    cells.append(rec)
                        if not cells:
                            continue
                        d = prepare(cfg, ds, protocol, seed, L, task)
                        for rec in cells:
                            hp = load_hp(cfg, ds, rec["model"], task)
                            t0 = time.time()
                            res, Pte = fit_eval(cfg, rec["model"], hp, d, seed, rec["imbalance"], with_cost, device)
                            rec.update(res)
                            rec.update({"hp": hp, "classes": d["classes"], "n_train": int(len(d["tr"]["y"])),
                                        "n_val": int(len(d["va"]["y"])), "n_test": int(len(d["te"]["y"])),
                                        "n_features": len(d["features"]), "purity_te": d["purity_te"],
                                        "no_embargo_classes": d["no_embargo_classes"], "device": str(device),
                                        "wall_s": time.time() - t0})
                            _save(cfg, ds, rec, Pte, d["te"]["y"])
                            print(f"[{exp}] {ds} {task} {protocol} L={L} s={seed} {rec['model']:11s} "
                                  f"{rec['imbalance']:17s} test macro-F1={rec['test']['macro_f1']:.4f} "
                                  f"({rec['wall_s']:.0f}s)", flush=True)
                        del d


def tune(cfg, datasets, tasks, models, L, trials):
    """Equal-budget Optuna (TPE) search per dataset x task x model on S-C validation macro-F1 (seed 0).
    The default configuration is always the first trial."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    device = get_device(cfg)
    for ds in datasets:
        release_except(ds)
        p = os.path.join(ds_dir(cfg, ds), "best_hp.json")
        best = json.load(open(p)) if os.path.exists(p) else {}
        for task in tasks:
            todo = [m for m in models if f"{m}|{task}" not in best]
            if not todo:
                continue
            d = prepare(cfg, ds, "SC", 0, L, task)
            for model in todo:
                log = open(os.path.join(ds_dir(cfg, ds), "tuning_log.jsonl"), "a")

                def objective(trial):
                    hp = suggest(trial, model)
                    t0 = time.time()
                    res, _ = fit_eval(cfg, model, hp, d, 0, device=device)
                    v = res["val"]["macro_f1"]
                    log.write(json.dumps({"dataset": ds, "task": task, "model": model, "trial": trial.number,
                                          "hp": hp, "val_macro_f1": v, "wall_s": time.time() - t0}, default=str) + "\n")
                    log.flush()
                    return v

                st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
                st.enqueue_trial({k: v for k, v in default_hp(model).items() if k != "heads"})
                t0 = time.time()
                st.optimize(objective, n_trials=trials)
                log.close()
                best[f"{model}|{task}"] = suggest(optuna.trial.FixedTrial(st.best_params), model)
                json.dump(best, open(p, "w"), indent=2)
                print(f"[tune] {ds} {task} {model:11s} best val macro-F1={st.best_value:.4f} "
                      f"({time.time() - t0:.0f}s) {best[f'{model}|{task}']}", flush=True)


# ================================================================ reading results back
def load_results(cfg, datasets=None) -> pd.DataFrame:
    rows = []
    for ds in datasets or list(cfg["datasets"]):
        p = results_path(cfg, ds)
        if not os.path.exists(p):
            continue
        for l in open(p):
            if not l.strip():
                continue
            r = json.loads(l)
            flat = {k: r.get(k) for k in ("key", "exp", "dataset", "model", "task", "protocol", "L", "seed",
                                          "imbalance", "train_time_s", "epochs", "n_train", "n_val", "n_test",
                                          "n_features", "purity_te", "params", "size_mb", "latency_ms_b1",
                                          "latency_ms_b1_p95", "throughput_fps", "flops_per_flow", "wall_s")}
            flat["classes"] = r.get("classes")
            for k, v in r["test"].items():
                flat[k] = v
            for k, v in r["val"].items():
                flat["val_" + k] = v
            rows.append(flat)
    df = pd.DataFrame(rows)
    for c in ("params", "size_mb", "latency_ms_b1", "latency_ms_b1_p95", "throughput_fps", "flops_per_flow",
              "train_time_s", "epochs", "purity_te"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if len(df):
        df["minority_recall"] = df.apply(_minority_recall, axis=1)
    return df


def _minority_recall(row):
    cls = row["classes"] or []
    if row["task"] != "multi":
        return np.nan
    vals = [row.get(f"recall_c{i}") for i, c in enumerate(cls) if c in D.MINORITY]
    vals = [v for v in vals if v is not None and not pd.isna(v)]
    return float(np.mean(vals)) if vals else np.nan


def choose_best_L(cfg, ds=None):
    """Best window length from B3, chosen on VALIDATION macro-F1 averaged over sequence models."""
    if cfg.get("best_L"):
        return int(cfg["best_L"])
    ds = ds or cfg["primary_dataset"]
    r = load_results(cfg, [ds])
    r = r[(r.exp == "B3") & (r.task == "multi") & (~r.model.isin(TABULAR))]
    if r.empty:
        raise RuntimeError("Run notebook 03 (B3) first, or set best_L in config.json")
    return int(r.groupby("L")["val_macro_f1"].mean().idxmax())


def choose_top_models(cfg, k=4, ds=None, L=None):
    """Top-k models by VALIDATION macro-F1 in the B2 (S-C) runs — used to select B4 models."""
    if cfg.get("b4_models"):
        return list(cfg["b4_models"])
    ds = ds or cfg["primary_dataset"]
    L = L or choose_best_L(cfg, ds)
    r = load_results(cfg, [ds])
    r = r[(r.exp == "B1") & (r.protocol == "SC") & (r.task == "multi") & (r.L == L)]
    if r.empty:
        raise RuntimeError("Run notebook 04 (B1/B2) first, or set b4_models in config.json")
    return list(r.groupby("model")["val_macro_f1"].mean().sort_values(ascending=False).index[:k])
