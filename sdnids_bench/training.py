"""Training, prediction, metrics and computational-cost measurement.

Neural models never materialise the full [n, L, d] window tensor: each mini-batch is gathered on the fly
from the flow matrix X [n, d] and the window index [n, L] (so CICIDS2017 / CSE-CIC-IDS2018 fit in RAM)."""
from __future__ import annotations

import copy
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, recall_score, precision_score, roc_curve, confusion_matrix)

from .datasets import gather
from .models import build_neural, n_params


# ================================================================ losses
class FocalLoss(nn.Module):
    """Lin et al. (2017), gamma = 2."""

    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, y):
        logp = F.log_softmax(logits, 1)
        lp = logp.gather(1, y[:, None]).squeeze(1)
        return (-(1 - lp.exp()) ** self.gamma * lp).mean()


class LogitAdjustedLoss(nn.Module):
    """Menon et al. (2021): cross-entropy on logits + tau * log(class prior)."""

    def __init__(self, prior, tau=1.0):
        super().__init__()
        self.register_buffer("adj", tau * torch.log(torch.as_tensor(prior, dtype=torch.float32) + 1e-12))

    def forward(self, logits, y):
        return F.cross_entropy(logits + self.adj, y)


def class_weights(y, C):
    cnt = np.bincount(y, minlength=C).astype(float)
    w = np.where(cnt > 0, cnt.sum() / (C * np.maximum(cnt, 1)), 0.0)
    return torch.tensor(w / w[cnt > 0].mean(), dtype=torch.float32)


def make_loss(kind, y, C):
    if kind in ("none", "balanced_sampling", "smote"):
        return nn.CrossEntropyLoss()
    if kind == "weighted_ce":
        return nn.CrossEntropyLoss(weight=class_weights(y, C))
    if kind == "focal":
        return FocalLoss(2.0)
    if kind == "logit_adjusted":
        return LogitAdjustedLoss(np.bincount(y, minlength=C) / len(y))
    raise ValueError(kind)


# ================================================================ neural
def _to(X, widx, rows, device):
    return torch.from_numpy(gather(X, widx[rows])).to(device, non_blocking=True)


def fit_neural(name, hp, tr, va, C, seed, device, imbalance="none", max_epochs=50, patience=5,
               batch_size=512, time_budget=None):
    """tr / va = dict(X=[n,d] float32, widx=[n,L] int64, y=[n] int). Early stopping on validation loss."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    L, d = tr["widx"].shape[1], tr["X"].shape[1]
    hp = dict(hp)
    lr, wd = hp.pop("lr", 1e-3), hp.pop("weight_decay", 1e-5)
    model = build_neural(name, d, L, C, hp).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = make_loss(imbalance, tr["y"], C).to(device)
    ytr = torch.from_numpy(tr["y"].astype(np.int64))
    yva = torch.from_numpy(va["y"].astype(np.int64))
    n = len(ytr)
    p_sample = None
    if imbalance == "balanced_sampling":
        cnt = np.bincount(tr["y"], minlength=C).astype(float)
        p_sample = 1.0 / cnt[tr["y"]]
        p_sample /= p_sample.sum()
    best, best_state, bad, epochs, t0 = np.inf, None, 0, 0, time.time()
    history = []
    for ep in range(max_epochs):
        model.train()
        order = rng.choice(n, n, replace=True, p=p_sample) if p_sample is not None else rng.permutation(n)
        tl = 0.0
        for i in range(0, n, batch_size):
            b = order[i:i + batch_size]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(_to(tr["X"], tr["widx"], b, device)), ytr[b].to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tl += loss.item() * len(b)
        epochs = ep + 1
        vl = val_loss(model, va, yva, device)
        history.append({"epoch": epochs, "train_loss": tl / n, "val_loss": vl})
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break
        if time_budget and time.time() - t0 > time_budget:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {"train_time_s": time.time() - t0, "epochs": epochs, "best_val_loss": best,
                   "history": history}


@torch.no_grad()
def val_loss(model, va, yva, device, bs=8192):
    model.eval()
    tot = 0.0
    for i in range(0, len(yva), bs):
        rows = np.arange(i, min(i + bs, len(yva)))
        tot += F.cross_entropy(model(_to(va["X"], va["widx"], rows, device)), yva[rows].to(device),
                               reduction="sum").item()
    return tot / len(yva)


@torch.no_grad()
def predict_neural(model, part, device, bs=8192):
    model.eval()
    out = []
    n = part["widx"].shape[0]
    for i in range(0, n, bs):
        rows = np.arange(i, min(i + bs, n))
        out.append(F.softmax(model(_to(part["X"], part["widx"], rows, device)), 1).float().cpu().numpy())
    return np.concatenate(out)


# ================================================================ tabular (flattened window)
def flat(part):
    return gather(part["X"], part["widx"]).reshape(part["widx"].shape[0], -1)


def fit_tabular(name, hp, tr, C, seed, imbalance="none", n_jobs=-1, use_gpu=False):
    t0 = time.time()
    Xf, y = flat(tr), tr["y"]
    sw = None
    if imbalance in ("weighted_ce", "balanced_sampling", "focal", "logit_adjusted"):
        sw = class_weights(y, C).numpy()[y]          # tree analogue of re-weighting
    if imbalance == "smote":
        from imblearn.over_sampling import SMOTE
        cnt = np.bincount(y, minlength=C)
        k = int(max(1, min(5, cnt[cnt > 0].min() - 1)))
        Xf, y = SMOTE(k_neighbors=k, random_state=seed).fit_resample(Xf, y)
    if name == "RF":
        from sklearn.ensemble import RandomForestClassifier
        m = RandomForestClassifier(n_estimators=hp.get("n_estimators", 200), max_depth=hp.get("max_depth"),
                                   min_samples_leaf=hp.get("min_samples_leaf", 1),
                                   max_features=hp.get("max_features", "sqrt"), n_jobs=n_jobs, random_state=seed)
    else:
        from xgboost import XGBClassifier
        m = XGBClassifier(n_estimators=hp.get("n_estimators", 300), max_depth=hp.get("max_depth", 8),
                          learning_rate=hp.get("learning_rate", 0.1), subsample=hp.get("subsample", 0.9),
                          colsample_bytree=hp.get("colsample_bytree", 0.8), tree_method="hist",
                          device="cuda" if use_gpu else "cpu", n_jobs=n_jobs, random_state=seed,
                          eval_metric="mlogloss" if C > 2 else "logloss")
        # XGBoost needs contiguous labels 0..k-1
        present = np.unique(y)
        m._sdn_classes = present
        y = np.searchsorted(present, y)
    m.fit(Xf, y, sample_weight=sw)
    return m, {"train_time_s": time.time() - t0, "epochs": None, "best_val_loss": None, "history": None}


def predict_tabular(m, part, C, bs=200_000):
    n = part["widx"].shape[0]
    out = []
    for i in range(0, n, bs):
        sub = {"X": part["X"], "widx": part["widx"][i:i + bs]}
        P = m.predict_proba(flat(sub))
        classes = getattr(m, "_sdn_classes", getattr(m, "classes_", np.arange(P.shape[1])))
        full = np.zeros((P.shape[0], C), dtype=np.float32)
        full[:, classes] = P
        out.append(full)
    return np.concatenate(out)


# ================================================================ metrics
def metrics(y, P, C, normal=0):
    yhat = P.argmax(1)
    labels = list(range(C))
    present = [c for c in labels if (y == c).any()]
    res = {
        "accuracy": accuracy_score(y, yhat),
        "macro_f1": f1_score(y, yhat, labels=present, average="macro", zero_division=0),
        "weighted_f1": f1_score(y, yhat, labels=present, average="weighted", zero_division=0),
        "macro_precision": precision_score(y, yhat, labels=present, average="macro", zero_division=0),
        "macro_recall": recall_score(y, yhat, labels=present, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y, yhat),
        "balanced_acc": balanced_accuracy_score(y, yhat),
    }
    big = [c for c in present if (y == c).sum() >= 10]
    res["macro_f1_min10"] = f1_score(y, yhat, labels=big, average="macro", zero_division=0) if big else np.nan
    aps = [average_precision_score((y == c).astype(int), P[:, c]) for c in present if not (C == 2 and c == normal)]
    res["pr_auc_macro"] = float(np.mean(aps)) if aps else np.nan
    rec = recall_score(y, yhat, labels=labels, average=None, zero_division=0)
    f1c = f1_score(y, yhat, labels=labels, average=None, zero_division=0)
    for c in labels:
        res[f"recall_c{c}"] = float(rec[c]) if (y == c).any() else np.nan
        res[f"f1_c{c}"] = float(f1c[c]) if (y == c).any() else np.nan
        res[f"n_test_c{c}"] = int((y == c).sum())
    is_att, pred_att = y != normal, yhat != normal
    res["fpr"] = float(pred_att[~is_att].mean()) if (~is_att).any() else np.nan
    res["fnr"] = float((~pred_att)[is_att].mean()) if is_att.any() else np.nan
    res["detection_rate"] = float(pred_att[is_att].mean()) if is_att.any() else np.nan
    if is_att.any() and (~is_att).any():
        fpr, tpr, _ = roc_curve(is_att.astype(int), 1.0 - P[:, normal])
        res["fpr_at_95dr"] = float(fpr[min(np.searchsorted(tpr, 0.95), len(fpr) - 1)])
        from sklearn.metrics import roc_auc_score
        res["roc_auc_attack"] = float(roc_auc_score(is_att.astype(int), 1.0 - P[:, normal]))
    return res


def confusion(y, P, C):
    return confusion_matrix(y, P.argmax(1), labels=list(range(C)))


# ================================================================ cost
def recurrent_flops(m, L):
    """Analytic multiply-add FLOPs (x2) of every nn.LSTM / nn.GRU in the model for one window of length L.
    Per time step and direction: gates x (input x hidden + hidden x hidden) multiply-adds."""
    total = 0
    for mod in m.modules():
        if isinstance(mod, (nn.LSTM, nn.GRU)):
            gates = 4 if isinstance(mod, nn.LSTM) else 3
            dirs = 2 if mod.bidirectional else 1
            h = mod.hidden_size
            for layer in range(mod.num_layers):
                inp = mod.input_size if layer == 0 else h * dirs
                total += 2 * gates * (inp * h + h * h) * L * dirs
    return total


def model_flops(m, d, L):
    """FLOPs per window: torch's FLOP counter for every operation except the recurrent layers, whose
    fused kernels it does not see consistently; those are added analytically (recurrent_flops)."""
    from torch.utils.flop_counter import FlopCounterMode
    x1 = torch.zeros(1, L, d)
    was_training = m.training
    m.train()       # disables PyTorch's fused inference fast path (invisible to the counter); dropout costs 0 FLOPs
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        m(x1)
    m.train(was_training)
    total = int(fc.get_total_flops())
    counts = fc.get_flop_counts()
    for name, mod in m.named_modules():
        if isinstance(mod, (nn.LSTM, nn.GRU)) and name:
            for key, ops in counts.items():
                if key == name or key.endswith("." + name):
                    total -= int(sum(ops.values()))
                    break
    return total + recurrent_flops(m, L)


def cost_neural(model, d, L, reps=300):
    """CPU latency (batch 1, single thread, median of `reps`), CPU throughput (batch 1024, all threads),
    parameters, size, FLOPs per window."""
    m = copy.deepcopy(model).to("cpu").eval()
    x1, xb = torch.zeros(1, L, d), torch.zeros(1024, L, d)
    n_threads = torch.get_num_threads()
    with torch.no_grad():
        torch.set_num_threads(1)              # one request = one core, comparable across models
        for _ in range(20):
            m(x1)
        ts = []
        for _ in range(reps):
            t = time.perf_counter()
            m(x1)
            ts.append(time.perf_counter() - t)
        torch.set_num_threads(n_threads)
        m(xb)
        t = time.perf_counter()
        for _ in range(10):
            m(xb)
        thr = 10 * 1024 / (time.perf_counter() - t)
    p = n_params(m)
    try:
        flops = model_flops(m, d, L)
    except Exception:
        flops = None
    return {"params": p, "size_mb": p * 4 / 1e6, "latency_ms_b1": 1e3 * float(np.median(ts)),
            "latency_ms_b1_p95": 1e3 * float(np.percentile(ts, 95)), "throughput_fps": thr,
            "flops_per_flow": flops}


def cost_tabular(m, d, L, reps=200):
    x1 = np.zeros((1, L * d), dtype=np.float32)
    xb = np.zeros((1024, L * d), dtype=np.float32)
    params = m.get_params() if hasattr(m, "get_params") else {}
    if "device" in params:
        m.set_params(device="cpu")
    n_jobs_before = params.get("n_jobs")
    if "n_jobs" in params:
        m.set_params(n_jobs=1)                # single thread for batch-1 latency (no thread start-up cost)
    for _ in range(5):
        m.predict_proba(x1)
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        m.predict_proba(x1)
        ts.append(time.perf_counter() - t)
    if "n_jobs" in params:
        m.set_params(n_jobs=n_jobs_before)    # all cores for throughput
    t = time.perf_counter()
    for _ in range(5):
        m.predict_proba(xb)
    thr = 5 * 1024 / (time.perf_counter() - t)
    if hasattr(m, "estimators_"):
        nodes = int(sum(e.tree_.node_count for e in m.estimators_))
    else:
        try:
            nodes = int(m.get_booster().trees_to_dataframe().shape[0])
        except Exception:
            nodes = None
    return {"params": nodes, "size_mb": len(pickle.dumps(m)) / 1e6, "latency_ms_b1": 1e3 * float(np.median(ts)),
            "latency_ms_b1_p95": 1e3 * float(np.percentile(ts, 95)), "throughput_fps": thr,
            "flops_per_flow": None}
