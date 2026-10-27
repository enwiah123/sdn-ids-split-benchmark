"""Dataset loading, leakage-free preprocessing, splits and causal windowing.

Supported datasets (all CICFlowMeter-family flow records):
  * InSDN (Elsayed et al., 2020)                  loader="insdn"
  * CICIDS2017, corrected release (Engelen et al., 2021) or original   loader="cic"
  * CSE-CIC-IDS2018, corrected or original        loader="cic"

Pipeline
  1. read every CSV under the dataset folder (chunked), canonicalise column names and labels
  2. optional per-class cap (two-pass, uniform within class, order preserved) for very large datasets
  3. identifiers (Flow ID, IPs, Src Port, Timestamp) kept in a side table only — never model inputs
  4. inf/NaN -> 0; exact duplicates removed BEFORE splitting; conflicting duplicates (same features,
     different labels) removed entirely
  5. splits: S-R raw | S-R dedup | S-C per-class chronological with embargo
  6. variance filter, signed log1p and standardisation fitted on the training partition only
  7. causal windows: the L most recent flows sharing a context key (default destination host); label = last flow
"""
from __future__ import annotations

import glob
import os
import re
import warnings

import numpy as np
import pandas as pd

CANONICAL = ["Normal", "DDoS", "DoS", "Probe", "BFA", "Web-Attack", "BOTNET", "U2R", "Other"]
MINORITY = ["BFA", "Web-Attack", "BOTNET", "U2R", "Other"]
ID_COLS = ["Flow ID", "Src IP", "Src Port", "Dst IP", "Timestamp"]
LABEL = "Label"
META = ["_source", "_row", "_t"]
_ALIASES = {
    "flow id": "Flow ID", "flowid": "Flow ID", "flow_id": "Flow ID",
    "src ip": "Src IP", "source ip": "Src IP", "src_ip": "Src IP",
    "src port": "Src Port", "source port": "Src Port", "src_port": "Src Port",
    "dst ip": "Dst IP", "destination ip": "Dst IP", "dst_ip": "Dst IP",
    "dst port": "Dst Port", "destination port": "Dst Port", "dst_port": "Dst Port",
    "protocol": "Protocol", "timestamp": "Timestamp", "label": "Label",
}
_DROP_ALWAYS = {"id", "attempted category", "unnamed: 0", "unnamed:0"}


# ================================================================ labels
def canonical_label(raw: str, attempted: str = "benign"):
    """Map any InSDN / CICIDS2017 / CSE-CIC-IDS2018 label to the shared taxonomy.
    Returns None for rows to drop (e.g. 'Attempted' flows when attempted='drop')."""
    s = str(raw).strip().lower()
    if "attempted" in s:
        return "Normal" if attempted == "benign" else None
    s2 = re.sub(r"[^a-z0-9]", "", s)
    if s2 in ("", "nan", "none"):          # blank rows (e.g. original CICIDS2017 Thursday web-attack file)
        return None
    if s2 in ("benign", "normal"):
        return "Normal"
    if "ddos" in s2:
        return "DDoS"
    if s2.startswith("dos") or "dosattack" in s2:
        return "DoS"
    if "portscan" in s2 or s2 == "probe":
        return "Probe"
    if "web" in s2 or "xss" in s2 or "sqlinjection" in s2:
        return "Web-Attack"
    if "patator" in s2 or "bruteforce" in s2 or s2 == "bfa":
        return "BFA"
    if s2.startswith("bot"):
        return "BOTNET"
    if s2 == "u2r":
        return "U2R"
    if "infiltration" in s2 or "infilteration" in s2 or "heartbleed" in s2:
        return "Other"
    if s2 == "label":                      # repeated header rows (CSE-CIC-IDS2018)
        return None
    raise ValueError(f"Unmapped label: {raw!r} — add it to canonical_label()")


def _map_labels(series: pd.Series, attempted: str) -> pd.Series:
    lookup = {u: canonical_label(u, attempted) for u in series.unique()}
    return series.map(lookup)


# ================================================================ reading
def _canon_cols(cols):
    out = []
    for c in cols:
        c2 = str(c).strip()
        out.append(_ALIASES.get(c2.lower(), c2))
    return out


def _csv_files(path):
    files = sorted(glob.glob(os.path.join(path, "**", "*.csv"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No CSV files under {path}")
    return files


def _read(path, **kw):
    return pd.read_csv(path, low_memory=False, encoding="utf-8", encoding_errors="replace",
                       skipinitialspace=True, **kw)


def _label_col_raw(path):
    head = _read(path, nrows=0)
    for raw, can in zip(head.columns, _canon_cols(head.columns)):
        if can == LABEL:
            return raw
    return None


def parse_timestamps(s: pd.Series, default_dayfirst: bool = True):
    """Parse CICFlowMeter timestamps robustly. Day-first vs month-first is decided from the data
    (a first field > 12 means day-first); ISO strings are parsed directly. Returns (int64 ns, info)."""
    s = s.astype(str).str.strip()
    info = {}
    m = s.str.extract(r"^(\d{1,4})[/\-.](\d{1,2})[/\-.](\d{1,4})")
    if m[0].str.len().max() == 4:
        dt = pd.to_datetime(s, errors="coerce", format="mixed")
        info["format"] = "ISO"
    else:
        a = pd.to_numeric(m[0], errors="coerce")
        b = pd.to_numeric(m[1], errors="coerce")
        if (a > 12).any():
            dayfirst, info["format"] = True, "day-first (detected)"
        elif (b > 12).any():
            dayfirst, info["format"] = False, "month-first (detected)"
        else:
            dayfirst, info["format"] = default_dayfirst, f"ambiguous; dayfirst={default_dayfirst} assumed"
        dt = pd.to_datetime(s, errors="coerce", dayfirst=dayfirst, format="mixed")
    has_ampm = s.str.contains(r"\b(AM|PM)\b", case=False, regex=True).any()
    hours = dt.dt.hour
    info["nat_share"] = float(dt.isna().mean())
    info["twelve_hour_without_ampm_suspected"] = bool((not has_ampm) and hours.notna().any()
                                                      and hours.max() <= 12 and info["format"] != "ISO")
    t = np.full(len(s), -1, dtype=np.int64)
    ok = dt.notna().values
    t[ok] = dt[ok].astype("int64").values
    return t, info


def load_dataset(name: str, spec: dict, sample_seed: int = 12345, chunksize: int = 500_000, verbose=True):
    """Load one dataset into a single DataFrame of canonical columns.

    spec keys: path, loader ('insdn'|'cic'), max_rows_per_class (None = keep all), attempted ('benign'|'drop').
    The per-class cap keeps each row of class c with probability min(1, cap / n_c): uniform within class,
    chronological order preserved, deterministic for a fixed sample_seed."""
    files = _csv_files(spec["path"])
    attempted = spec.get("attempted", "benign")
    cap = spec.get("max_rows_per_class")
    info = {"dataset": name, "files": [os.path.relpath(f, spec["path"]) for f in files]}

    # ---- pass 1: label counts (cheap; only the label column)
    raw_counts = {}
    for f in files:
        lc = _label_col_raw(f)
        if lc is None:
            continue
        for chunk in _read(f, usecols=[lc], chunksize=2_000_000):
            lab = _map_labels(chunk[lc], attempted)
            for k, v in lab.value_counts(dropna=True).items():
                raw_counts[k] = raw_counts.get(k, 0) + int(v)
    info["per_class_raw"] = raw_counts
    keep_p = {c: (1.0 if cap is None else min(1.0, cap / n)) for c, n in raw_counts.items()}
    info["keep_probability"] = keep_p

    # ---- pass 2: read, canonicalise, sample
    rng = np.random.default_rng(sample_seed)
    parts, feat_sets, ts_info, offset = [], [], {}, 0
    for f in files:
        if _label_col_raw(f) is None:
            continue
        fparts = []
        for chunk in _read(f, chunksize=chunksize):
            chunk.columns = _canon_cols(chunk.columns)
            chunk = chunk.loc[:, ~pd.Index(chunk.columns).duplicated()]
            chunk = chunk[[c for c in chunk.columns if c.lower() not in _DROP_ALWAYS]]
            lab = _map_labels(chunk[LABEL], attempted)
            chunk = chunk.assign(**{LABEL: lab})
            chunk["_row"] = offset + np.arange(len(chunk))
            offset += len(chunk)
            chunk = chunk[chunk[LABEL].notna()]
            if cap is not None:
                p = chunk[LABEL].map(keep_p).values
                chunk = chunk[rng.random(len(chunk)) < p]
            fparts.append(chunk)
        fdf = pd.concat(fparts, ignore_index=True)
        fdf["_source"] = os.path.basename(f)
        feats = [c for c in fdf.columns if c not in ID_COLS + [LABEL] + META]
        for c in feats:
            if not pd.api.types.is_numeric_dtype(fdf[c]):
                fdf[c] = pd.to_numeric(fdf[c], errors="coerce")
        feat_sets.append({c for c in feats if fdf[c].notna().mean() > 0.5})
        if "Timestamp" in fdf.columns:
            fdf["_t"], ts_info[os.path.basename(f)] = parse_timestamps(fdf["Timestamp"])
        else:
            fdf["_t"] = -1
        parts.append(fdf)
        if verbose:
            print(f"  read {os.path.basename(f)}: {len(fdf):,} rows")
    common = set.intersection(*feat_sets)
    # keep only what later steps need: features, the IPs (window grouping), label, order/time
    keep_cols = [c for c in parts[0].columns if c in common] + \
                [c for c in ("Src IP", "Dst IP") if any(c in p.columns for p in parts)] + [LABEL] + META
    df = pd.concat([p.reindex(columns=keep_cols) for p in parts], ignore_index=True)
    feats = feature_columns(df)
    df[feats] = df[feats].astype(np.float32)
    info["timestamp_parsing"] = ts_info
    info["n_features_common"] = len(feats)
    info["dropped_noncommon_features"] = sorted(set.union(*feat_sets) - common)
    info["per_class_loaded"] = df[LABEL].value_counts().to_dict()
    if any(v.get("twelve_hour_without_ampm_suspected") for v in ts_info.values()):
        warnings.warn(f"{name}: timestamps look like a 12-hour clock without AM/PM (known fault of the "
                      "original CICIDS2017). Use the corrected release for chronological splits.")
    return df, info


def feature_columns(df: pd.DataFrame) -> list[str]:
    drop = set(ID_COLS) | {LABEL} | set(META)
    return [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]


def class_list(df: pd.DataFrame) -> list[str]:
    """Classes present, in canonical order (Normal first)."""
    present = set(df[LABEL].unique())
    return [c for c in CANONICAL if c in present]


def clean(df: pd.DataFrame, dedup) -> tuple[pd.DataFrame, dict]:
    """dedup = False (keep all rows), True (drop exact feature+label duplicates, then every row whose
    feature vector occurs with more than one label) or "features" (label-free: order rows by time and keep
    the earliest occurrence of each feature vector with its own label; labels never decide inclusion)."""
    feats = feature_columns(df)
    if dedup == "features":
        df = df.copy()
        X = df[feats].to_numpy(dtype=np.float32, copy=True)
        bad = ~np.isfinite(X)
        info = {"rows_in": len(df), "nonfinite_values_replaced": int(bad.sum()), "mode": "features"}
        if bad.any():
            X[bad] = 0.0
            df[feats] = X
        info["per_class_in"] = df[LABEL].value_counts().to_dict()
        df = df.sort_values(["_t", "_row"], kind="mergesort")
        before = len(df)
        df = df.drop_duplicates(subset=feats, keep="first")
        info["feature_duplicates_removed"] = before - len(df)
        info["rows_out"] = len(df)
        info["per_class_out"] = df[LABEL].value_counts().to_dict()
        return df.sort_values("_row", kind="mergesort").reset_index(drop=True), info
    df = df.copy()
    X = df[feats].to_numpy(dtype=np.float32, copy=True)   # writable copy (pandas >= 3 returns read-only views)
    bad = ~np.isfinite(X)
    info = {"rows_in": len(df), "nonfinite_values_replaced": int(bad.sum())}
    if bad.any():
        X[bad] = 0.0
        df[feats] = X
    info["per_class_in"] = df[LABEL].value_counts().to_dict()
    if dedup:
        before = len(df)
        df = df.drop_duplicates(subset=feats + [LABEL], keep="first")
        info["exact_duplicates_removed"] = before - len(df)
        conflict = df.duplicated(subset=feats, keep=False)
        info["conflicting_rows_removed"] = int(conflict.sum())
        df = df[~conflict]
    info["rows_out"] = len(df)
    info["per_class_out"] = df[LABEL].value_counts().to_dict()
    return df.reset_index(drop=True), info


# ================================================================ splits
def split_random(df: pd.DataFrame, seed: int, frac=(0.70, 0.15, 0.15)) -> np.ndarray:
    """S-R: stratified random split -> array of 0/1/2 (train/val/test)."""
    rng = np.random.default_rng(seed)
    part = np.zeros(len(df), dtype=np.int8)
    for _, idx in df.groupby(LABEL).indices.items():
        idx = rng.permutation(idx)
        n = len(idx)
        n_tr, n_va = int(round(frac[0] * n)), int(round(frac[1] * n))
        part[idx[n_tr:n_tr + n_va]] = 1
        part[idx[n_tr + n_va:]] = 2
    return part


def split_chrono_per_class(df: pd.DataFrame, embargo: int, frac=(0.70, 0.15, 0.15)):
    """S-C: within each class, order by time (ties: original row order); first 70% train, next 15%
    validation, last 15% test. `embargo` flows before each boundary are dropped (-1). Classes with fewer
    than 20*embargo flows keep all flows (embargo 0); this is returned for reporting."""
    part = np.full(len(df), -1, dtype=np.int8)
    no_embargo = []
    for c, idx in df.groupby(LABEL).indices.items():
        order = idx[np.lexsort((df["_row"].values[idx], df["_t"].values[idx]))]
        n = len(order)
        e = embargo if n > 20 * embargo else 0
        if e == 0 and embargo > 0:
            no_embargo.append(c)
        n_tr, n_va = int(round(frac[0] * n)), int(round(frac[1] * n))
        part[order[:max(n_tr - e, 1)]] = 0
        part[order[n_tr:n_tr + n_va - e]] = 1
        part[order[n_tr + n_va:]] = 2
    return part, no_embargo


# ================================================================ preprocessing (fit on train only)
class Preprocessor:
    """Variance filter + signed log1p + standardisation; clip to +/-10 SD."""

    def __init__(self, var_eps: float = 1e-8):
        self.var_eps = var_eps

    @staticmethod
    def _log(X):
        X = X.astype(np.float64, copy=False)
        return np.sign(X) * np.log1p(np.abs(X))

    def fit(self, X: np.ndarray, names: list[str]):
        Xl = self._log(X)
        self.keep = Xl.var(axis=0) > self.var_eps
        self.names = [n for n, k in zip(names, self.keep) if k]
        Xk = Xl[:, self.keep]
        self.mu, self.sd = Xk.mean(axis=0), Xk.std(axis=0) + 1e-6
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = (self._log(X)[:, self.keep] - self.mu) / self.sd
        return np.clip(Z, -10, 10).astype(np.float32)


# ================================================================ windows
def context_keys(df_part: pd.DataFrame, key: str) -> np.ndarray:
    if key == "pair" and {"Src IP", "Dst IP"} <= set(df_part.columns):
        return (df_part["Src IP"].astype(str) + ">" + df_part["Dst IP"].astype(str)).values
    if key in df_part.columns:
        return df_part[key].astype(str).values
    return np.zeros(len(df_part), dtype=object)          # no key: one global stream ordered by time


def window_index(df_part: pd.DataFrame, L: int, key: str = "Dst IP") -> np.ndarray:
    """[n, L] positional indices of the L most recent flows sharing the context key (current flow last),
    -1 padded. Only flows of the same partition and at or before the current flow's time are used."""
    n = len(df_part)
    out = np.full((n, L), -1, dtype=np.int64)
    if L == 1:
        out[:, 0] = np.arange(n)
        return out
    keys = context_keys(df_part, key)
    order = np.lexsort((df_part["_row"].values, df_part["_t"].values, keys))
    ks = keys[order]
    starts = np.r_[0, np.flatnonzero(ks[1:] != ks[:-1]) + 1]
    grp_start = np.repeat(starts, np.diff(np.r_[starts, n]))
    pos = np.arange(n)
    for j in range(L):
        src = pos - (L - 1 - j)
        ok = src >= grp_start
        out[order[ok], j] = order[src[ok]]
    return out


def gather(X: np.ndarray, widx: np.ndarray) -> np.ndarray:
    """[n, L, d] window tensor with zero padding (use on batches, not whole datasets)."""
    Z = np.zeros((widx.shape[0], widx.shape[1], X.shape[1]), dtype=np.float32)
    m = widx >= 0
    Z[m] = X[widx[m]]
    return Z


def host_label_purity(df_part: pd.DataFrame, key: str = "Dst IP") -> float:
    """Share of flows whose context key carries a single class (leakage indicator; report it)."""
    has_key = {"Src IP", "Dst IP"} <= set(df_part.columns) if key == "pair" else key in df_part.columns
    if not has_key:
        return float("nan")
    keys = context_keys(df_part, key)
    s = pd.DataFrame({"k": keys, "y": df_part[LABEL].values})
    nun = s.groupby("k")["y"].nunique()
    return float(s["k"].isin(nun[nun == 1].index).mean())


def dataset_profile(load_info: dict, clean_raw: dict, clean_dedup: dict, classes: list[str]) -> pd.DataFrame:
    """Table 1 rows: per class, flows in source -> loaded (after any cap) -> after deduplication."""
    rows = []
    for c in classes:
        rows.append({"Class": c,
                     "Source flows": load_info["per_class_raw"].get(c, 0),
                     "Loaded flows": load_info["per_class_loaded"].get(c, 0),
                     "After deduplication": clean_dedup["per_class_out"].get(c, 0)})
    t = pd.DataFrame(rows)
    tot = t[["Source flows", "Loaded flows", "After deduplication"]].sum()
    t["Share after dedup. (%)"] = 100 * t["After deduplication"] / max(tot["After deduplication"], 1)
    t["Removed by dedup. (%)"] = 100 * (1 - t["After deduplication"] / t["Loaded flows"].clip(lower=1))
    total = {"Class": "Total", **tot.to_dict(), "Share after dedup. (%)": 100.0,
             "Removed by dedup. (%)": 100 * (1 - tot["After deduplication"] / max(tot["Loaded flows"], 1))}
    return pd.concat([t, pd.DataFrame([total])], ignore_index=True)
