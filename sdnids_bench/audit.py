"""Data-preparation audit: what deduplication removes, whether labels decide inclusion,
how duplicates fall across the per-class chronological split, and whether a global time cut is feasible.
Every function works on the cached raw (non-deduplicated) data."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import datasets as D


def _feat_key(df: pd.DataFrame, feats: list[str]) -> pd.Series:
    """One hash per feature vector (identical vectors -> identical key)."""
    return pd.util.hash_pandas_object(df[feats], index=False)


def duplicate_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Per class: rows, exact duplicates (same features and label), rows in conflicting groups (same features,
    more than one label), and what the published rule and the label-free rule would keep."""
    feats = D.feature_columns(df)
    k = _feat_key(df, feats)
    tmp = pd.DataFrame({"k": k.values, "y": df[D.LABEL].values})
    n_lab = tmp.groupby("k")["y"].nunique()
    conflict = tmp["k"].map(n_lab).values > 1
    exact_dup = tmp.duplicated(subset=["k", "y"], keep="first").values
    feat_dup_earliest = df.assign(_k=k.values).sort_values(["_t", "_row"]).duplicated("_k", keep="first") \
        .reindex(df.index).values
    rows = []
    for c, idx in tmp.groupby("y").indices.items():
        rows.append({"class": c, "rows": len(idx),
                     "exact_duplicates": int(exact_dup[idx].sum()),
                     "rows_in_conflicting_groups": int(conflict[idx].sum()),
                     "kept_published_rule": int((~exact_dup[idx] & ~conflict[idx]).sum()),
                     "kept_label_free_rule": int((~feat_dup_earliest[idx]).sum()),
                     "distinct_vectors": int(tmp["k"].iloc[idx].nunique())})
    out = pd.DataFrame(rows)
    out.loc[len(out)] = {"class": "Total", **{c: int(out[c].sum()) for c in out.columns if c != "class"}}
    return out


def conflict_pairs(df: pd.DataFrame, top: int = 15) -> pd.DataFrame:
    """Which label combinations share identical feature vectors, and how many rows."""
    feats = D.feature_columns(df)
    tmp = pd.DataFrame({"k": _feat_key(df, feats).values, "y": df[D.LABEL].values})
    labs = tmp.groupby("k")["y"].agg(lambda s: " / ".join(sorted(set(s))))
    labs = labs[labs.str.contains(" / ")]
    if labs.empty:
        return pd.DataFrame(columns=["labels", "vectors", "rows"])
    rows = tmp[tmp["k"].isin(labs.index)].assign(labels=lambda t: t["k"].map(labs))
    return rows.groupby("labels").agg(vectors=("k", "nunique"), rows=("k", "size")) \
        .sort_values("rows", ascending=False).head(top).reset_index()


def most_repeated(df: pd.DataFrame, cls: str, top: int = 10) -> pd.DataFrame:
    """For one class: the most repeated feature vectors, with how many rows, distinct sources/destinations and
    the time span they cover (are the copies separate flows on the wire, or repeated records?)."""
    feats = D.feature_columns(df)
    sub = df[df[D.LABEL] == cls].assign(_k=_feat_key(df[df[D.LABEL] == cls], feats).values)
    g = sub.groupby("_k")
    out = pd.DataFrame({"rows": g.size()})
    for col in ("Src IP", "Dst IP"):
        if col in sub.columns:
            out[f"distinct_{col.replace(' ', '_')}"] = g[col].nunique()
    if "_t" in sub.columns:
        out["first_seen"] = g["_t"].min()
        out["last_seen"] = g["_t"].max()
        out["distinct_timestamps"] = g["_t"].nunique()
    if "_source" in sub.columns:
        out["files"] = g["_source"].agg(lambda s: ", ".join(sorted(set(map(str, s)))))
    # a few readable feature values for the top vectors
    show = [c for c in ("Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts", "TotLen Fwd Pkts", "Protocol", "Dst Port")
            if c in sub.columns]
    if show:
        out = out.join(g[show].first())
    return out.sort_values("rows", ascending=False).head(top).reset_index(drop=True)


def cross_partition_duplicates(df: pd.DataFrame, frac=(0.70, 0.15, 0.15)) -> pd.DataFrame:
    """Split the RAW data with the per-class chronological rule (no embargo) and report, per class, how many test
    rows have a feature vector that also occurs in training, and how many test rows the published rule would
    drop because of a conflicting label elsewhere in the data (labels from any partition)."""
    feats = D.feature_columns(df)
    part, _ = D.split_chrono_per_class(df, embargo=0, frac=frac)
    k = _feat_key(df, feats).values
    y = df[D.LABEL].values
    train_keys = set(k[part == 0])
    tmp = pd.DataFrame({"k": k, "y": y})
    conflict = tmp["k"].map(tmp.groupby("k")["y"].nunique()).values > 1
    rows = []
    for c in pd.unique(y):
        te = (part == 2) & (y == c)
        n = int(te.sum())
        in_train = int(np.isin(k[te], list(train_keys)).sum()) if n else 0
        rows.append({"class": c, "test_rows_raw": n, "test_rows_seen_in_train": in_train,
                     "share_seen_in_train": in_train / n if n else np.nan,
                     "test_rows_in_conflicting_groups": int((te & conflict).sum())})
    return pd.DataFrame(rows).sort_values("test_rows_raw", ascending=False).reset_index(drop=True)


def time_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Per class: first and last timestamp, number of distinct days, and the files it comes from."""
    if np.issubdtype(df["_t"].dtype, np.datetime64):
        t = df["_t"]
    else:  # stored as integer microseconds since the epoch
        t = pd.to_datetime(df["_t"].astype("int64"), unit="us")
    g = df.assign(_tt=t).groupby(D.LABEL)
    out = pd.DataFrame({"rows": g.size(), "first": g["_tt"].min(), "last": g["_tt"].max(),
                        "days": g["_tt"].agg(lambda s: s.dt.date.nunique())})
    if "_source" in df.columns:
        out["files"] = g["_source"].agg(lambda s: ", ".join(sorted(set(map(str, s)))))
    return out.sort_values("first")


def global_cut_feasibility(df: pd.DataFrame, frac=(0.70, 0.15, 0.15)) -> pd.DataFrame:
    """Class counts per partition if ALL flows were split by one global time cut (70/15/15 by row count).
    Classes with zero training rows cannot be learned in a multi-class task under such a cut."""
    order = np.lexsort((df["_row"].values, df["_t"].values))
    n = len(df)
    n_tr, n_va = int(round(frac[0] * n)), int(round(frac[1] * n))
    part = np.empty(n, dtype=np.int8)
    part[order[:n_tr]] = 0
    part[order[n_tr:n_tr + n_va]] = 1
    part[order[n_tr + n_va:]] = 2
    t = pd.crosstab(pd.Series(df[D.LABEL].values, name="class"), pd.Series(part, name="partition")).rename(columns={0: "train", 1: "validation", 2: "test"})
    for c in ("train", "validation", "test"):
        if c not in t:
            t[c] = 0
    t = t[["train", "validation", "test"]]
    t["learnable"] = np.where(t["train"] > 0, "yes", "NO (absent from training)")
    return t
