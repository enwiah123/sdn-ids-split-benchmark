"""Analyses reported in the paper.

H1  Random splits overstate macro-F1 relative to the per-class chronological split (S-C) and change the ranking.
H2  Under S-C, deep sequence models match XGBoost on macro-F1 (pre-specified margin +/-0.005) but differ on
    minority-class recall.
H3  Temporal context (L > 1) improves minority-class recall over single-flow input (L = 1).
H4  A GRU matches a Transformer in macro-F1 at lower CPU latency.
B6  Model rankings agree across datasets (Kendall's tau).

Units of analysis
  Statements across models (how many of the nine models were inflated, mean and range of gaps) are descriptive:
  the models are a fixed set of algorithms, not a sample.
  Statements about one model use (a) seed-level tests, conditional on each dataset's fixed split, and (b) intervals
  for the mean over seeds from a two-level bootstrap that also resamples the test flows (see stats.py and
  design_tables below).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import stats as S
from .models import TABULAR

NEURAL_ORDER = ["Transformer", "GRU", "BiLSTM", "CNN1D", "MLP", "CNN-LSTM", "LSTM"]


def _sc(res, ds, task, L, exp="B1", protocol="SC"):
    return res[(res.exp == exp) & (res.dataset == ds) & (res.task == task) & (res.L == L) & (res.protocol == protocol)]


# ------------------------------------------------------------------ H1: protocol audit (descriptive across models)
def protocol_gaps(res, comparisons=(("SR_raw", "SC"), ("SR_dedup", "SC"), ("SR_raw", "SR_dedup"))):
    """Per dataset x task x L: gap between protocols computed per model from seed-averaged scores, then summarised
    over the nine models (mean, min, max, models higher), plus model-seed pairs higher over the shared seeds,
    Kendall's tau between the S-R raw and S-C rankings and the best model under each."""
    b = res[res.exp == "B1"]
    rows = []
    for (ds, task, L), g in b.groupby(["dataset", "task", "L"]):
        m = g.groupby(["model", "protocol"])["macro_f1"].mean().unstack()
        shared = sorted(set.intersection(*[set(g[g.protocol == p].seed) for p in m.columns]))
        row = {"dataset": ds, "task": task, "L": int(L)}
        for a, c in comparisons:
            if a not in m or c not in m:
                continue
            dif = (m[a] - m[c]).dropna()
            pv = g[g.seed.isin(shared)].pivot_table(index=["model", "seed"], columns="protocol", values="macro_f1")
            pairs = (pv[a] - pv[c]).dropna()
            k = f"{a}-{c}"
            row.update({f"{k} mean": float(dif.mean()), f"{k} min": float(dif.min()), f"{k} max": float(dif.max()),
                        f"{k} models higher": f"{int((dif > 0).sum())}/{len(dif)}",
                        f"{k} pairs higher": f"{int((pairs > 0).sum())}/{len(pairs)}"})
        if {"SR_raw", "SC"} <= set(m.columns):
            row["kendall_tau_SRraw_SC"], row["kendall_p"] = S.kendall(m["SR_raw"], m["SC"])
            row["best_SR_raw"], row["best_SC"] = m["SR_raw"].idxmax(), m["SC"].idxmax()
        rows.append(row)
    return pd.DataFrame(rows)


def per_model_gaps(res, ds, task, L, a="SR_raw", c="SC"):
    g = res[(res.exp == "B1") & (res.dataset == ds) & (res.task == task) & (res.L == L)]
    m = g.groupby(["model", "protocol"])["macro_f1"].mean().unstack()
    return (m[a] - m[c]).rename("gap").reset_index()


def sensitivity_table(res, L=8):
    """Multi-class macro-F1 per model under S-R raw, S-R dedup, S-C, S-C raw and S-C label-free (S1)."""
    cmp = res[((res.exp == "S1") | ((res.exp == "B1") & res.protocol.isin(["SC", "SR_raw", "SR_dedup"])))
              & (res.task == "multi") & (res.L == L)]
    return cmp.pivot_table(index=["dataset", "model"], columns="protocol", values="macro_f1", aggfunc="mean")


# ------------------------------------------------------------------ H2 / H4: seed-level comparisons
def h2_seed_stats(res, ds, task, L, exp="B1", protocol="SC", ref="XGBoost", margins=(0.005, 0.02),
                  metric="macro_f1"):
    """Every model against the reference on the same split and seeds: mean seed-paired difference, seeds won,
    percentile bootstrap interval of the mean difference over seeds (seed-level interval), exact Wilcoxon and
    Mann-Whitney tests with Holm correction over the comparisons in this dataset x task x protocol, and TOST."""
    g = _sc(res, ds, task, L, exp, protocol)
    piv = g.pivot_table(index="seed", columns="model", values=metric)
    rows = []
    for m in piv.columns:
        if m == ref:
            continue
        both = piv[[m, ref]].dropna()
        x = (both[m] - both[ref]).values
        lo, hi = S.mean_ci_bootstrap(x)
        r = {"dataset": ds, "task": task, "protocol": protocol, "metric": metric, "model": m,
             "diff": float(x.mean()), "seeds_won": f"{int((x > 0).sum())}/{len(x)}",
             "seed_ci_low": lo, "seed_ci_high": hi, "p_wilcoxon": S.wilcoxon_exact(x),
             "p_mannwhitney": S.mannwhitney(both[m], both[ref]),
             "max_model_seed": float(both[m].max()), "min_ref_seed": float(both[ref].min())}
        for mg in margins:
            r[f"tost_p_{mg}"] = S.tost_paired(both[m], both[ref], mg)
        rows.append(r)
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    t["p_holm"] = S.holm(t["p_wilcoxon"].values)
    t["p_mannwhitney_holm"] = S.holm(t["p_mannwhitney"].values)
    def verdict(r):
        if r.p_holm < 0.05:
            return "better" if r["diff"] > 0 else "worse"
        if r[f"tost_p_{margins[-1]}"] < 0.05:
            return f"equivalent (±{margins[-1]})"
        return "inconclusive"
    t["verdict"] = t.apply(verdict, axis=1)
    return t.sort_values("diff", ascending=False).reset_index(drop=True)


def h4_gru_vs_transformer(res, ds, L, margins=(0.005, 0.02)):
    rows = []
    for task in ("multi", "binary"):
        piv = _sc(res, ds, task, L).pivot_table(index="seed", columns="model", values="macro_f1")
        if not {"GRU", "Transformer"} <= set(piv.columns):
            continue
        p = piv[["GRU", "Transformer"]].dropna()
        x = (p["GRU"] - p["Transformer"]).values
        r = {"task": task, "diff_GRU_minus_Transformer": float(x.mean()), "p_wilcoxon": S.wilcoxon_exact(x)}
        for mg in margins:
            r[f"tost_p_{mg}"] = S.tost_paired(p["GRU"], p["Transformer"], mg)
        rows.append(r)
    out = pd.DataFrame(rows)
    c = res[(res.exp == "B5") & (res.dataset == ds)].groupby("model")[["latency_ms_b1", "params", "flops_per_flow",
                                                                         "throughput_fps"]].mean()
    return out, c.loc[[m for m in ("GRU", "Transformer") if m in c.index]]


# ------------------------------------------------------------------ H3: temporal context
def h3_summary(res, ds):
    """Per neural model: validation-selected L, minority-class recall and macro-F1 at L = 1 and at the selected L,
    seeds on which recall improved; plus a descriptive summary over the neural models."""
    g = res[(res.exp == "B3") & (res.dataset == ds) & (res.task == "multi")]
    rows = []
    for m, gm in g.groupby("model"):
        bestL = int(gm.groupby("L")["val_macro_f1"].mean().idxmax())
        r = {"model": m, "neural": m not in TABULAR, "best_L_val": bestL,
             "minority_recall_L1": float(gm[gm.L == 1].minority_recall.mean()),
             "minority_recall_bestL": float(gm[gm.L == bestL].minority_recall.mean()),
             "macro_f1_L1": float(gm[gm.L == 1].macro_f1.mean()),
             "macro_f1_bestL": float(gm[gm.L == bestL].macro_f1.mean())}
        if bestL != 1:
            pv = gm[gm.L.isin([1, bestL])].pivot_table(index="seed", columns="L", values="minority_recall").dropna()
            d = pv[bestL] - pv[1]
            r["seeds_improved"] = f"{int((d > 0).sum())}/{len(d)}"
        rows.append(r)
    t = pd.DataFrame(rows)
    nn = t[t.neural]
    gain = nn.minority_recall_bestL - nn.minority_recall_L1
    summary = {"neural_models": len(nn), "models_improved": int((gain > 0).sum()), "mean_gain": float(gain.mean()),
               "min_gain": float(gain.min()), "max_gain": float(gain.max()),
               "macro_f1_mean_change": float((nn.macro_f1_bestL - nn.macro_f1_L1).mean())}
    return t, summary


# ------------------------------------------------------------------ B2 ranks and B6 agreement
def friedman_per_setting(res, datasets, L, tasks=("multi", "binary")):
    """Friedman test within each dataset x task, blocks = seeds (conditional on the data)."""
    from scipy import stats as sps
    rows = []
    for ds in datasets:
        for task in tasks:
            piv = _sc(res, ds, task, L).pivot_table(index="seed", columns="model", values="macro_f1").dropna(axis=1)
            if piv.shape[1] < 3:
                continue
            st = sps.friedmanchisquare(*[piv[c] for c in piv.columns])
            k, n = piv.shape[1], piv.shape[0]
            q = S.Q_ALPHA_005.get(k, np.nan)
            ranks = piv.rank(axis=1, ascending=False).mean().sort_values()
            rows.append({"dataset": ds, "task": task, "chi2": float(st.statistic), "p": float(st.pvalue),
                         "nemenyi_cd": float(q * np.sqrt(k * (k + 1) / (6 * n))),
                         "mean_ranks": "; ".join(f"{a} {v:.1f}" for a, v in ranks.items())})
    return pd.DataFrame(rows)


def rank_agreement(res, datasets, L_by_ds, task="multi"):
    means = {}
    for ds in datasets:
        g = _sc(res, ds, task, L_by_ds[ds])
        if len(g):
            means[ds] = g.groupby("model")["macro_f1"].mean()
    M = pd.DataFrame(means).dropna()
    pairs = []
    cols = list(M.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            tau, p = S.kendall(M[cols[i]], M[cols[j]])
            pairs.append({"dataset_a": cols[i], "dataset_b": cols[j], "kendall_tau": tau, "p": p,
                          "top_a": M[cols[i]].idxmax(), "top_b": M[cols[j]].idxmax()})
    return M, pd.DataFrame(pairs)


def best_neural_and_tabular(res, ds, task, L):
    g = _sc(res, ds, task, L).groupby("model")["val_macro_f1"].mean()
    neural = g[~g.index.isin(TABULAR)]
    tab = g[g.index.isin(TABULAR)]
    return (neural.idxmax() if len(neural) else None), (tab.idxmax() if len(tab) else None)




# ------------------------------------------------------------------ two-level bootstrap from saved predictions
def _files(out_dir, ds, keys):
    import os
    fs = [os.path.join(out_dir, ds, "preds", k + ".npz") for k in keys]
    return [f for f in fs if os.path.exists(f)]



class TestMeta:
    """Rebuilds, without features, the test partition exactly as experiments.prepare orders it, so each saved
    prediction row can be matched to its destination host and timestamp. Cached per (dataset, protocol, seed, L)."""

    def __init__(self, cfg):
        self.cfg, self._m = cfg, {}

    def get(self, ds, protocol, seed, L, task):
        from . import experiments as E
        from . import datasets as D
        key = (ds, protocol, seed if not protocol.startswith("SC") else -1, L, task)
        if key in self._m:
            return self._m[key]
        df, classes = E.load_clean(self.cfg, ds, dedup=E.DEDUP_MODE[protocol], verbose=False)
        frac = tuple(self.cfg["split_frac"])
        part = D.split_chrono_per_class(df, embargo=L, frac=frac)[0] if protocol.startswith("SC") \
            else D.split_random(df, seed, frac=frac)
        m = part == 2
        cmap = {c: i for i, c in enumerate(classes)}
        y = df.loc[m, D.LABEL].map(cmap).values.astype(np.int64)
        if task == "binary":
            y = (y != 0).astype(np.int64)
        dp = df.loc[m, [c for c in ("Src IP", "Dst IP") if c in df.columns] + ["_t", "_row", D.LABEL]]
        out = {"y": y, "host": D.context_keys(dp.reset_index(drop=True), self.cfg["window_key"]),
               "t": np.lexsort((dp["_row"].values, dp["_t"].values)).argsort()}   # time rank, ties by file row
        self._m[key] = out
        return out


def _runs(res, out_dir, meta, ds, task, L, exp, protocol, model, design, block, samplers):
    """(codes, sampler) per seed for one model, checking that saved labels match the rebuilt test partition."""
    g = res[(res.exp == exp) & (res.dataset == ds) & (res.task == task) & (res.L == L) & (res.protocol == protocol)
            & (res.model == model)].sort_values("seed")
    C = len(g["classes"].iloc[0]) if task == "multi" else 2
    out = []
    for _, r in g.iterrows():
        f = _files(out_dir, ds, [r["key"]])
        if not f:
            continue
        z = np.load(f[0]); y = z["y"].astype(np.int64)
        mt = meta.get(ds, protocol, int(r["seed"]), L, task)
        if len(mt["y"]) != len(y) or not np.array_equal(mt["y"], y):
            raise ValueError(f"rebuilt test partition does not match saved predictions: {r['key']}")
        sk = (ds, protocol, -1 if protocol.startswith("SC") else int(r["seed"]), L, task, design, block)
        if sk not in samplers:
            samplers[sk] = S.FlowSampler(y, design, host=mt["host"], t=mt["t"], block=block)
        out.append((y * C + z["P"].astype(np.float32).argmax(1), samplers[sk]))
    return out, C


def design_tables(res, out_dir, cfg, datasets, designs=("strat", "block", "host"), block=200, B=1000, L=8,
                  primary=None, log=print):
    """Intervals for the seed mean under each resampling design: model scores and protocol gaps (B1), comparisons
    with XGBoost (B1 S-C, and S1 S-C raw / S-C label-free), and minority-recall gains (B3). Returns dict of DataFrames."""
    from . import datasets as D
    from .models import TABULAR as _TAB
    meta, samplers = TestMeta(cfg), {}
    T = {k: [] for k in ("scores", "h1", "h2", "h3")}
    for ds in datasets:
        for task in ("multi", "binary"):
            g = res[(res.exp == "B1") & (res.dataset == ds) & (res.task == task) & (res.L == L)]
            models = sorted(g.model.unique())
            if not models:
                continue
            for dz in designs:
                # scores and H1 gaps
                for m in models:
                    rc, C = _runs(res, out_dir, meta, ds, task, L, "B1", "SC", m, dz, block, samplers)
                    lo, hi = S.design_mean_ci(rc, C, B=B)
                    T["scores"].append({"dataset": ds, "task": task, "design": dz, "model": m, "ci_low": lo, "ci_high": hi})
                    ra, _ = _runs(res, out_dir, meta, ds, task, L, "B1", "SR_raw", m, dz, block, samplers)
                    lo, hi = S.design_unpaired_diff_ci(ra, rc, C, B=B)
                    T["h1"].append({"dataset": ds, "task": task, "design": dz, "model": m, "ci_low": lo, "ci_high": hi})
                # H2 under SC (both tasks) and, multi-class only, SC_raw and SC_feat
                for exp, prot in [("B1", "SC")] + ([("S1", "SC_raw"), ("S1", "SC_feat")] if task == "multi" else []):
                    if res[(res.exp == exp) & (res.dataset == ds) & (res.protocol == prot)].empty:
                        continue
                    rx, C = _runs(res, out_dir, meta, ds, task, L, exp, prot, "XGBoost", dz, block, samplers)
                    for m in models:
                        if m == "XGBoost":
                            continue
                        rm, _ = _runs(res, out_dir, meta, ds, task, L, exp, prot, m, dz, block, samplers)
                        lo, hi = S.design_paired_diff_ci([a for a, _ in rm], [a for a, _ in rx], rm[0][1], C, B=B)
                        T["h2"].append({"dataset": ds, "task": task, "protocol": prot, "design": dz, "model": m,
                                        "ci_low": lo, "ci_high": hi})
                log(f"{ds} {task} {dz} done")
    # H3 (B3, primary dataset)
    ds = primary or datasets[0]
    g = res[(res.exp == "B3") & (res.dataset == ds) & (res.task == "multi")]
    for m, gm in g.groupby("model"):
        if m in _TAB:
            continue
        bestL = int(gm.groupby("L")["val_macro_f1"].mean().idxmax())
        cls = gm["classes"].iloc[0]; mask = np.array([c in D.MINORITY for c in cls])
        for dz in designs:
            ra, C = _runs(res, out_dir, meta, ds, "multi", bestL, "B3", "SC", m, dz, block, samplers)
            rb, _ = _runs(res, out_dir, meta, ds, "multi", 1, "B3", "SC", m, dz, block, samplers)
            lo, hi = S.design_unpaired_diff_ci(ra, rb, C, metric="minority_recall", classes_mask=mask, B=B)
            T["h3"].append({"model": m, "best_L": bestL, "design": dz, "ci_low": lo, "ci_high": hi})
    log("H3 done")
    return {k: pd.DataFrame(v) for k, v in T.items()}


def host_diagnostics(cfg, datasets, L=8):
    """Test-set host structure under S-C and, over the whole deduplicated data, how many destination hosts each class
    reaches (a class seen at a single host cannot appear on both sides of a host-disjoint split)."""
    from . import experiments as E
    from . import datasets as D
    meta = TestMeta(cfg); rows, per_class = [], []
    for ds in datasets:
        mt = meta.get(ds, "SC", 0, L, "multi")
        h = pd.Series(mt["host"]); vc = h.value_counts()
        s = pd.DataFrame({"h": mt["host"], "y": mt["y"]})
        nun = s.groupby("h")["y"].nunique()
        rows.append({"dataset": ds, "test_flows": len(h), "test_hosts": int(vc.size),
                     "share_top_host": float(vc.iloc[0] / len(h)), "share_top5_hosts": float(vc.iloc[:5].sum() / len(h)),
                     "hosts_with_>=100_flows": int((vc >= 100).sum()),
                     "share_flows_single_class_hosts": float(s["h"].isin(nun[nun == 1].index).mean())})
        df, classes = E.load_clean(cfg, ds, dedup=True, verbose=False)
        keys = D.context_keys(df, cfg["window_key"])
        t = pd.DataFrame({"h": keys, "c": df[D.LABEL].values})
        for c, gc in t.groupby("c"):
            vc2 = gc["h"].value_counts()
            per_class.append({"dataset": ds, "class": c, "flows": len(gc), "dest_hosts": int(vc2.size),
                              "share_top_host": float(vc2.iloc[0] / len(gc))})
    return pd.DataFrame(rows), pd.DataFrame(per_class)
