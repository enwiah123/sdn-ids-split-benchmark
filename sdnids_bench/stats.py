"""Statistics used in the paper.

Seed-level tests (conditional on each dataset's fixed split)
  * exact Wilcoxon signed-rank test on seed-paired differences, Holm correction within each family
  * Mann-Whitney U test over seeds (does not rely on pairing; used because on InSDN the seeds share no
    training sample across models)
  * paired TOST equivalence test (margins in macro-F1 units)
  * percentile bootstrap interval of the mean seed-paired difference ("seed-level interval")
  * Friedman test with seeds as blocks, Nemenyi critical difference (Demsar, 2006); Kendall's tau

Intervals for means over seeds that also allow for sampling of the test set
  * two-level bootstrap: seeds resampled with replacement; test flows resampled with one of the designs in
    FlowSampler. The paper uses design="block" (blocks of 200 consecutive flows within each class); "strat"
    (single flows within each class) and other block lengths are reported as sensitivity checks; "host"
    (destination hosts as clusters) is provided to show why it is unusable on these datasets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# Nemenyi two-tailed q_alpha at alpha = 0.05 for k = 2..10 models (Demsar, 2006, Table 5a)
Q_ALPHA_005 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


def holm(pvals):
    p = np.asarray(pvals, float)
    p = np.where(np.isnan(p), 1.0, p)
    order, m = np.argsort(p), len(p)
    adj, running = np.empty_like(p), 0.0
    for r, i in enumerate(order):
        running = max(running, (m - r) * p[i])
        adj[i] = min(1.0, running)
    return adj


def wilcoxon_exact(x) -> float:
    """Exact two-sided Wilcoxon signed-rank p-value for paired differences (1.0 if all zero)."""
    from scipy import stats as sps
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0 or np.all(x == 0):
        return 1.0
    return float(sps.wilcoxon(x, method="exact").pvalue)


def mannwhitney(x, y) -> float:
    """Exact two-sided Mann-Whitney U test of two independent samples of seed scores."""
    return float(stats.mannwhitneyu(np.asarray(x, float), np.asarray(y, float), alternative="two-sided",
                                    method="exact").pvalue)


def tost_paired(x, y, margin):
    """Two one-sided paired t-tests; returns the TOST p-value (equivalent if p < 0.05)."""
    d = np.asarray(x, float) - np.asarray(y, float)
    n = len(d)
    if n < 3:
        return float("nan")
    sd = d.std(ddof=1)
    if sd == 0:
        return 0.0 if abs(d.mean()) < margin else 1.0
    se = sd / np.sqrt(n)
    p_lower = 1 - stats.t.cdf((d.mean() + margin) / se, n - 1)
    p_upper = stats.t.cdf((d.mean() - margin) / se, n - 1)
    return float(max(p_lower, p_upper))


def mean_ci_bootstrap(x, B=10000, seed=0, alpha=0.05):
    """Percentile bootstrap CI of the mean of x (x = per-model or per-seed values)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    m = rng.choice(x, (B, len(x)), replace=True).mean(1)
    return float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))


def friedman_nemenyi(scores: pd.DataFrame):
    """scores: rows = blocks (e.g. dataset x task x seed), columns = models (higher = better).
    Returns dict with Friedman statistic, p, average ranks (1 = best) and the Nemenyi CD."""
    s = scores.dropna(axis=0)
    k, n = s.shape[1], s.shape[0]
    ranks = s.rank(axis=1, ascending=False, method="average")
    avg = ranks.mean().sort_values()
    out = {"k": k, "n_blocks": n, "avg_rank": avg.to_dict()}
    if k >= 3 and n >= 2:
        fr = stats.friedmanchisquare(*[s[c].values for c in s.columns])
        out.update({"friedman_chi2": float(fr.statistic), "friedman_p": float(fr.pvalue)})
        # Iman-Davenport correction (less conservative)
        chi2 = fr.statistic
        denom = n * (k - 1) - chi2
        if denom > 0:
            ff = (n - 1) * chi2 / denom
            out["iman_davenport_F"] = float(ff)
            out["iman_davenport_p"] = float(1 - stats.f.cdf(ff, k - 1, (k - 1) * (n - 1)))
    if k in Q_ALPHA_005:
        out["nemenyi_cd"] = float(Q_ALPHA_005[k] * np.sqrt(k * (k + 1) / (6.0 * n)))
    return out


def kendall(a: pd.Series, b: pd.Series):
    common = a.index.intersection(b.index)
    t = stats.kendalltau(a[common], b[common])
    return float(t.statistic), float(t.pvalue)



# ---------------------------------------------------------------- two-level bootstrap of the seed mean
def _metric_from_cm(cm, metric, classes_mask=None):
    tp = np.diag(cm)
    support = cm.sum(1)
    if metric == "macro_f1":
        pred = cm.sum(0)
        prec = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
        rec = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
        f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
        return float(f1[support > 0].mean())
    if metric == "minority_recall":
        sel = np.asarray(classes_mask, bool) & (support > 0)
        return float((tp[sel] / support[sel]).mean()) if sel.any() else np.nan
    raise ValueError(metric)



class FlowSampler:
    def __init__(self, y, design="iid", host=None, t=None, block=200):
        self.n, self.design = len(y), design
        y = np.asarray(y)
        if design in ("strat", "block"):
            self.groups = []
            for c in np.unique(y):
                idx = np.flatnonzero(y == c)
                if design == "block":
                    if t is not None:
                        idx = idx[np.argsort(np.asarray(t)[idx], kind="mergesort")]
                    self.groups.append([idx[i:i + block] for i in range(0, len(idx), block)])
                else:
                    self.groups.append(idx)
        elif design == "host":
            codes, uniq = pd.factorize(pd.Series(host).astype(str))
            order = np.argsort(codes, kind="mergesort")
            bounds = np.r_[0, np.flatnonzero(np.diff(codes[order])) + 1, len(order)]
            self.clusters = [order[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]
        elif design != "iid":
            raise ValueError(design)

    def __call__(self, rng):
        n = self.n
        if self.design == "iid":
            return np.bincount(rng.integers(0, n, n), minlength=n)
        if self.design == "strat":
            pick = [g[rng.integers(0, len(g), len(g))] for g in self.groups]
        elif self.design == "block":
            pick = []
            for blocks in self.groups:
                for j in rng.integers(0, len(blocks), len(blocks)):
                    pick.append(blocks[j])
        else:
            G = len(self.clusters)
            pick = [self.clusters[j] for j in rng.integers(0, G, G)]
        return np.bincount(np.concatenate(pick), minlength=n)


def _cm(k, w, C):
    return np.bincount(k, weights=w, minlength=C * C).reshape(C, C)


def design_mean_ci(runs, C, metric="macro_f1", classes_mask=None, B=1000, seed=0, alpha=0.05):
    """runs: list of (codes, sampler) per seed. If every run shares one sampler object (same test set), one flow
    resample per replicate is applied to all seeds; otherwise each drawn seed is resampled with its own sampler."""
    rng = np.random.default_rng(seed)
    shared = all(r[1] is runs[0][1] for r in runs)
    S_ = len(runs); vals = np.empty(B)
    for b in range(B):
        w = runs[0][1](rng) if shared else None
        v = [_metric_from_cm(_cm(runs[s][0], w if shared else runs[s][1](rng), C), metric, classes_mask)
             for s in rng.integers(0, S_, S_)]
        vals[b] = np.nanmean(v)
    return float(np.nanquantile(vals, alpha / 2)), float(np.nanquantile(vals, 1 - alpha / 2))


def design_paired_diff_ci(codes_a, codes_b, sampler, C, metric="macro_f1", classes_mask=None, B=1000, seed=0,
                          alpha=0.05):
    """Mean over seeds of metric(a) - metric(b) on one shared test set; seeds and flows resampled jointly."""
    rng = np.random.default_rng(seed)
    S_ = len(codes_a); vals = np.empty(B)
    for b in range(B):
        w = sampler(rng)
        v = [_metric_from_cm(_cm(codes_a[s], w, C), metric, classes_mask)
             - _metric_from_cm(_cm(codes_b[s], w, C), metric, classes_mask) for s in rng.integers(0, S_, S_)]
        vals[b] = np.nanmean(v)
    return float(np.nanquantile(vals, alpha / 2)), float(np.nanquantile(vals, 1 - alpha / 2))


def design_unpaired_diff_ci(runs_a, runs_b, C, metric="macro_f1", classes_mask=None, B=1000, seed=0, alpha=0.05):
    """mean(a) - mean(b) for runs on different test sets; each side resampled on its own (see design_mean_ci)."""
    rng = np.random.default_rng(seed)
    def side(runs):
        shared = all(r[1] is runs[0][1] for r in runs)
        w = runs[0][1](rng) if shared else None
        return np.nanmean([_metric_from_cm(_cm(runs[s][0], w if shared else runs[s][1](rng), C), metric, classes_mask)
                           for s in rng.integers(0, len(runs), len(runs))])
    vals = np.array([side(runs_a) - side(runs_b) for _ in range(B)])
    return float(np.nanquantile(vals, alpha / 2)), float(np.nanquantile(vals, 1 - alpha / 2))
