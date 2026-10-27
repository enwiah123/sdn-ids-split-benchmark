"""Publication figures for Paper 2. Every figure is saved as vector PDF + 600-dpi PNG.

Sizes follow common journal columns: single column 3.5 in, double column 7.16 in (IEEE / Elsevier).
Colour: a colour-vision-deficiency-validated categorical order; sequential blue for magnitudes;
identity is never carried by colour alone (markers / direct labels / legends)."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import numpy as np                                  # noqa: E402
import pandas as pd                                 # noqa: E402

from .models import FAMILY, FAMILY_ORDER            # noqa: E402

SINGLE, DOUBLE = 3.5, 7.16
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
BLUES = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
PROTO_STYLE = {"SR_raw": ("Random split, raw", SERIES[1], "s"),
               "SR_dedup": ("Random split, deduplicated", SERIES[2], "^"),
               "SC": ("Per-class chronological (S-C)", SERIES[0], "o")}


def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7, "lines.linewidth": 1.6,
        "lines.markersize": 4.5, "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 110,
    })


def save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        p = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(p, **kw)
        paths.append(p)
    return paths


def family_color(model):
    return SERIES[FAMILY_ORDER.index(FAMILY.get(model, FAMILY_ORDER[0]))]


def family_marker(model):
    return MARKERS[FAMILY_ORDER.index(FAMILY.get(model, FAMILY_ORDER[0]))]


# ------------------------------------------------------------------ Fig. 1 dataset profile
def fig_class_distribution(profiles: dict, out_dir, name="fig1_class_distribution"):
    """profiles: {dataset: profile DataFrame from datasets.dataset_profile}. Log-scale horizontal bars."""
    style()
    n = len(profiles)
    fig, axes = plt.subplots(1, n, figsize=(DOUBLE if n > 1 else SINGLE, 2.4), squeeze=False)
    for ax, (ds, prof) in zip(axes[0], profiles.items()):
        p = prof[prof["Class"] != "Total"].iloc[::-1]
        y = np.arange(len(p))
        ax.barh(y + 0.2, p["Loaded flows"].clip(lower=0.8), height=0.38, color=SERIES[1], label="Loaded")
        ax.barh(y - 0.2, p["After deduplication"].clip(lower=0.8), height=0.38, color=SERIES[0],
                label="After deduplication")
        ax.set_xscale("log")
        ax.set_yticks(y, p["Class"])
        ax.set_title(ds, loc="left", fontweight="bold")
        ax.set_xlabel("Flows (log scale)")
        ax.grid(axis="y", visible=False)
        for yi, v, v0 in zip(y, p["After deduplication"], p["Loaded flows"]):
            ax.text(max(v, 1) * 1.15, yi - 0.2, f"{int(v):,}", va="center", fontsize=6, color=INK2)
            if v0 > 0 and (v0 - v) / v0 > 0.05:          # show the loaded count where dedup. removed > 5 %
                ax.text(max(v0, 1) * 1.15, yi + 0.2, f"{int(v0):,}", va="center", fontsize=6, color=INK2)
    axes[0][0].legend(loc="lower right")
    fig.tight_layout()
    return save(fig, out_dir, name)


# ------------------------------------------------------------------ Fig. 2 protocol inflation (B1)
def fig_protocol_inflation(b1: pd.DataFrame, out_dir, dataset, L, name="fig2_protocol_inflation"):
    """Dot plot: macro-F1 of every model under the three protocols, per task. Mean +/- SD over seeds."""
    style()
    g = b1[(b1.dataset == dataset) & (b1.L == L)]
    tasks = [t for t in ("multi", "binary") if t in set(g.task)]
    fig, axes = plt.subplots(1, len(tasks), figsize=(DOUBLE, 2.9), squeeze=False, sharey=True)
    order = g[(g.protocol == "SC") & (g.task == tasks[0])].groupby("model")["macro_f1"].mean().sort_values().index
    for ax, task in zip(axes[0], tasks):
        s = g[g.task == task].groupby(["model", "protocol"])["macro_f1"].agg(["mean", "std"]).reset_index()
        for i, m in enumerate(order):
            row = s[s.model == m]
            if len(row) > 1:
                ax.plot([row["mean"].min(), row["mean"].max()], [i, i], color=GRID, lw=2.2, zorder=1)
        for k, (proto, (lab, col, mk)) in enumerate(PROTO_STYLE.items()):
            r = s[s.protocol == proto].set_index("model").reindex(order)
            sd = r["std"].fillna(0)
            xerr = np.vstack([np.minimum(sd, r["mean"]), np.minimum(sd, 1 - r["mean"])])   # never beyond [0, 1]
            ax.errorbar(r["mean"], np.arange(len(order)) + (k - 1) * 0.12, xerr=xerr, fmt=mk, color=col,
                        mec="white", mew=0.6, ms=5, elinewidth=0.8, capsize=0, label=lab, zorder=3)
        ax.set_yticks(np.arange(len(order)), order)
        ax.set_xlabel("Test macro-F1")
        ax.set_title({"multi": "Multi-class", "binary": "Binary (normal vs attack)"}[task], loc="left",
                     fontweight="bold")
        ax.grid(axis="y", visible=False)
        lo = np.nanmin((s["mean"] - s["std"].fillna(0)).clip(lower=0).values)
        hi = np.nanmax((s["mean"] + s["std"].fillna(0)).clip(upper=1).values)
        pad = 0.06 * (hi - lo if hi > lo else 0.01)
        ax.set_xlim(lo - pad, hi + pad)       # keep markers near 1.0 fully visible
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, handletextpad=0.3,
               columnspacing=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=2.0)
    return save(fig, out_dir, f"{name}_{dataset}_L{L}")


# ------------------------------------------------------------------ Fig. 3 critical-difference diagram
def fig_cd_diagram(avg_rank: dict, cd: float, out_dir, title="", name="fig3_cd_diagram"):
    """Demsar (2006) critical-difference diagram; thick bars join models not significantly different."""
    style()
    items = sorted(avg_rank.items(), key=lambda kv: kv[1])
    k = len(items)
    fig, ax = plt.subplots(figsize=(DOUBLE, 0.55 + 0.21 * k))
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(0, 1.14)              # head-room so the CD bar and its label clear the title
    ax.axis("off")
    y0 = 0.86
    ax.plot([1, k], [y0, y0], color=INK, lw=0.9)
    for r in range(1, k + 1):
        ax.plot([r, r], [y0, y0 + 0.03], color=INK, lw=0.9)
        ax.text(r, y0 + 0.05, str(r), ha="center", va="bottom", fontsize=7)
    # CD bar
    ax.plot([1, 1 + cd], [0.99, 0.99], color=INK, lw=1.4)
    ax.text(1 + cd / 2, 1.0, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=7)
    half = (k + 1) // 2
    step = 0.62 / max(half, 1)
    for i, (m, r) in enumerate(items):
        left = i < half
        j = i if left else k - 1 - i
        yy = y0 - 0.12 - j * step
        xe = 0.55 if left else k + 0.45
        ax.plot([r, r, xe], [y0, yy, yy], color=INK2, lw=0.8)
        ax.text(xe + (-0.04 if left else 0.04), yy, f"{m} ({r:.2f})", ha="right" if left else "left",
                va="center", fontsize=7.5)
    # cliques: maximal groups whose rank span < CD
    ranks = [r for _, r in items]
    cliques = []
    for i in range(k):
        j = i
        while j + 1 < k and ranks[j + 1] - ranks[i] < cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in cliques):
            cliques.append((i, j))
    for n, (a, b) in enumerate(cliques):
        yy = y0 - 0.05 - n * 0.035
        ax.plot([ranks[a] - 0.03, ranks[b] + 0.03], [yy, yy], color=SERIES[0], lw=3, solid_capstyle="round")
    if title:
        ax.set_title(title, loc="left", fontweight="bold")
    return save(fig, out_dir, name)


# ------------------------------------------------------------------ Fig. 4 per-class recall heatmap
def fig_per_class_heatmap(mat: pd.DataFrame, n_test: dict, out_dir, title="", name="fig4_per_class_recall"):
    """mat: rows = models, columns = classes, values = mean test recall."""
    style()
    fig, ax = plt.subplots(figsize=(DOUBLE * 0.82, 0.35 + 0.27 * len(mat)))
    im = ax.imshow(mat.values, cmap=BLUES, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(mat.shape[1]), [f"{c}\n(n={n_test.get(c, 0):,})" for c in mat.columns], fontsize=6.5)
    ax.set_yticks(range(mat.shape[0]), mat.index)
    ax.grid(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color="white" if v > 0.6 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Recall (detection rate)")
    cb.outline.set_visible(False)
    ax.spines[:].set_visible(False)
    if title:
        ax.set_title(title, loc="left", fontweight="bold")
    fig.tight_layout()
    return save(fig, out_dir, name)


# ------------------------------------------------------------------ Fig. 5 temporal context (B3)
def fig_context_length(b3: pd.DataFrame, out_dir, dataset, name="fig5_context_length"):
    """Small multiples, one panel per model: macro-F1 and minority-class recall against window length L."""
    style()
    g = b3[(b3.dataset == dataset) & (b3.task == "multi")]
    models = [m for m in ["RF", "XGBoost", "MLP", "CNN1D", "LSTM", "GRU", "BiLSTM", "CNN-LSTM", "Transformer"]
              if m in set(g.model)]
    ncol = 3
    nrow = int(np.ceil(len(models) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(DOUBLE, 1.75 * nrow), sharex=True, sharey=True, squeeze=False)
    for ax in axes.flat[len(models):]:
        ax.set_visible(False)
    for ax, m in zip(axes.flat, models):
        s = g[g.model == m].groupby("L")[["macro_f1", "minority_recall"]].agg(["mean", "std"])
        Ls = s.index.values
        for k, (col, lab) in enumerate((("macro_f1", "Macro-F1"), ("minority_recall", "Minority-class recall"))):
            ax.errorbar(Ls, s[(col, "mean")], yerr=s[(col, "std")], color=SERIES[k], marker=MARKERS[k], ms=3.5,
                        elinewidth=0.7, capsize=0, label=lab)
        ax.set_xscale("log", base=2)
        ax.set_xticks(Ls, [str(int(x)) for x in Ls])
        ax.set_title(m, loc="left", fontweight="bold")
        ax.set_ylim(-0.02, 1.02)
    for ax in axes[-1]:
        ax.set_xlabel("Window length L (flows)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Test score")
    axes[0][0].legend(loc="lower left")
    fig.tight_layout()
    return save(fig, out_dir, f"{name}_{dataset}")


# ------------------------------------------------------------------ Fig. 6 imbalance trade-off (B4)
STRAT_LABEL = {"none": "None", "weighted_ce": "Weighted CE", "focal": "Focal", "logit_adjusted": "Logit-adj.",
               "balanced_sampling": "Bal. sampling", "smote": "SMOTE"}


STRAT_ORDER = ["none", "weighted_ce", "focal", "logit_adjusted", "balanced_sampling", "smote"]
STRAT_STYLE = {"none": (INK2, "s"), "weighted_ce": (SERIES[0], "o"), "focal": (SERIES[1], "^"),
               "logit_adjusted": (SERIES[2], "D"), "balanced_sampling": (SERIES[6], "v"), "smote": (SERIES[3], "P")}


def fig_imbalance_tradeoff(b4: pd.DataFrame, out_dir, dataset, name="fig6_imbalance_tradeoff"):
    """Per model: minority-class recall (benefit) against false-positive rate (cost) for each strategy.
    Each strategy has its own colour AND marker (identity never by colour alone) and one shared legend,
    so nearly coincident points stay readable. For trees, 'weighted CE' is reported as class weighting."""
    style()
    g = b4[(b4.dataset == dataset) & (b4.task == "multi")]
    models = list(dict.fromkeys(g.model))
    fig, axes = plt.subplots(1, len(models), figsize=(DOUBLE, 2.5), squeeze=False, sharey=True)
    seen = {}
    for ax, m in zip(axes[0], models):
        gm = g[g.model == m]
        tree = m in ("RF", "XGBoost")
        if tree:
            gm = gm[~gm.imbalance.isin(["focal", "logit_adjusted", "balanced_sampling"])]
        s = gm.groupby("imbalance")[["fpr", "minority_recall"]].agg(["mean", "std"])
        for strat in [x for x in STRAT_ORDER if x in s.index]:
            row = s.loc[strat]
            col, mk = STRAT_STYLE[strat]
            lab = "Class weighting (trees) / weighted CE" if strat == "weighted_ce" else STRAT_LABEL.get(strat, strat)
            h = ax.errorbar(100 * row[("fpr", "mean")], row[("minority_recall", "mean")],
                            xerr=100 * row[("fpr", "std")], yerr=row[("minority_recall", "std")],
                            fmt=mk, color=col, ms=5.5 if strat != "none" else 5, elinewidth=0.6, capsize=0,
                            mec="white", mew=0.5, alpha=0.95, zorder=4 if strat == "none" else 3)
            seen.setdefault(lab, h)
        ax.set_title(m, loc="left", fontweight="bold")
        ax.set_xlabel("False-positive rate (%)")
        ax.margins(x=0.2, y=0.12)
    axes[0][0].set_ylabel("Minority-class recall")
    fig.legend(list(seen.values()), list(seen.keys()), loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=min(6, len(seen)), handletextpad=0.2, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return save(fig, out_dir, f"{name}_{dataset}")


# ------------------------------------------------------------------ Fig. 7 accuracy-cost Pareto (B5)
# label positions (offset in points, horizontal alignment) chosen so that no two labels overlap on InSDN
PARETO_LABELS = {"XGBoost": (8, 6, "left"), "Transformer": (8, 6, "left"), "CNN1D": (6, 7, "left"),
                 "BiLSTM": (6, 8, "left"), "GRU": (6, -13, "left"), "CNN-LSTM": (-6, -14, "right"),
                 "LSTM": (6, -14, "left"), "MLP": (6, 8, "left"), "RF": (-6, 10, "right")}


def fig_pareto(b5: pd.DataFrame, out_dir, dataset, name="fig7_pareto_latency", labels=PARETO_LABELS):
    style()
    g = b5[b5.dataset == dataset].groupby("model")[["macro_f1", "latency_ms_b1"]].mean()
    fig, ax = plt.subplots(figsize=(DOUBLE, 4.2))
    for fam in FAMILY_ORDER:
        sub = g[[FAMILY.get(m) == fam for m in g.index]]
        if len(sub):
            ax.scatter(sub["latency_ms_b1"], sub["macro_f1"], s=55, color=SERIES[FAMILY_ORDER.index(fam)],
                       marker=MARKERS[FAMILY_ORDER.index(fam)], edgecolor="white", linewidth=0.6, label=fam, zorder=3)
    for m, r in g.iterrows():
        dx, dy, ha = labels.get(m, (5, 4, "left"))
        ax.annotate(m, (r["latency_ms_b1"], r["macro_f1"]), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, fontsize=8.5, color=INK)
    # Pareto front: lower latency and higher macro-F1
    pts = g.sort_values("latency_ms_b1")
    front, best = [], -np.inf
    for m, r in pts.iterrows():
        if r["macro_f1"] > best:
            front.append((r["latency_ms_b1"], r["macro_f1"]))
            best = r["macro_f1"]
    if len(front) > 1:
        fx, fy = zip(*front)
        ax.step(fx, fy, where="post", color=INK2, lw=1.1, ls="--", zorder=2, label="Pareto front")
    ax.set_xscale("log")
    ax.set_xlabel("CPU inference latency per flow window, batch 1 (ms, log scale)", fontsize=9)
    ax.set_ylabel("Test macro-F1", fontsize=9)
    ax.tick_params(labelsize=8.5)
    ax.legend(loc="lower right", bbox_to_anchor=(0.98, 0.04), fontsize=8)
    fig.tight_layout()
    return save(fig, out_dir, f"{name}_{dataset}")


# ------------------------------------------------------------------ Fig. 8 confusion matrices
def fig_confusions(mats: dict, classes: list, out_dir, name="fig8_confusion"):
    """mats: {title: confusion matrix (counts)}; rows normalised to recall."""
    style()
    n = len(mats)
    fig, axes = plt.subplots(1, n, figsize=(DOUBLE, DOUBLE / n + 0.5), squeeze=False)
    for ax, (title, cm) in zip(axes[0], mats.items()):
        cm = np.asarray(cm, float)
        norm = np.divide(cm, cm.sum(1, keepdims=True), out=np.full_like(cm, np.nan), where=cm.sum(1, keepdims=True) > 0)
        ax.imshow(norm, cmap=BLUES, vmin=0, vmax=1)
        for i in range(len(classes)):
            for j in range(len(classes)):
                v = norm[i, j]
                if not np.isnan(v) and (v >= 0.005 or i == j):
                    ax.text(j, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=5.5,
                            color="white" if v > 0.6 else INK)
        ax.set_xticks(range(len(classes)), classes, rotation=60, ha="right", fontsize=6)
        ax.set_yticks(range(len(classes)), classes if ax is axes[0][0] else [""] * len(classes), fontsize=6)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(False)
        ax.spines[:].set_visible(False)
        ax.set_xlabel("Predicted")
    axes[0][0].set_ylabel("True (row-normalised, %)")
    fig.tight_layout()
    return save(fig, out_dir, name)


# ------------------------------------------------------------------ Fig. 9 ranking agreement (B6)
def fig_rank_agreement(mean_scores: pd.DataFrame, out_dir, name="fig9_rank_agreement"):
    """mean_scores: rows = models, columns = datasets (macro-F1). Slope chart of ranks across datasets."""
    style()
    ranks = mean_scores.rank(ascending=False)
    cols = list(mean_scores.columns)
    fig, ax = plt.subplots(figsize=(SINGLE * (1 + 0.35 * (len(cols) - 2)), 2.8))
    for m in ranks.index:
        ax.plot(range(len(cols)), ranks.loc[m, cols], color=family_color(m), marker=family_marker(m), ms=4, lw=1.2)
        ax.text(-0.08, ranks.loc[m, cols[0]], m, ha="right", va="center", fontsize=6.5)
        ax.text(len(cols) - 1 + 0.08, ranks.loc[m, cols[-1]], m, ha="left", va="center", fontsize=6.5)
    ax.set_xticks(range(len(cols)), cols)
    ax.set_xlim(-0.9, len(cols) - 0.1)
    ax.invert_yaxis()
    ax.set_yticks(range(1, len(ranks) + 1))
    ax.set_ylabel("Rank by test macro-F1 (1 = best)")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return save(fig, out_dir, name)


# ------------------------------------------------------------------ Fig. S1 training curves
def fig_learning_curves(histories: dict, out_dir, name="figS1_learning_curves"):
    """histories: {model: [{'epoch','train_loss','val_loss'}, ...]} (one seed)."""
    style()
    models = list(histories)
    ncol = 4
    nrow = int(np.ceil(len(models) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(DOUBLE, 1.6 * nrow), squeeze=False)
    for ax in axes.flat[len(models):]:
        ax.set_visible(False)
    for ax, m in zip(axes.flat, models):
        h = pd.DataFrame(histories[m])
        ax.plot(h["epoch"], h["train_loss"], color=SERIES[0], label="Train")
        ax.plot(h["epoch"], h["val_loss"], color=SERIES[1], ls="--", label="Validation")
        ax.set_title(m, loc="left", fontweight="bold")
        ax.set_xlabel("Epoch")
    axes[0][0].set_ylabel("Cross-entropy")
    axes[0][0].legend()
    fig.tight_layout()
    return save(fig, out_dir, name)
