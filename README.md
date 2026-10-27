# Random splits flatter deep sequence models: benchmark code

This is the code for the paper *Random splits flatter deep sequence models: a per-class chronological benchmark for
intrusion detection in software-defined networks* (submitted to the **).

The benchmark compares nine models (Random Forest, XGBoost, MLP, 1D-CNN, LSTM, GRU, BiLSTM, CNN-LSTM and a Transformer)
on InSDN and the corrected CICIDS2017 under three evaluation protocols:

* **S-R raw**: random 70/15/15 split of the records as published, duplicates kept;
* **S-R dedup**: the same random split after deduplication;
* **S-C**: per-class chronological split. Within each class, records are ordered by time and the first 70% go to
  training, the next 15% to validation and the last 15% to test, with an embargo of L records before each
  boundary. This is the main protocol. It orders time within classes only and is not a global time holdout.

Every model receives the same input: a causal window of the last L flows sent to the same destination host. The tabular
models receive the window flattened.

## Repository layout

```
sdnids_bench/          the package
  config.py            settings; notebook 00 writes config.json
  datasets.py          loading, label mapping, deduplication, splits, causal windows
  models.py, training.py   the nine models and their training loops
  experiments.py       run grids, tuning, result storage (resumable)
  audit.py             data-preparation audit (what deduplication removes, global-cut feasibility)
  analysis.py          the analyses reported in the paper (H1–H4, B6, S1) and the bootstrap tables
  stats.py             tests (exact Wilcoxon, Mann–Whitney, TOST, Friedman–Nemenyi, Kendall) and the two-level bootstrap
  figures.py, paper_figures.py, tables.py   figures and tables
notebooks/             run in order, 00 to 10
data/review_coding/    how each of the 86 studies in our systematic review evaluated its models (Table 1)
```

## Installation

```bash
conda env create -f environment.yml      # or: python -m venv .venv, then pip install -r requirements.txt
conda activate sdn-ids-benchmark
jupyter lab
```

The paper used Python 3.11 on Windows 10 with the library versions pinned in `requirements.txt` (CPU only, Intel
Core i7-7700HQ, 17 GB RAM).

## Data

The datasets are not redistributed here. Download them from their publishers:

| Dataset | Source | Files |
|---|---|---|
| InSDN (Elsayed, Le-Khac and Jurcut, 2020) | the InSDN page of University College Dublin | `Normal_data.csv`, `OVS.csv`, `metasploitable-2.csv` |
| CICIDS2017, corrected release (Engelen, Rimmer and Joosen, 2021) | Distrinet, KU Leuven | the daily CSV files |

Put each dataset in its own folder and set the two paths in notebook 00. Use the **corrected** CICIDS2017: the
original release has labelling and timestamp faults that matter for a chronological split.

Labels are mapped to Normal, DDoS, DoS, Probe, BFA, Web-Attack, BOTNET, U2R and Other. In CICIDS2017, flows marked as
attempted attacks are treated as Normal, and the infiltration port-scan flows are mapped to Probe. Normal CICIDS2017
traffic is sampled with a fixed probability so that the data fit in laptop memory (600,247 normal records kept);
every attack record is kept.

## Reproducing the paper

Run the notebooks in order. Every experiment notebook is resumable: each finished run is one line in
`results/<dataset>/results.jsonl`, with its test predictions in `results/<dataset>/preds/`, and is skipped if the
notebook is run again.

| Notebook | What it does |
|---|---|
| 00 Setup and configuration | dataset paths, run mode; records the environment |
| 01 Data preparation | loading, label mapping, deduplication, split checks |
| 02 Hyperparameter tuning | 20 Optuna trials per model, task and dataset (S-C validation macro-F1, L = 8, seed 0) |
| 03 B3 temporal context | window lengths L = 1, 4, 8, 16 on InSDN, five seeds; selects L on validation data |
| 04 B1/B2 protocol audit and main benchmark | three protocols × two tasks × L ∈ {1, 8}; S-C ten seeds, S-R five |
| 05 B4 class imbalance | six strategies × top-4 models × five seeds on InSDN |
| 06 B5 computational cost | CPU latency, throughput, size and FLOPs, three seeds |
| 07 S1 sensitivity runs | S-C on raw data (both datasets) and after label-free deduplication (CICIDS2017), ten seeds |
| 08 Data audit | what deduplication removes; why a global time split is not used (no training) |
| 09 Figures and tables | every figure and the descriptive tables (no training) |
| 10 Statistics and intervals | every statistical result in the paper (no training; the bootstrap takes 1–2 hours on a laptop CPU) |

Notebooks 02–07 train models and take days on a laptop CPU in total; a GPU shortens them considerably.

All numbers in the paper come from `"mode": "reduced"` (the default): ten seeds for S-C, five for the random-split
arms, B3 and B4, three for B5; 20 tuning trials; window lengths up to 16; CICIDS2017 training data capped at
150,001 windows by stratified sampling with the rarest classes kept whole. Validation and test data are never capped,
and the test data are never used for any choice. `"mode": "full"` (40 trials, L up to 32) is provided for anyone with
a GPU.

### Where each result comes from

Outputs go to `results/paper/` (figures in `figures/`, descriptive tables in `tables/`, statistics in `statistics/`).

| In the paper | Produced by | File(s) |
|---|---|---|
| Table 1 (evaluation design of the 86 reviewed studies) | coded from the review's S1 File | `data/review_coding/` |
| Table 3, Figure 1 (records before and after deduplication) | 01, 09 | `tables/table_dataset_profile.csv`, `figures/fig1_class_distribution.*` |
| Figure 2 (protocols) | 09 | `figures/fig_protocols.*` |
| Table 4 (protocol audit), H1 counts and ranges | 10 | `statistics/H1_protocol_gaps.csv`, `H1_per_model_gaps.csv` |
| Figure 3 (protocol inflation) | 09 | `figures/fig2_inflation.png` (panels `fig2_protocol_inflation_*_L8.*`) |
| Table 5 (deduplication sensitivity) | 07, 10 | `statistics/S1_sensitivity_macro_f1.csv` |
| Table 6 (main results, S-C) | 09 | `tables/table_main_*_multi.csv` |
| Section 4.3 comparisons with XGBoost (seeds won, Wilcoxon, Mann–Whitney, TOST) | 10 | `statistics/H2_seed_level.csv`, `H2_minority_recall_seed_level.csv` |
| Table 7 (mean ranks, Friedman, Nemenyi) | 10 | `statistics/friedman_per_setting.csv` |
| Figure 4 (per-class recall), Figure 5 (confusion matrices) | 09 | `figures/fig4_per_class.png`, `figures/fig8_confusion_InSDN.*` |
| Table 8, Figure 6 (window length, H3) | 09, 10 | `tables/table_context_InSDN.csv`, `statistics/H3_summary.csv`, `figures/fig5_context_length_InSDN.*` |
| Table 9, Figure 7 (class imbalance) | 09 | `tables/table_imbalance_InSDN.csv`, `figures/fig6_imbalance_tradeoff_InSDN.*` |
| Table 10, Figure 8 (cost), H4 | 09, 10 | `tables/table_cost_InSDN.csv`, `figures/fig7_pareto_latency_InSDN.*`, `statistics/H4_*.csv` |
| Figure 9, B6 (rank agreement) | 09, 10 | `figures/fig9_rank_agreement.*`, `statistics/B6_rank_agreement.csv` |
| All 95% intervals for seed means (Sections 4.2–4.5) | 10 | `statistics/intervals_scores.csv`, `intervals_h1.csv`, `intervals_h2.csv`, `intervals_h3.csv` |
| Tables B1–B3 | 10 | `statistics/table_B1_*.csv`, `table_B2_*.csv`, `table_B3_*.csv` |
| Host structure (Section 6) | 10 | `statistics/host_structure_test.csv`, `hosts_per_class.csv` |
| Data audit (Sections 3.3, 3.4, 4.1) | 08 | `results/audit/` |
| Table A1 (selected hyperparameters) | 09 | `tables/table_hyperparameters.csv` |
| Figure S1 (learning curves) | 09 | `figures/figS1_learning_curves.*` |

## What the code guarantees

* Identifiers (flow ID, addresses, source port, timestamp) are never model inputs. Destination addresses are used
  only to build windows and timestamps only to order flows.
* Deduplication happens before any split. The main rule removes exact duplicates and then every record whose
  feature vector also occurs with another label. The label-free alternative (S1) keeps the earliest copy of each
  feature vector with its own label. Both use the same tie rule as the split: time, then position in the source file.
* Windows are built inside each partition from flows at or before the current flow, so no window contains records of
  two partitions. Under S-C the embargo also leaves a gap of L records of each class at each boundary.
* The variance filter, signed log transform and standardisation are fitted on training data only.
* All tuning and model selection (window length, top-4 models for B4) use validation data only.

## Statistics

* Statements across the nine models (how many were inflated, means and ranges) are descriptive.
* Comparisons with XGBoost are made per model on a fixed split. We report the mean seed-paired difference, the number of
  seeds won, a percentile bootstrap interval over seeds, an exact Wilcoxon signed-rank test with Holm correction over the
  eight comparisons in each dataset, task and protocol, and TOST at ±0.005 (pre-specified) and ±0.02. On CICIDS2017 runs
  with the same seed share their training sample; on InSDN they share nothing, so an unpaired Mann–Whitney test is
  reported as well.
* Intervals for means over seeds use a two-level bootstrap: seeds are resampled with replacement and test flows are
  resampled in blocks of 200 consecutive flows within each class (`stats.FlowSampler(design="block")`). Every class keeps
  its test count, and neighbouring flows, whose windows overlap, stay together. Single-flow resampling within classes and
  blocks of 50 and 1,000 are reported as checks. Resampling destination hosts as clusters is available
  (`design="host"`), but it is not usable on these datasets because most attack classes were sent to one to four hosts.
* Every bootstrap uses a fixed random seed, so notebook 10 reproduces the published intervals exactly from the saved
  predictions.

## History of the analysis

The statistical analysis was revised several times during internal review before submission. The final methods are the
ones in this repository. Earlier versions are described in `CHANGELOG.md` so that the changes are on record.

## Citation

See `CITATION.cff`. Please also cite InSDN and the corrected CICIDS2017 release if you use the data.

## Licence

MIT (see `LICENSE`).
