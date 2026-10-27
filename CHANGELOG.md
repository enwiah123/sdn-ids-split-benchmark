# Changelog

## 1.0.0 (release for submission)

The experiments themselves (data preparation, tuning, all training runs) were not changed during revision. What changed
was the statistical analysis and its description. This file records those changes so that readers can see how the final
methods were reached.

### Final methods (this release)

* Across the nine models: descriptive counts, means and ranges only.
* Per model against XGBoost: mean seed-paired difference, seeds won, seed-level percentile interval, exact Wilcoxon with
  Holm correction (eight comparisons per dataset, task and protocol), unpaired Mann–Whitney with Holm correction, TOST
  at ±0.005 and ±0.02.
* Friedman test per dataset and task with seeds as blocks; Nemenyi critical difference; Kendall's τ across datasets.
* Intervals for seed means: two-level bootstrap that resamples seeds and blocks of 200 consecutive test flows within
  each class. Checks: single flows within each class, blocks of 50 and 1,000. Host-cluster resampling is documented as
  unusable on these datasets.
* Data audit, sensitivity arms S1 (S-C on raw data; S-C after label-free deduplication) and host-structure diagnostics.

### Superseded during internal review (not used for any result in the paper)

1. P-values that pooled model–seed pairs across the nine models, and Wilcoxon tests that treated the models as
   independent observations. Replaced by descriptive statements across models.
2. The Vargha–Delaney A12 effect size, which assumes independent samples. Replaced by seeds won.
3. A Friedman test that pooled 40 blocks (2 datasets × 2 tasks × 10 seeds). Replaced by one test per dataset and task.
4. A bootstrap that drew one seed per replicate. Its intervals described a single training run rather than the mean over
   seeds. Replaced by a two-level bootstrap of the mean.
5. A two-level bootstrap that resampled single test flows without regard to class. Rare classes (two U2R test flows in
   InSDN) dropped out of many resamples, which changed the set of classes that macro-F1 and minority-class recall were
   averaged over, and neighbouring flows with overlapping windows were treated as independent. Replaced by the
   class-preserving block bootstrap.
