# GRID: Guided Rule Induction for Discovery

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT">
  <img src="https://img.shields.io/badge/version-2.2.0-orange" alt="Version 2.2.0">
  <img src="https://img.shields.io/badge/paper-DMKD-red" alt="Paper: DMKD">
</p>

**GRID** is a subgroup discovery framework that integrates seven synergistic enhancements into a single, theoretically grounded pipeline. It discovers interpretable rule-based descriptions of data subsets whose target distribution deviates meaningfully from the population baseline—with formal guarantees on diversity, statistical validity, and scalability.

> **Paper**: *GRID: Guided Rule Induction for Discovery — Conditional Discretization, Lift-Weighted Submodular Selection, and Distribution-Free Validation for Subgroup Discovery*
> Submitted to Data Mining and Knowledge Discovery (DMKD).

---

## Table of Contents

- [Highlights](#highlights)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [The Seven Phases](#the-seven-phases)
- [Benchmark Results](#benchmark-results)
- [Reproducing the Paper](#reproducing-the-paper)
- [Configuration Guide](#configuration-guide)
- [Use Cases](#use-cases)
- [Citation](#citation)
- [License](#license)

---

## Highlights

| Feature | Description |
|---|---|
| **Highest Lift** | 2.26 ± 1.88 across 20 benchmark datasets — surpassing CN2 (+11.9%), PRIM (+51.7%), SD-Map (+37.0%), DT-Rules (+30.6%), RIPPER (+67.4%) |
| **Best CVR** | 0.73 ± 0.41 — highest out-of-sample conformal validity rate among all methods |
| **Best DC/Lift** | 3.27 — most efficient interpretability-to-quality ratio |
| **Competitive DI** | 0.66 ± 0.37 — formal diversity via lift-weighted submodular maximisation |
| **Scalable** | Near-linear scaling to N = 10⁶ (91.9s single-worker, <50 MB memory) |
| **Expressive** | DNF rules, negated selectors, ratio features — captures patterns invisible to conjunctions-only methods |
| **Statistically Rigorous** | Three-tier validation: BH-FDR control + permutation calibration + split-conformal coverage |
| **Deployable** | Auto-tuned hyperparameters, natural-language explanations (EN/PT), SHAP integration |

---

## Architecture

GRID implements a seven-phase pipeline where each component is (a) motivated by a specific shortcoming of prior work, (b) grounded in a formal guarantee, and (c) validated via per-component ablation.

```
Dataset
  │
  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Phase 1: Conditional Discretization                                │
│  Global WoE binning → context-aware re-binning per subgroup         │
│  Theorem: IG(conditional) ≥ IG(global)                              │
├─────────────────────────────────────────────────────────────────────┤
│  Phase 2: Multi-Width Beam Search                                   │
│  Depth-aware beam decay (β₁=3 → β_d=1)                             │
│  Negative-correlation penalty (λ=0.8, calibrated by ablation)       │
│  Bidirectional search + DNF expansion                               │
├─────────────────────────────────────────────────────────────────────┤
│  Phase 3: Three-Tier Statistical Validation                         │
│  Tier 1: Benjamini–Hochberg FDR (always active)                     │
│  Tier 2: Permutation-calibrated p-values (N<200 or dependence)      │
│  Tier 3: Split-conformal coverage (deployment mode)                 │
├─────────────────────────────────────────────────────────────────────┤
│  Phase 4: Lift-Weighted Submodular Diversity                        │
│  Facility location objective with α_lift = 0.6                      │
│  (1-1/e)-approximation guarantee + MinHash pre-dedup                │
│  Hard budget cap: |R| ≤ K                                           │
├─────────────────────────────────────────────────────────────────────┤
│  Phase 5: Scalability                                               │
│  Stratified sampling (adaptive n_s schedule)                        │
│  Parallel seed exploration (joblib)                                 │
├─────────────────────────────────────────────────────────────────────┤
│  Phase 6: Extended Expressiveness                                   │
│  DNF rules (disjunctive normal form)                                │
│  Negated selectors (X ≠ v)                                          │
│  Ratio features (X_j / X_k)                                        │
├─────────────────────────────────────────────────────────────────────┤
│  Phase 7: Deployment Enhancements                                   │
│  Meta-learned hyperparameter tuning (2-layer MLP)                   │
│  Natural-language explanations (EN/PT)                              │
│  SHAP–subgroup integration (Δϕ_j)                                   │
└─────────────────────────────────────────────────────────────────────┘
  │
  ▼
Discovered Subgroups (rules + metrics + explanations)
```

---

## Installation

### From PyPI (recommended)

```bash
pip install grid-discovery
```

### From source

```bash
git clone https://github.com/valteresj/grid-discovery.git
cd grid-discovery
pip install -e .
```

### Dependencies

- Python ≥ 3.10
- NumPy, Pandas, Scikit-learn
- SciPy (for statistical tests)
- Joblib (for parallelism)
- OptionalBinning (for WoE discretization)

---

## Quick Start

### Basic Usage

```python
from grid_discovery import GRIDClassifier
import pandas as pd

# Load your data
X = pd.read_csv("data.csv")
y = X.pop("target")

# Initialize and fit
grid = GRIDClassifier(
    target_label="default",
    max_depth=4,
    max_rules=30,
    auto_tune=True,
)
grid.fit(X, y)

# View discovered subgroups
summary = grid.summary()
print(f"Rules found: {summary['n_rules']}")
print(f"Average Lift: {summary['avg_lift']:.2f}")
print(f"Diversity Index: {summary['di']:.2f}")

# Natural-language explanations
for rule in grid.rules_:
    print(rule.to_natural_language())
```

### With Train/Calibration Split (Conformal Validation)

```python
from sklearn.model_selection import train_test_split

X_train, X_cal, y_train, y_cal = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

grid = GRIDClassifier(
    target_label="default",
    auto_tune=True,
    n_workers=-1,       # use all CPU cores
    random_state=42,
)
grid.fit(X_train, y_train)

# Conformal validity on held-out data
for rule in grid.rules_:
    idx = rule.evaluate(grid.discretizer_.transform(X_cal))
    if len(idx) > 0:
        cal_lift = y_cal.iloc[idx].mean() / y_cal.mean()
        print(f"{rule}: cal_lift = {cal_lift:.2f}")
```

### Credit Risk Example (German Credit)

```python
import openml

# Load German Credit dataset
ds = openml.datasets.get_dataset(31)
X, y_raw, _, _ = ds.get_data(target="class", dataset_format="dataframe")
y = (y_raw.astype(str) == "2").astype(int)  # 2 = bad credit

grid = GRIDClassifier(
    target_label="bad_credit",
    max_depth=3,
    lift_upper=1.5,
    max_bins=5,
    enable_conditional=True,
    enable_negation=True,
    enable_ratios=True,
    enable_dnf=True,
    redundancy_mode="submodular",
    auto_tune=True,
)
grid.fit(X, y)

# Top rules with natural-language explanations
s = grid.summary()
print(f"Top rule lift: {s['max_lift']:.2f}")
print(f"Rules: {s['n_rules']} conjunctive + {s['n_dnf_rules']} DNF")
print(f"SVR: {s['svr']:.2f}, DI: {s['di']:.2f}")
```

Expected output:
```
Top rule lift: 2.63
Rules: 3 conjunctive + 2 DNF
SVR: 0.97, DI: 0.85

Rule 1 (Negation):
  "Applicants with loan duration between 58 and 71 months, unskilled
   employment, and known (non-missing) savings status have a default
   rate of 86.8%, which is 2.63× the population average of 33%.
   This segment covers 3% of records.
   p-value (BH-corrected): <0.001.
   Recommendation: enhanced due diligence and risk-adjusted pricing."
```

---

## API Reference

### `GRIDClassifier`

#### Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `target_label` | `str` | required | Label of the positive class |
| `max_depth` | `int` | `4` | Maximum rule depth (number of selectors) |
| `max_bins` | `int` | `5` | Maximum bins for discretization |
| `min_coverage` | `float` | `0.03` | Minimum subgroup coverage |
| `lift_upper` | `float` | `1.2` | Minimum lift threshold for positive rules |
| `lift_lower` | `float` | `0.8` | Maximum lift threshold for negative rules |
| `alpha` | `float` | `0.05` | FDR significance level |
| `discretization` | `str` | `"quantile"` | Discretization strategy |
| `enable_conditional` | `bool` | `True` | Enable conditional re-binning (Phase 1) |
| `enable_ratios` | `bool` | `True` | Enable ratio features X_j/X_k (Phase 6) |
| `beam_width` | `int` | `3` | Multi-width beam search width β (Phase 2) |
| `corr_penalty_lambda` | `float` | `0.8` | Correlation penalty strength λ (Phase 2) |
| `enable_negation` | `bool` | `True` | Enable negated selectors X ≠ v (Phase 6) |
| `enable_bidirectional` | `bool` | `True` | Enable bottom-up pruning (Phase 2) |
| `redundancy_mode` | `str` | `"submodular"` | Selection mode: `"submodular"` or `"greedy"` |
| `max_rules` | `int` | `30` | Hard budget cap K (Phase 4) |
| `enable_dnf` | `bool` | `True` | Enable DNF rule merging (Phase 6) |
| `dnf_max_width` | `int` | `2` | Maximum DNF width w (Phase 6) |
| `dnf_lift_delta` | `float` | `0.3` | Lift similarity threshold for DNF merging |
| `auto_tune` | `bool` | `True` | Enable meta-learned hyperparameters (Phase 7) |
| `n_workers` | `int` | `-1` | Parallel workers (-1 = all CPUs) |
| `verbose` | `int` | `0` | Verbosity level (0, 1, 2) |
| `random_state` | `int` | `42` | Random seed for reproducibility |

#### Methods

| Method | Returns | Description |
|---|---|---|
| `fit(X, y)` | `self` | Discover subgroups from data |
| `summary()` | `dict` | Aggregated metrics (lift, WRAcc, DI, SVR, etc.) |
| `predict(X)` | `np.ndarray` | Predict subgroup membership |
| `explain(lang="en")` | `list[str]` | Natural-language rule explanations |

#### Attributes (after fit)

| Attribute | Type | Description |
|---|---|---|
| `rules_` | `list[Rule]` | Discovered conjunctive rules |
| `dnf_rules_` | `list[DNFRule]` | Discovered DNF rules |
| `discretizer_` | `Discretizer` | Fitted discretizer object |
| `base_rate_` | `float` | Population base rate π₀ |
| `meta_features_` | `dict` | Dataset meta-features used by auto-tuner |

---

## The Seven Phases

### Phase 1: Conditional Discretization

Global Weight-of-Evidence binning followed by context-aware re-binning within each subgroup. The key insight is that bin boundaries optimal for the whole population may be suboptimal for a specific subgroup.

**Guarantee**: Conditional binning never decreases Information Gain within the subgroup (Theorem 1).

**Lazy evaluation**: Re-binning triggers only when (a) |S| < 0.5|S_parent| or (b) any bin receives >80% of S. At most ~15 re-binnings per root-to-leaf path.

### Phase 2: Multi-Width Beam Search

Features are ranked by adjusted Information Gain with a negative-correlation penalty:

```
IG_adj(X_j) = IG(X_j; Y | S) × ∏_k (1 - |ρ_jk|^0.8)
```

The penalty at λ=0.8 ensures:
- Highly correlated features (|ρ| > 0.9): strongly penalised (~8% retained)
- Moderately correlated (|ρ| = 0.5): partially penalised (~43% retained)
- Weakly correlated (|ρ| = 0.3): minimally penalised (~62% retained)

Beam width decays with depth: β_d = max(1, β - max(0, d-1)).

### Phase 3: Three-Tier Statistical Validation

| Tier | Method | When Active | Guarantee |
|---|---|---|---|
| 1 | Benjamini–Hochberg FDR | Always | FDR ≤ α |
| 2 | Permutation p-values | N < 200 or dependence | Exact Type I control |
| 3 | Split-conformal coverage | Deployment mode | P(Lift_test ≥ λ⁺) ≥ 1-α |

A rule is retained if it passes Tier 1 AND at least one of Tier 2 or Tier 3.

### Phase 4: Lift-Weighted Submodular Diversity

The facility location objective with lift weighting:

```
f(R') = Σ_i max_{r ∈ R'} w(r) · 1[i ∈ ext(r)]
w(r) = 0.6 · Lift(r) + 0.4
```

**Guarantee**: Greedy achieves ≥ (1 - 1/e) ≈ 63% of the optimal solution (Theorem 5).

**MinHash pre-deduplication**: 128 hash functions, 8 bands × 16 rows. Near-duplicates (Jaccard > 0.3) are removed before selection.

### Phase 5: Scalability

Adaptive sampling schedule:
| Dataset Size | Sample Size |
|---|---|
| N ≤ 5,000 | Full data |
| N ≤ 50,000 | 5,000 |
| N ≤ 500,000 | 10,000 |
| N > 500,000 | 20,000 |

Parallel seed exploration via joblib reduces wall-clock time proportional to min(p, n_workers).

### Phase 6: Extended Expressiveness

| Feature | Example | Advantage |
|---|---|---|
| **DNF rules** | (A ∧ B) ∨ (C ∧ D) | Doubles coverage without sacrificing lift |
| **Negated selectors** | savings ≠ "unknown" | Captures exclusion patterns across categories |
| **Ratio features** | income / debt ∈ [2, 5) | Single selector captures diagonal boundaries |

**Ablation impact**: Removing all three causes the largest quality drop: −21.2% lift.

### Phase 7: Deployment Enhancements

- **Meta-learned auto-tuning**: 2-layer MLP maps dataset meta-features to optimal (d_max, η, λ⁺, L_max)
- **Natural-language reports**: Template-based explanations in English and Portuguese
- **SHAP integration**: Within-subgroup SHAP differential Δϕ_j identifies features more predictive inside the subgroup

---

## Benchmark Results

### Main Results (Table 2 — 20 datasets × 5-fold CV)

| Method | Lift ↑ | WRAcc ↑ | |R| ↓ | d ↓ | SVR ↑ | DC ↓ | DI ↑ | CVR ↑ |
|---|---|---|---|---|---|---|---|---|
| CN2 | 2.02±1.81 | 0.016±0.015 | 6.7±4.8 | 3.5±2.1 | 0.76±0.42 | 10.5±6.2 | **0.76±0.42** | 0.62±0.41 |
| RIPPER | 1.35±1.17 | **0.140±0.351** | 10.4±29.5 | 2.0±1.6 | 0.76±0.42 | 5.9±4.8 | 0.63±0.41 | 0.50±0.46 |
| PRIM | 1.49±1.35 | 0.007±0.007 | 4.8±3.6 | 2.8±1.8 | 0.76±0.42 | 14.0±9.1 | **0.76±0.42** | – |
| SD-Map | 1.65±1.15 | 0.052±0.051 | 20.0±0.0 | **1.9±0.6** | **1.00±0.00** | **5.7±1.9** | 0.66±0.29 | 0.70±0.42 |
| DT-Rules | 1.73±1.14 | 0.025±0.026 | **3.7±2.5** | 2.7±1.5 | 0.76±0.42 | 8.0±4.6 | **0.76±0.42** | 0.65±0.41 |
| **GRID** | **2.26±1.88** | 0.037±0.037 | 14.2±10.4 | 2.5±1.4 | 0.76±0.42 | 7.4±4.3 | 0.66±0.37 | **0.73±0.41** |

### Per-Component Ablation

| Removed Component | ΔLift | Key Observation |
|---|---|---|
| DNF / Negation / Ratio | **−21.2%** | Largest quality drop; depth drops from 2.46 to 1.91 |
| Multi-Beam (β=1) | −5.8% | Narrower beam misses deeper refinements |
| Submodular → Greedy | +10.6% lift | But DI collapses from 0.66 to 0.27 (−59%) |
| Conditional Disc. | −0.5% | Marginal on aggregate; impactful on heterogeneous datasets |
| BH → Bonferroni | −0.0% lift | But SVR drops from 0.765 to 0.730 |
| Auto-tune | −0.8% | Pure usability enhancement |
| Scalability | −0.0% | Pure efficiency enhancement |

### Scalability (Adult Dataset)

| N | Time (1 worker) | Time (8 workers) | Memory |
|---|---|---|---|
| 10,000 | 2.8 ± 0.2s | 5.9 ± 0.7s | 3 ± 1 MB |
| 50,000 | 6.9 ± 0.1s | 9.3 ± 0.1s | 5 ± 1 MB |
| 100,000 | 12.7 ± 0.1s | 13.6 ± 0.2s | 4 ± 1 MB |
| 500,000 | 45.5 ± 1.5s | 38.1 ± 0.7s | 25 ± 2 MB |
| 1,000,000 | 91.9 ± 1.1s | 73.8 ± 1.0s | 47 ± 2 MB |

Runtime scales as approximately O(N^0.76) due to the adaptive sampling schedule.

---

## Reproducing the Paper

### Requirements

```bash
pip install grid-discovery openml pysubgroup wittgenstein prim scikit-learn pandas numpy
```

### Run All Experiments

```bash
# Table 2: Main results (20 datasets × 5-fold CV)
python generate_table3.py

# Quick smoke test (1 dataset, 1 fold)
python generate_table3.py --smoke

# Fast mode (5 datasets × 3 folds)
python generate_table3.py --fast

# Resume from checkpoint
python generate_table3.py --resume
```

### Individual Experiment Scripts

| Script | Output |
|---|---|
| `generate_table2.py` | Table 2 — Aggregated performance |
| `generate_table3.py` | Table 3 — Per-component ablation |
| `generate_table4.py` | Table 4 — Sensitivity analysis |
| `generate_table2b.py` | Scalability analysis |
| `generate_table2c.py` | Stability and interpretability |

### Hardware Used in Paper

- CPU: Intel i7-12700K
- RAM: 32 GB
- OS: Ubuntu 22.04
- Python: 3.12

---

## Configuration Guide

### Recommended Defaults

```python
# Standard configuration (works well for most datasets)
grid = GRIDClassifier(
    target_label="your_target",
    auto_tune=True,       # let meta-learner choose hyperparameters
    n_workers=-1,         # use all CPUs
    random_state=42,
)
```

### For Small Datasets (N < 500)

```python
grid = GRIDClassifier(
    target_label="your_target",
    max_depth=3,
    beam_width=2,
    min_coverage=0.05,    # higher min coverage for small N
    max_bins=4,
    enable_dnf=False,     # fewer candidates to avoid overfitting
)
```

### For Large Datasets (N > 100,000)

```python
grid = GRIDClassifier(
    target_label="your_target",
    beam_width=2,         # narrower beam for speed
    max_depth=3,          # shallower search
    n_workers=-1,         # parallelise
    auto_tune=True,       # auto-adapts sampling
)
```

### For Maximum Diversity

```python
grid = GRIDClassifier(
    target_label="your_target",
    redundancy_mode="submodular",
    max_rules=20,
    # Lower alpha_lift → more diversity, less lift
)
```

### For Maximum Lift (at the cost of diversity)

```python
grid = GRIDClassifier(
    target_label="your_target",
    redundancy_mode="greedy",  # or submodular with high alpha_lift
    beam_width=5,
    max_depth=5,
    enable_dnf=True,
    enable_ratios=True,
    enable_negation=True,
)
```

---

## Use Cases

### Credit Risk

Identify borrower segments with elevated default rates for risk-adjusted pricing, enhanced due diligence, or portfolio monitoring.

```python
# Example output:
# "Applicants with loan duration 58-71 months, unskilled employment,
#  and known savings status: default rate 86.8% (2.63× baseline)"
```

### Clinical Research

Characterise patient subpopulations with divergent treatment responses for precision medicine and clinical trial design.

### Marketing Analytics

Find audience segments with high conversion potential for targeted campaigns and resource allocation.

### Fraud Detection

Discover transaction patterns with elevated fraud rates for rule-based alert systems.

---

## Citation

If you use GRID in your research, please cite:

```bibtex
@article{junior2025grid,
  title={{GRID}: Guided Rule Induction for Discovery---Conditional 
         Discretization, Lift-Weighted Submodular Selection, and 
         Distribution-Free Validation for Subgroup Discovery},
  author={Junior, Valter E. S.},
  journal={Data Mining and Knowledge Discovery},
  year={2025},
  note={Submitted}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contributing

Contributions are welcome. Please open an issue first to discuss proposed changes.

### Development Setup

```bash
git clone https://github.com/valteresj/grid-discovery.git
cd grid-discovery
pip install -e ".[dev]"
pytest tests/
```

---

<p align="center">
  <b>GRID</b> — Discovering interpretable subgroups with formal guarantees.<br>
  Built in São Paulo, Brazil.
</p>
