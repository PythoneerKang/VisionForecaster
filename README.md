````markdown
# Cross-Scale GBDT for Financial Network Link Prediction

> LightGBM Pipeline · O(1) Graph Feature Extraction · Strict Temporal CV · F2-Aligned Calibration

A robust Gradient Boosted Decision Tree (GBDT) pipeline designed to predict the formation of edges in a target-scale financial network (e.g., $w_{70}$) using topological anomalies extracted from shorter source-scale networks (e.g., $w_{35}$, $w_{70}$, $w_{120}$). Given historical IQD adjacency matrices, the model outputs the **probability of an edge forming** between stock pairs at $t+1$.

---

## Pipeline Overview

```text
Raw PKL Files (IQD Matrices)
        │
        │  Epsilon thresholding + GICS Reordering
        ▼
Source & Target Adjacency Matrices (457 × 457)
        │
        │  O(S·N³) Graph Precomputation + O(F·P) Pair Extraction
        ▼
Pairwise Feature Matrix (P × F)
        │
        ├──────────────────────────────────────┐
        ▼                                      ▼
┌─────────────────────────────┐    ┌─────────────────────────────┐
│  Strict Temporal CV Split   │    │   Ablation Study Masking    │
│  Train (Rolling + Tail)     │    │   Full vs Pure Cross-Scale  │
│  Gap (Zero-Overlap)         │    └─────────────────────────────┘
│  Calibration                │                  │
│  Test (Blind)               │                  │
└─────────────────────────────┘                  │
        │                                       │
        ▼                                       ▼
LightGBM (Logloss Early Stopping) ──────► Masked Features
        │
        ▼
Isotonic/Platt Calibration (on Calib Set)
        │
        ▼
F2 Threshold Optimization (on Calib Set)
        │
        ▼
Final Evaluation (AUC, AP, F2, SHAP on Test Set)
```

For the full interactive diagram, open [`docs/index.html`](docs/index.html) in a browser.

---

## Hyperparameters & Feature Configuration

| Hyperparameter         | Value                      |
|------------------------|----------------------------|
| `target_w`             | e.g., 70                   |
| `source_ws`            | e.g., [35, 70, 120]        |
| `history_lags`         | e.g., 4                    |
| `first_lag`            | $(2 \times w_{target}) // 5$ (Default) / 0 (State-Var) |
| `neg_ratio`            | 1 (Train subsampling)      |
| `num_leaves`           | 63                         |
| `min_child_samples`    | 20                         |
| `learning_rate`        | 0.02                       |
| `n_estimators`         | 2000                       |
| `early_stopping`       | 50 iterations (Logloss)    |
| `is_unbalance`         | True                       |
| `reg_alpha`            | 0.0                        |

---

## Key Components

### 1. O(1) Graph Feature Extraction (v5 Perf Fix)
Instead of massive $\mathcal{O}(P \cdot N^2)$ matrix multiplications per pair, the pipeline precomputes global graph metrics ($A^2$, $A^3$, Clustering Coefficients) in $\mathcal{O}(S \cdot N^3)$ time and $\mathcal{O}(N^2)$ space. Pair-level features are then extracted via strict NumPy indexing in $\mathcal{O}(F \cdot P)$ time. 

Features are divided into four categories:
1.  **Sector Prior (1):** Binary same-sector indicator.
2.  **Target-Scale Attributes (6 per lag):** Edge, degree, common neighbors, Jaccard.
3.  **Source-Scale Standard (7 per lag per source):** Edge, 3-hop bridge, degree, Jaccard.
4.  **Source-Scale Topological "Neck" (6 per lag per source):** Neckness ($1 - C_i$), Cross-sector degree, Clustering boundary diff, Remote bridge capacity.

### 2. Remote Bridge Capacity (Network Theory)
A continuous edge-level metric measuring how heavily node $i$ relies on the pair $(i,j)$ to route information to distant clusters, bypassing local cliques. 
$$B_{ij} = A^3_{ij} \cdot (1 - A_{ij}) \cdot \Big(1 - \mathbb{I}(A^2_{ij} > 0)\Big)$$
$$\text{Capacity}_{ij} = \frac{B_{ij}}{d_i + \epsilon}$$
*(Note: No summation variable $m$ is used; this evaluates a single structural coordinate).*

### 3. Strict Temporal CV & The Gap
Data is split using a purged rolling window:
*   **Train (Rolling):** Fixed max window (e.g., 120 weeks) of past data.
*   **Train (Tail-Eval):** The final 20% of the train window, used *exclusively* for Logloss early stopping to prevent look-ahead bias.
*   **Gap:** A `first_lag` exclusion zone ensuring source-scale windows do not temporally overlap with target-scale labels (can be set to 0 if IQDs are treated as instantaneous state variables).
*   **Calibration:** Used to fit Platt/Isotonic scaling and hunt for the optimal F2 threshold.
*   **Test:** Completely blind evaluation.

### 4. F2-Aligned Thresholding
The model is trained to optimize smooth Logloss gradients. Post-training, the raw probabilities are calibrated, and an exhaustive search over the Precision-Recall curve on the **Calibration Set** finds the threshold that maximizes the $F_2$ score. This threshold is frozen and applied to the Test Set.

---

## Feature Budget

Instead of parameters, GBDT complexity is defined by feature space. For 3 source scales and 4 history lags:

| Feature Category              | Count                            |
|-------------------------------|----------------------------------|
| Sector Prior                  | 1                                |
| Target-Scale Attrs            | $6 \times 4 = 24$                |
| Source-Scale Standard         | $7 \times 3 \times 4 = 84$       |
| Source-Scale Neck             | $6 \times 3 \times 4 = 72$       |
| **Total Features ($F$)**      | **181**                          |

---

## Data Flow

| Stage | Tensor/Array Shape | Complexity |
|---|---|---|
| Graph Precomp ($A^2, A^3$) | `(457, 457)` per scale | Time: $\mathcal{O}(S \cdot N^3)$, Space: $\mathcal{O}(N^2)$ |
| Pair Extraction (Train) | `(subsampled_P, 181)` | Time: $\mathcal{O}(F \cdot P_{sub})$, Space: $\mathcal{O}(F \cdot P_{sub})$ |
| Pair Extraction (Eval/Test) | `(~101,275, 181)` | Time: $\mathcal{O}(F \cdot P)$, Space: $\mathcal{O}(F \cdot P)$ |
| LGBM Early Stopping | Evaluated on Tail | Monitors `binary_logloss` (first metric only) |
| Calibration Output | `(P_test,)` probabilities | Brier Score minimized on Calib set |

**Training objective:** Minimize `binary_logloss`. **Selection objective:** Maximize $F_2 = \frac{5 \cdot P \cdot R}{4 \cdot P + R}$ on the Calibration set.

---

## Interpretability

The following plots are generated automatically per fold:

| Plot | Description |
|---|---|
| `baseline_comparison_pr.png` | PR curves comparing Full model, Pure Cross-Scale, Marginal Prior, and Short-Scale Oracle |
| `feature_importance.png` | Top-20 LightGBM split gains |
| `shap_summary.png` | SHAP beeswarm plot showing feature impact on model output |
| `feature_snapshot_standardized.png` | Z-score heatmap of raw features grouped by $y=0$ and $y=1$ |
| `calibration_curve.png` | Reliability diagram of predicted probabilities on Test Set |
| `threshold_curves.png` | F1, F2, Precision, Recall vs. Threshold with the chosen F2 threshold marked |
| `data_timeline.png` | Visual breakdown of Rolling Train, Tail-Eval, Gap, Calib, and Test blocks |

---

## Ablation Study

The pipeline natively supports a `pure_cross_scale` ablation via substring-masked feature dropping. It dynamically removes all features starting with `w{target}_` or `deg{target}_`. This proves that target-scale historical edges act as noise/overfitting traps, and that pure multi-scale topological "neck" anomalies carry the true predictive signal.
````