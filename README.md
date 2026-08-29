# C5: Broad Obesity-3 Thermogenic Gene-Pair Ranking

CrunchDAO Hackathon Submission, Broad Obesity Challenge (Phase 3)

This model predicts twelve thermogenic signature z-scores for 4,474,413 candidate gene pairs and ranks them by `FinalAggScore`, the mean of the three largest predicted z-scores. It's a fixed, rank-normalized blend of two components: a parameter-free mean-delta estimator (CPA) and a small trained residual-network ensemble (ThermoNet). Each component is computed per single gene first, then composed across the pair.

Peer review ranked this submission #2. Final leaderboard evaluation is still pending.

See [REPORT.md](REPORT.md) for the full method description, design rationale, and data provenance.

---

## Method Overview

### CPA: the mean-delta estimator

For each of the 122 experimentally perturbed genes, CPA's delta is the observed mean expression shift, knockout cells minus control cells, computed over the top-5,000 highly variable genes and projected onto each of the twelve thermogenic signature gene sets. Genes that were never perturbed get a proxy delta instead: a cosine-similarity-weighted average of the five most similar perturbed genes, with similarity measured in the pretrained scGPT gene-embedding space. There's no fitting involved beyond that average, so with only 122 labelled genes to work with, there isn't much room for the component to overfit.

### ThermoNet: the trained ensemble

ThermoNet is an ensemble of eight small symmetric residual networks that map a pair of scGPT gene embeddings to twelve z-scores, trained to reproduce the experimentally measured single- and double-gene scores. If a test pair reduces to one of the 133 experimentally measured single genes, its score gets overridden with that gene's exact measured z-scores instead of a network prediction.

### Combining two genes into a pair

The two components handle pairs differently. CPA's mean-delta is purely additive in delta-expression space, with no synergy term. ThermoNet adds a learned residual on top of the additive sum of its own single-gene predictions, so any pairwise synergy in the blend comes only from that residual.

### Blend

Each component produces twelve per-signature scores, rank-normalised independently to [0, 1] across all 4,474,413 candidate pairs, then combined as a fixed weighted sum with weights chosen to maximise held-out double-gene Spearman correlation:

| Component | Weight |
|-----------|--------|
| CPA | 0.54 |
| ThermoNet | 0.46 |

`FinalAggScore` is the mean of the three largest of the twelve blended z-scores per pair.

### A component we dropped

We also tried a third signal: the embedding shift from in-silico removing a gene's token from a different pretrained single-cell transformer (Geneformer) and re-encoding the cell. It didn't hold up. Across 20 probe genes, only one showed a silencing effect distinguishable from cell-to-cell noise, and a paired bootstrap CI on the resulting eval-Spearman change included zero. Rather than ship it at some small, unvalidated weight, we dropped it and redistributed its weight proportionally to CPA and ThermoNet.

---

## Validation

`evaluate()` checks predictions against the labelled TF150 and TF15_MOI2 perturbations and reports a Spearman breakdown by gene count. The single-gene `FinalAggScore` number looks great, but that's because single-gene pairs get overridden with ground truth through the memorization lookup, so it's inflated by design. The double-gene `FinalAggScore` is the fairer read since there's no ground-truth leakage there, though only about 123 labelled doubles exist, and ThermoNet's synergy residual was itself fit on those same pairs.

REPORT.md states the caveat directly: the reported double-gene Spearman is optimistic for the truly novel gene pairs that dominate the roughly 4.47M-pair test set, since the only held-out doubles available come from a small 15-gene combinatorial screen that also supervised the residual term.

---

## Repository Structure

```
.
├── main.py                    # Inference entrypoint: train() / infer() / evaluate()
├── REPORT.md                  # Method description, design rationale, data & resources
├── requirements.txt           # Python dependencies
└── resources/                 # Pre-baked weights (inference reads ONLY from here)
    ├── thermo_c3.pt                # ThermogenicScoreNet ensemble checkpoint (78 MB);
    │                                #   bundles its own scGPT embedding matrix and the
    │                                #   measured single-gene scores used for the override
    ├── cpa_raw_sig_scores.npy      # Per-gene CPA mean-delta signature scores
    ├── cpa_sig_stats.json          # CPA normalisation statistics
    └── test_gene_order.json        # Row-order index mapping genes → CPA scores
```

Offline precomputation (`01_train_cpa.py`, `02_precompute_cpa.py`) produced `cpa_raw_sig_scores.npy` on HPC and isn't included in this submission bundle; see [Train](#train) below.

---

## Data

Requires the Broad Obesity Phase 3 dataset (not included, available from CrunchDAO), placed in `data/`:

| File | Used by | Description |
|------|---------|-------------|
| `predict_perturbations_3.parquet` | `infer()` | The 4,474,413 candidate gene pairs |
| `TF150_ThermoScores_perturbation.csv` | `evaluate()` | Single-gene TF knockout screen z-scores (122 usable genes) |
| `TF15_MOI2_ThermoScores_perturbation.csv` | `evaluate()` | 15-gene combinatorial screen (singles + doubles) |
| `TF150.h5ad` | offline CPA training only | Raw scRNA-seq matrix (control vs. knockout cells) |
| `thermogenic_signatures.csv` | offline CPA training only | The twelve thermogenic signature gene sets |

Inference needs only `predict_perturbations_3.parquet`; every model weight is pre-baked into `resources/`.

No external single-cell datasets, regulatory or protein-interaction networks, or literature-derived gene sets were used beyond the competition-provided files and the pretrained scGPT checkpoint.

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Usage

### Inference (the submission path)

```bash
python main.py --data_dir data --model_dir resources
```

Writes `prediction.parquet` with 15 columns (`GenePairID`, the twelve signature scores, `FinalAggScore`, `Rank`) over all candidate pairs.

Or from Python:

```python
from main import infer

pred_df = infer(
    data_directory_path       = "data/",
    model_directory_path      = "resources/",
    prediction_directory_path = "predictions/",
)
```

### Evaluate

```bash
python main.py --evaluate --data_dir data --model_dir resources
```

Prints per-signature Spearman correlations plus the single-gene (memorized) vs. double-gene (fair) `FinalAggScore` breakdown.

### Train

`train()` is a no-op at submission time; precomputation ran offline on HPC:

```bash
python C5/scripts/01_train_cpa.py
python C5/scripts/02_precompute_cpa.py
cp C3/resources/thermo_c3.pt C5/resources/thermo_c3.pt   # ThermoNet reused from C3 unchanged
```

(These scripts are not included in this submission bundle, which ships only the resulting `resources/`.)

---

## CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--data_dir` | `data` | Competition data directory |
| `--model_dir` | `resources` | Self-contained weights directory |
| `--pred_dir` | `predictions` | Where `prediction.parquet` is written |
| `--evaluate` | off | Run `evaluate()` instead of `infer()` |

---

## Key Design Choices

We went with a parameter-free mean-delta estimator instead of a learned generative model. A compositional autoencoder was the original plan, but it wasn't available in the current toolchain, so a direct average of observed knockout-versus-control deltas reconstructs the training signal with zero extra fitted parameters and no added overfitting risk on 122 labelled genes.

For genes that were never perturbed, we proxy the delta using a cosine-similarity kNN with k=5, on the assumption that genes with similar global expression context (in scGPT embedding space) produce similar perturbation responses.

The blend itself is a fixed rank-normalized combination, with weights tuned on held-out double-gene Spearman. The two components sit on different raw scales, and rank normalization keeps a simple linear blend well-behaved.

We dropped the Geneformer silencing signal because it didn't clear the bar: only 1 of 20 probe genes showed a silencing effect above noise, and the bootstrap CI on the resulting Spearman change included zero. The rule we followed throughout was that a component needs a validated benefit before it earns a place in the blend.

Finally, measured singles get a ground-truth override rather than a prediction, since a real measurement beats a guess. That does inflate the single-gene eval number, which is exactly why the double-gene-only Spearman is the number to trust.
