# Broad Obesity Phase 3 — C5: CPA + ThermogenicScoreNet Ensemble
#
# Loads two sets of precomputed resources:
#   CPA:        cpa_raw_sig_scores.npy  — per-gene raw signature means from CPA decoder
#   ThermoNet:  thermo_c3.pt            — C3 ensemble weights (same as C3/main.py)
#
# Geneformer KO-silencing was dropped from the ensemble: 05_diagnose_geneformer_signal.py
# showed only 1/20 probe genes' silencing deltas exceed the bootstrap noise floor (no
# usable per-gene signal at the embedding-extraction stage), and a paired bootstrap CI
# confirmed the resulting eval-Spearman swing (0.6592->0.6107 on n=123) is not
# statistically distinguishable from sampling noise (95% CI on the difference includes 0).
# See Models.md §4.4 for the full investigation.
#
# Pair scoring is fully vectorized (numpy) — all 4.47M pairs scored in < 5 min.
#
# Platform convention:
#   train(data_directory_path, model_directory_path)          → no-op
#   infer(data_directory_path, model_directory_path, ...)     → pd.DataFrame
#   evaluate(data_directory_path, model_directory_path, ...)  → dict

import argparse
import json
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Config — all constants in one place
# ─────────────────────────────────────────────────────────────────────────────

class Config:
    SIG_COLS = [
        "REACTOME_AMPK_inhibits_chREBP_R-HSA-163680",
        "REACTOME_Integration_of_energy_metabolism_R-HSA-163685",
        "REACTOME_Mitochondrial_biogenesis_R-HSA-1592230",
        "emont.hAd6",
        "lit.thermogenic",
        "C2.WP_THERMOGENESIS",
        "C5.GOBP_ADAPTIVE_THERMOGENESIS",
        "C5.GOBP_BROWN_FAT_CELL_DIFFERENTIATION",
        "C5.GOBP_NEGATIVE_REGULATION_OF_COLD_INDUCED_THERMOGENESIS",
        "C5.GOBP_POSITIVE_REGULATION_OF_COLD_INDUCED_THERMOGENESIS",
        "yi.Thermogenesis_Village",
        "nonucp1.all",
    ]
    N_SIGS        = len(SIG_COLS)

    # Blend weights (rank-normalized scores): CPA + ThermoNet
    # Geneformer (formerly 0.35) was dropped — see header comment — and its
    # weight redistributed proportionally to the remaining two components.
    CPA_WEIGHT    = 0.54
    THERMO_WEIGHT = 0.46

    CKPT_NAME     = "thermo_c3.pt"
    SCORE_BATCH   = 500_000   # pairs per batch for CPA vectorized scoring


# ─────────────────────────────────────────────────────────────────────────────
# ThermogenicScoreNet — C3 architecture, reused for ensemble component
# ─────────────────────────────────────────────────────────────────────────────

class ThermogenicScoreNet(nn.Module):
    def __init__(self, embed_dim: int = 512, hidden: int = 256,
                 n_sigs: int = 12, dropout: float = 0.35):
        super().__init__()
        self.gene_encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden * 2), nn.LayerNorm(hidden * 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden), nn.LayerNorm(hidden),
        )
        self.pair_decoder = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden, n_sigs),
        )

    def encode_gene(self, x: torch.Tensor) -> torch.Tensor:
        return self.gene_encoder(x)

    def decode_pair(self, combined: torch.Tensor) -> torch.Tensor:
        return self.pair_decoder(combined)

    def forward(self, ea: torch.Tensor, eb: torch.Tensor) -> torch.Tensor:
        return self.pair_decoder(self.gene_encoder(ea) + self.gene_encoder(eb))


# ─────────────────────────────────────────────────────────────────────────────
# CPA scorer — vectorized pair scoring from precomputed raw signature means
# ─────────────────────────────────────────────────────────────────────────────

class CpaScorer:
    """
    Loads precomputed CPA raw signature means and normalization stats.
    Scores pairs as: z_AB = (raw_A + raw_B - mu) / std  (additive in delta-expr space).
    """

    def __init__(self, model_dir: Path, sig_cols: list):
        raw_path   = model_dir / "cpa_raw_sig_scores.npy"
        stats_path = model_dir / "cpa_sig_stats.json"
        order_path = model_dir / "test_gene_order.json"

        if not raw_path.exists():
            raise FileNotFoundError(f"{raw_path} — run C5/scripts/02_precompute_cpa.py first")

        print("[CPA] Loading cpa_raw_sig_scores.npy...")
        self.raw_scores = np.load(str(raw_path))       # (n_genes, 12)

        with open(order_path) as f:
            gene_order = json.load(f)
        self.gene_to_row = {g: i for i, g in enumerate(gene_order)}
        self.sentinel    = len(gene_order)              # row for unknown genes

        # Pad raw_scores with a zero sentinel row
        self.raw_scores = np.vstack(
            [self.raw_scores, np.zeros((1, len(sig_cols)), dtype=np.float32)]
        )

        with open(stats_path) as f:
            stats = json.load(f)
        self.sig_means = np.array([stats[s]["mean"] for s in sig_cols], dtype=np.float32)
        self.sig_stds  = np.array([max(stats[s]["std"], 1e-8) for s in sig_cols], dtype=np.float32)

        n_genes = self.raw_scores.shape[0] - 1
        print(f"[CPA] Ready: {n_genes} genes × {len(sig_cols)} signatures")

    def gene_index(self, gene: str) -> int:
        return self.gene_to_row.get(gene, self.sentinel)

    def score_pairs(
        self, idx_A: np.ndarray, idx_B: np.ndarray, batch_size: int = Config.SCORE_BATCH
    ) -> np.ndarray:
        """Return z-scores (n_pairs, n_sigs) for all pairs."""
        n_pairs  = len(idx_A)
        z_scores = np.zeros((n_pairs, self.sig_means.shape[0]), dtype=np.float32)

        for start in tqdm(range(0, n_pairs, batch_size), desc="CPA scoring"):
            end    = min(start + batch_size, n_pairs)
            raw_AB = self.raw_scores[idx_A[start:end]] + self.raw_scores[idx_B[start:end]]
            z_scores[start:end] = (raw_AB - self.sig_means) / self.sig_stds

        return z_scores


# ─────────────────────────────────────────────────────────────────────────────
# ThermogenicScoreNet inference
# ─────────────────────────────────────────────────────────────────────────────

def _run_thermo_net(
    pair_ids: list,
    model_dir: Path,
    device: torch.device,
    n_sigs: int,
) -> np.ndarray:
    """
    Run C3 ThermogenicScoreNet ensemble on pair_ids.
    Returns (n_pairs, n_sigs) or None if checkpoint missing.
    """
    ckpt_path = model_dir / Config.CKPT_NAME
    if not ckpt_path.exists():
        print(f"[ThermoNet] {ckpt_path} not found — skipping ThermoNet component.")
        return None

    print(f"[ThermoNet] Loading {ckpt_path}...")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    embed_matrix      = ckpt["embed_matrix"]
    embed_vocab       = ckpt["embed_vocab"]
    mean_embed        = ckpt["mean_embed"].astype(np.float32)
    embed_dim         = ckpt["embed_dim"]
    hidden            = ckpt.get("hidden", 256)
    single_score_dict = ckpt.get("single_score_dict", {})

    def get_embed(gene: str) -> np.ndarray:
        idx = embed_vocab.get(gene)
        return embed_matrix[idx].astype(np.float32) if idx is not None else mean_embed.copy()

    ensemble = []
    for state in ckpt["ensemble_states"]:
        m = ThermogenicScoreNet(embed_dim, hidden, n_sigs).to(device)
        m.load_state_dict(state)
        m.eval()
        ensemble.append(m)
    print(f"[ThermoNet] {len(ensemble)} ensemble models loaded.")

    # Pre-encode unique genes
    unique_genes = sorted(set(
        g for pair in pair_ids for g in pair.split("+") if g != "NC"
    ))
    gene_pos    = {g: i for i, g in enumerate(unique_genes)}
    gene_embeds = np.array([get_embed(g) for g in unique_genes], dtype=np.float32)
    n_unique    = len(unique_genes)
    ENC_BATCH   = 2048

    all_encs = []
    with torch.no_grad():
        for m in ensemble:
            chunks = []
            for s in range(0, n_unique, ENC_BATCH):
                bt = torch.tensor(gene_embeds[s : s + ENC_BATCH], device=device)
                chunks.append(m.encode_gene(bt).cpu().numpy())
            model_encs = np.vstack(chunks).astype(np.float32)
            zero_enc   = m.encode_gene(
                torch.zeros(1, embed_dim, device=device)
            ).cpu().numpy()
            all_encs.append(np.vstack([model_encs, zero_enc]).astype(np.float32))
    zero_row = n_unique

    # Single-gene additive scores
    single_sum = np.zeros((n_unique, n_sigs), np.float32)
    with torch.no_grad():
        for encs, m in zip(all_encs, ensemble):
            for s in range(0, n_unique, ENC_BATCH):
                e        = min(s + ENC_BATCH, n_unique)
                combined = torch.tensor(encs[s:e] + encs[zero_row], dtype=torch.float32, device=device)
                single_sum[s:e] += m.decode_pair(combined).cpu().numpy()
    single_pred = single_sum / len(ensemble)

    # Use experimental scores for known genes; zero for others to prevent NN extrapolation
    # dominating rankings for genes with no experimental data in single_score_dict.
    single_pred_final = np.zeros_like(single_pred)
    for i, gene in enumerate(unique_genes):
        if gene in single_score_dict:
            single_pred_final[i] = np.array(single_score_dict[gene], dtype=np.float32)
    single_pred  = single_pred_final
    singles_ext  = np.vstack([single_pred, np.zeros((1, n_sigs), np.float32)])

    # Build pair index arrays
    n_pairs = len(pair_ids)
    idx_A   = np.full(n_pairs, zero_row, np.int32)
    idx_B   = np.full(n_pairs, zero_row, np.int32)
    for i, pair in enumerate(pair_ids):
        parts = [g for g in pair.split("+") if g != "NC"]
        if parts:
            idx_A[i] = gene_pos.get(parts[0], zero_row)
        if len(parts) > 1:
            idx_B[i] = gene_pos.get(parts[1], zero_row)

    # Residual inference in batches
    PAIR_BATCH    = 200_000
    all_residuals = np.zeros((n_pairs, n_sigs), np.float32)
    print(f"[ThermoNet] Scoring {n_pairs:,} pairs...")
    for s in tqdm(range(0, n_pairs, PAIR_BATCH), desc="ThermoNet scoring"):
        e    = min(s + PAIR_BATCH, n_pairs)
        ia   = idx_A[s:e]
        ib   = idx_B[s:e]
        bsum = np.zeros((e - s, n_sigs), np.float32)
        for encs, m in zip(all_encs, ensemble):
            combined = torch.tensor(encs[ia] + encs[ib], dtype=torch.float32, device=device)
            with torch.no_grad():
                bsum += m.decode_pair(combined).cpu().numpy()
        all_residuals[s:e] = bsum / len(ensemble)

    return (singles_ext[idx_A] + singles_ext[idx_B] + all_residuals).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Rank normalization
# ─────────────────────────────────────────────────────────────────────────────

def _rank_normalize(scores: np.ndarray) -> np.ndarray:
    """Map each column to [0, 1] via rank order."""
    out = np.zeros_like(scores)
    n   = scores.shape[0]
    for j in range(scores.shape[1]):
        out[:, j] = np.argsort(np.argsort(scores[:, j])) / (n - 1)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# train() — no-op: all heavy computation done offline in C5/scripts/
# ─────────────────────────────────────────────────────────────────────────────

def train(
    data_directory_path: str,
    model_directory_path: str,
) -> None:
    print("[train] C5 precomputation is done offline via C5/scripts/ on HPC.")
    print("  Steps:")
    print("  1. python C5/scripts/01_train_cpa.py")
    print("  2. python C5/scripts/02_precompute_cpa.py")
    print("  3. cp C3/resources/thermo_c3.pt C5/resources/thermo_c3.pt")


# ─────────────────────────────────────────────────────────────────────────────
# infer()
# ─────────────────────────────────────────────────────────────────────────────

def infer(
    data_directory_path: str,
    model_directory_path: str,
    prediction_directory_path: str = None,
) -> pd.DataFrame:
    """
    CPA + ThermogenicScoreNet ensemble scorer.

    Loads precomputed resources, scores all 4,474,413 gene pairs, writes prediction.parquet.
    """
    cfg       = Config()
    data_dir  = Path(data_directory_path)
    model_dir = Path(model_directory_path)
    pred_dir  = (Path(prediction_directory_path) if prediction_directory_path
                 else model_dir.parent.parent / "prediction")
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[infer] data={data_dir}  model={model_dir}  pred={pred_dir}  device={device}")

    # ── Load candidate pairs ──────────────────────────────────────────────
    print("[infer] Loading predict_perturbations_3.parquet...")
    pp   = pd.read_parquet(data_dir / "predict_perturbations_3.parquet")
    col1 = "gene_1" if "gene_1" in pp.columns else pp.columns[0]
    col2 = "gene_2" if "gene_2" in pp.columns else pp.columns[1]
    pair_ids = (pp[col1].astype(str) + "+" + pp[col2].astype(str)).tolist()
    n_pairs  = len(pair_ids)
    print(f"  {n_pairs:,} pairs to score.")

    # ── Determine which models are available ──────────────────────────────
    cpa_available = (model_dir / "cpa_raw_sig_scores.npy").exists()
    tn_available  = (model_dir / Config.CKPT_NAME).exists()

    print(f"[infer] Components available: CPA={cpa_available}  "
          f"ThermoNet={tn_available}")
    if not any([cpa_available, tn_available]):
        raise RuntimeError(
            f"No model resources found in {model_dir}. "
            "Run C5/scripts/01-02 and copy thermo_c3.pt first."
        )

    # ── CPA scoring ───────────────────────────────────────────────────────
    cpa_scores = None
    if cpa_available:
        cpa_scorer = CpaScorer(model_dir, cfg.SIG_COLS)
        idx_A_cpa  = np.array(
            [cpa_scorer.gene_index(p.split("+")[0]) for p in pair_ids], dtype=np.int32
        )
        idx_B_cpa  = np.array(
            [cpa_scorer.gene_index(p.split("+")[1] if "+" in p else "NC") for p in pair_ids],
            dtype=np.int32,
        )
        cpa_scores = cpa_scorer.score_pairs(idx_A_cpa, idx_B_cpa)
        print(f"[CPA] scores: mean={cpa_scores.mean():.4f}  std={cpa_scores.std():.4f}")

    # ── ThermoNet scoring (C3 ensemble) ───────────────────────────────────
    thermo_scores = None
    if tn_available:
        thermo_scores = _run_thermo_net(pair_ids, model_dir, device, cfg.N_SIGS)
        if thermo_scores is not None:
            print(f"[TN]  scores: mean={thermo_scores.mean():.4f}  std={thermo_scores.std():.4f}")

    # ── Blend (rank-normalize each model, then weighted sum) ──────────────
    available_components = []
    weights_used = []

    if cpa_scores is not None:
        available_components.append(_rank_normalize(cpa_scores))
        weights_used.append(cfg.CPA_WEIGHT)
    if thermo_scores is not None:
        available_components.append(_rank_normalize(thermo_scores))
        weights_used.append(cfg.THERMO_WEIGHT)

    # Re-normalize weights to sum to 1.0 in case some components are missing
    total_w    = sum(weights_used)
    weights_used = [w / total_w for w in weights_used]
    component_names = (
        (["CPA"] if cpa_scores is not None else []) +
        (["TN"]  if thermo_scores is not None else [])
    )
    print(f"[infer] Blending: "
          + " + ".join(f"{w:.2f}×{n}" for w, n in zip(weights_used, component_names)))

    all_scores = sum(
        w * comp for w, comp in zip(weights_used, available_components)
    ).astype(np.float32)

    # ── Single-gene experimental overrides from ThermoNet checkpoint ──────
    if tn_available and (model_dir / Config.CKPT_NAME).exists():
        ckpt = torch.load(str(model_dir / Config.CKPT_NAME), map_location="cpu", weights_only=False)
        single_score_dict = ckpt.get("single_score_dict", {})
        if single_score_dict:
            n_overridden = 0
            for i, pair in enumerate(pair_ids):
                parts = [p for p in pair.split("+") if p not in ("", "NC")]
                if len(parts) == 1 and parts[0] in single_score_dict:
                    all_scores[i] = np.array(single_score_dict[parts[0]], dtype=np.float32)
                    n_overridden += 1
            print(f"[infer] Experimental single-gene overrides applied: {n_overridden}")

    # ── FinalAggScore = mean(top-3 signatures) ────────────────────────────
    top3      = np.partition(all_scores, -3, axis=1)[:, -3:]
    final_agg = top3.mean(axis=1).astype(np.float32)

    order = np.argsort(-final_agg, kind="stable")
    ranks = np.empty(n_pairs, dtype=np.int32)
    ranks[order] = np.arange(1, n_pairs + 1, dtype=np.int32)

    # ── Assemble output DataFrame ─────────────────────────────────────────
    print("[infer] Assembling output DataFrame...")
    out = {"GenePairID": pair_ids}
    for j, col in enumerate(cfg.SIG_COLS):
        out[col] = all_scores[:, j].astype(np.float32)
    out["FinalAggScore"] = final_agg
    out["Rank"]          = ranks

    pred_df = pd.DataFrame(out)
    print(f"[infer] Output: {len(pred_df):,} rows × {len(pred_df.columns)} cols")
    print("  Top-5 pairs by FinalAggScore:")
    print(pred_df.nlargest(5, "FinalAggScore")[["GenePairID", "FinalAggScore", "Rank"]]
          .to_string(index=False))

    # ── Save prediction.parquet ───────────────────────────────────────────
    out_path = pred_dir / "prediction.parquet"
    try:
        pred_dir.mkdir(parents=True, exist_ok=True)
        pred_df.to_parquet(str(out_path), index=False, engine="pyarrow")
        print(f"[infer] Saved → {out_path}")
    except Exception as exc:
        fallback = model_dir / "prediction.parquet"
        pred_df.to_parquet(str(fallback), index=False, engine="pyarrow")
        print(f"[infer] Saved (fallback) → {fallback}  ({exc})")

    return pred_df


# ─────────────────────────────────────────────────────────────────────────────
# evaluate() — local sanity check against labeled training data
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    data_directory_path: str,
    model_directory_path: str,
    output_directory_path: str,
) -> dict:
    """Spearman check against TF150 + TF15_MOI2 labeled training pairs."""
    cfg      = Config()
    data_dir = Path(data_directory_path)
    model_dir = Path(model_directory_path)

    def n_real(g: str) -> int:
        return len([x for x in str(g).split("+") if x not in ("", "NC")])

    tf150 = pd.read_csv(data_dir / "TF150_ThermoScores_perturbation.csv")
    tf15  = pd.read_csv(data_dir / "TF15_MOI2_ThermoScores_perturbation.csv")
    all_labeled = (
        pd.concat([tf150, tf15])
        .assign(_n=lambda d: d["gene"].apply(n_real))
        .query("_n in [1, 2]")
        .drop_duplicates("gene", keep="last")
        .reset_index(drop=True)
    )
    # Normalize GENE+NC → GENE
    all_labeled["gene"] = all_labeled["gene"].str.replace(r"\+NC$", "", regex=True)
    all_labeled = all_labeled.drop_duplicates("gene", keep="last").reset_index(drop=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        pairs = pd.DataFrame({
            "gene_1": all_labeled["gene"].apply(lambda g: g.split("+")[0]),
            "gene_2": all_labeled["gene"].apply(
                lambda g: g.split("+")[1] if "+" in str(g) else "NC"
            ),
        })
        pairs.to_parquet(str(tmp / "predict_perturbations_3.parquet"), index=False)
        pred_df = infer(tmpdir, str(model_dir), tmpdir)

    pred_df["_gene"] = (
        pred_df["GenePairID"]
        .str.replace(r"\+NC$", "", regex=True)
        .str.replace(r"^NC\+", "", regex=True)
    )
    gt_keyed   = all_labeled.set_index("gene")[cfg.SIG_COLS]
    pred_keyed = pred_df.groupby("_gene").last()
    common     = gt_keyed.index.intersection(pred_keyed.index)
    gt_aligned = gt_keyed.loc[common]
    pd_aligned = pred_keyed.loc[common]

    results = {}
    for col in cfg.SIG_COLS:
        if col in pd_aligned.columns:
            sp = spearmanr(pd_aligned[col], gt_aligned[col]).correlation
            results[col] = float(sp)
            print(f"  Spearman {col[:42]:42s}: {sp:.4f}")

    gt_agg = gt_aligned[cfg.SIG_COLS].apply(
        lambda r: float(np.mean(sorted(r.values)[-3:])), axis=1
    )
    agg_sp = spearmanr(pd_aligned["FinalAggScore"], gt_agg).correlation
    results["FinalAggScore"] = float(agg_sp)
    print(f"  Spearman FinalAggScore: {agg_sp:.4f}")

    # ── Single- vs double-gene breakdown (same fairness check as compare_models.py) ──
    # Single-gene pairs get GT overridden via single_score_dict — inflates the
    # all-pair number above. Double-gene pairs are never overridden, so this is
    # the only fair read on whether the model generalizes vs. memorizes.
    n_keyed = all_labeled.set_index("gene")["_n"].loc[common]
    for label, mask_n in [("single-gene (memorized via override)", 1),
                           ("double-gene (no GT leakage)", 2)]:
        sel = n_keyed == mask_n
        if sel.sum() >= 3:
            sp_n = spearmanr(pd_aligned.loc[sel, "FinalAggScore"],
                              gt_agg.loc[sel]).correlation
            print(f"  Spearman FinalAggScore [{label}, n={int(sel.sum())}]: {sp_n:.4f}")
            results[f"FinalAggScore_{mask_n}gene"] = float(sp_n)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="C5 CPA+ThermoNet inference")
    parser.add_argument("--data_dir",  default="data",           help="Data directory")
    parser.add_argument("--model_dir", default="resources",   help="C5 resources directory")
    parser.add_argument("--pred_dir",  default="predictions", help="Output directory")
    parser.add_argument("--evaluate",  action="store_true",      help="Run evaluation mode")
    args = parser.parse_args()

    if args.evaluate:
        evaluate(args.data_dir, args.model_dir, args.pred_dir)
    else:
        infer(args.data_dir, args.model_dir, args.pred_dir)
