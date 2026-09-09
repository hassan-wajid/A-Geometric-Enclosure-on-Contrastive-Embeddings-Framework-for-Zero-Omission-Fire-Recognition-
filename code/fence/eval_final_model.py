"""
Eval-only: load a finalized Stage 2 model (final_model.pt) and report
FNR / FPR / abstention on a held-out split. No training.

This is the clean replacement for the train+test logic in test_all_datasets.py.
It uses the full deployment prediction rule (inside-ball | shell | OOD).

Usage:
    python code/fence/eval_final_model.py --dataset ustc_smokers --split val
    python code/fence/eval_final_model.py --dataset dfire --ckpt path/to/final_model.pt --split test
"""

import os
import sys
from pathlib import Path
os.environ['OMP_NUM_THREADS'] = '1'
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import argparse
import json
from tqdm import tqdm

from fence.config import (
    DEVICE, FENCE_BACKBONE, FENCE_EMBEDDING_DIM, FENCE_CHECKPOINT_DIR,
    FENCE_NUM_CLUSTERS, FENCE_MAX_CLUSTERS, FENCE_MIN_CLUSTER_SIZE,
    FENCE_RADIUS_MARGIN, SPECTRAL_NORM_ENABLED, MSC_TEMPERATURE, MSC_LAMBDA,
    USE_AMP, SEED, BATCH_SIZE, FENCE_RADIUS_QUANTILE,
    FENCE_OOD_THRESHOLD_QUANTILE, FL_EPSILON, FL_SLACK_ENABLED,
)
from fence.core.fence_model import FENCEModel
from fence.stage2_train import get_dataset_paths
from fence.training.feasible_trainer import unpack_batch
from fence.geometry.ball import Ball
from fence.utils.metrics import compute_all_metrics
from train_datasets.dataset_factory import create_dataset, create_dataloaders, auto_splits


def restore_union(model, union_dict, device):
    """Rebuild the union of balls (centers + deployment radii) from a dict."""
    union = model.union_of_balls
    union.balls = []
    for bd in union_dict.get('balls', []):
        center = torch.tensor(bd['center'], device=device)
        ball = Ball(center, radius=bd['radius'], margin=bd['margin'])
        union.balls.append(ball)
    union.margin = union_dict.get('margin', union.margin)
    union._feature_dim = union_dict.get('feature_dim', None)
    return union


def restore_quarantine(model, qballs, device):
    """Rebuild the personal balls around far positives (the FNR=0 coverage)."""
    model.quarantine_balls = []
    for bd in qballs or []:
        center = torch.tensor(bd['center'], device=device)
        model.quarantine_balls.append(Ball(center, radius=bd['radius'], margin=bd['margin']))
    return model.quarantine_balls


@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    embs, labs = [], []
    for batch_data in tqdm(loader, desc="  extracting", ncols=80, leave=False):
        images, labels, _ = unpack_batch(batch_data, device)
        images = images.to(device)
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            z = model.get_embeddings(images)
        embs.append(z.float())
        labs.append(labels.to(device))
    return torch.cat(embs), torch.cat(labs)


@torch.no_grad()
def _neg_nn_quantile(neg_emb, q, chunk=2048):
    """Quantile of within-negative nearest-neighbour distances (chunked so it
    scales to FASDD-sized negative sets)."""
    nn = []
    for i in range(0, len(neg_emb), chunk):
        d = torch.cdist(neg_emb[i:i + chunk], neg_emb)
        rows = torch.arange(d.shape[0], device=d.device)
        d[rows, i + rows] = float('inf')          # exclude self-distance
        nn.append(d.min(dim=1).values)
    return float(torch.quantile(torch.cat(nn), q))


@torch.no_grad()
def _min_dist_chunked(A, B, chunk=2048):
    """Row-chunked min distance from each row of A to the set B -> np array."""
    out = []
    for i in range(0, len(A), chunk):
        out.append(torch.cdist(A[i:i + chunk], B).min(dim=1).values)
    return torch.cat(out).cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description="Eval-only for a finalized FENCE model")
    ap.add_argument("--dataset", default="ustc_smokers",
                    choices=["fasdd", "dfire", "ustc_smokers", "flame"])
    ap.add_argument("--ckpt", default=None,
                    help="Path to final_model.pt (default: standard Stage 2 location)")
    ap.add_argument("--split", default="val", choices=["val", "cal", "test"],
                    help="Which held-out split to evaluate on")
    ap.add_argument("--knn", type=int, default=15,
                    help="k for the FN neighbourhood proof")
    ap.add_argument("--save_fn_dir", default=None,
                    help="Folder to save the real FN images "
                         "(default: <ckpt_dir>/fn_images_<split>)")
    ap.add_argument("--tau_quantile", type=float, default=None,
                    help="Override the negative-cloud threshold quantile for THIS run "
                         "(default: keep the model's stored τ). Lower = stricter NEG "
                         "declarations = fewer silent FN, more abstention.")
    ap.add_argument("--knn_veto", type=int, default=0,
                    help="If >0: veto a NEG declaration when ANY of the k nearest "
                         "TRAIN neighbours is POSITIVE -> abstain instead "
                         "(zero-omission guard). 0 = off (default).")
    ap.add_argument("--tau_sweep", action="store_true",
                    help="Print the FNR/FPR/abstention frontier over τ quantiles "
                         "(diagnostic only; does not change saved metrics).")
    ap.add_argument("--pos_cap", action="store_true",
                    help="OPT-IN: enable the positive-capped τ (dual-sided) rule. "
                         "Default OFF = plain 0.99 negative-spacing quantile "
                         "(matches USTC/FLAME). On dfire the cap over-abstains "
                         "(positives overlap negatives) — kept only for the ablation.")
    ap.add_argument("--pos_cap_beta", type=float, default=0.9,
                    help="Safety factor β for the positive cap: "
                         "τ = min(coverage_quantile, β·quantile(trainPos→trainNeg)). "
                         "Fixed across datasets; do not tune per dataset.")
    ap.add_argument("--pos_cap_quantile", type=float, default=0.01,
                    help="Low quantile for the positive cap term b (robust to a few "
                         "outlier positives inside the negative cloud; raw min would "
                         "collapse τ). Fixed across datasets.")
    args = ap.parse_args()

    device = DEVICE
    ckpt_path = Path(args.ckpt) if args.ckpt else (
        FENCE_CHECKPOINT_DIR / args.dataset / FENCE_BACKBONE / "stage2" / "final_model.pt")
    print(f"Loading finalized model: {ckpt_path}")
    if not ckpt_path.exists():
        print(f"  ❌ Not found. Run Stage 2 first (stage2_train.py).")
        return

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = state['model_state_dict']
    ckpt_has_sn = any('weight_orig' in k for k in sd.keys())

    model = FENCEModel(
        backbone_name=FENCE_BACKBONE, embedding_dim=FENCE_EMBEDDING_DIM,
        num_balls=FENCE_NUM_CLUSTERS, max_balls=FENCE_MAX_CLUSTERS,
        min_cluster_size=FENCE_MIN_CLUSTER_SIZE, margin=FENCE_RADIUS_MARGIN,
        temperature=MSC_TEMPERATURE, lambda_class=MSC_LAMBDA,
        use_mean_shift=True, spectral_norm=ckpt_has_sn, device=device,
    )
    model.load_state_dict(sd)
    if 'union_of_balls' in state:
        restore_union(model, state['union_of_balls'], device)
    restore_quarantine(model, state.get('quarantine_balls'), device)
    print(f"  ✓ Loaded. Main balls: {len(model.union_of_balls)}  "
          f"Quarantine balls: {len(model.quarantine_balls)}  "
          f"(verified at finalize: {state.get('verified', 'N/A')})")

    # --- COMPLETE GEOMETRY REPORT ---
    print("\n  --- ENCLOSURE GEOMETRY ---")
    print(f"  margin m (abstain band width)     : {model.union_of_balls.margin:.4f}")
    radii = [b.radius for b in model.union_of_balls.balls]
    for j, b in enumerate(model.union_of_balls.balls):
        occ = b.occupancy() if hasattr(b, 'occupancy') else len(getattr(b, '_assigned_indices', []))
        print(f"    main ball {j}: R={b.radius:.4f}  (covers ~{occ} train pos)")
    if radii:
        print(f"  main radii  min/mean/max          : {min(radii):.4f} / "
              f"{sum(radii)/len(radii):.4f} / {max(radii):.4f}")
    qr = [b.radius for b in model.quarantine_balls]
    print(f"  personal (quarantine) balls       : {len(qr)}  "
          f"(each radius = m = {model.union_of_balls.margin:.4f})")
    print(f"  decision: inside ball -> POS | in negative cloud -> NEG | else -> ABSTAIN")

    # Data — AUTO split (decides val/test per dataset automatically)
    image_dir, ann_dir = get_dataset_paths(args.dataset)
    ds = create_dataset(args.dataset, image_dir, ann_dir)
    splits = auto_splits(ds)

    eval_samples = splits.get(args.split)
    if not eval_samples:
        print(f"  ⚠️ '{args.split}' empty for {args.dataset}; using 'val'.")
        args.split = 'val'
        eval_samples = splits['val']

    loaders = create_dataloaders(
        {'train': splits['train'], 'eval': eval_samples}, test_mode=False
    )
    train_loader = loaders['train']
    eval_loader = loaders['eval']

    # Re-populate the OOD + negative-cloud reference (not stored in the checkpoint)
    # so predict()'s OOD and "confidently negative" tests behave as in deployment.
    print("  Rebuilding OOD + negative-cloud reference from train embeddings...")
    train_emb, train_lab = extract_embeddings(model, train_loader, device)
    model.store_ood_reference(train_emb, train_lab,
                              pos_cap=args.pos_cap,
                              pos_cap_beta=args.pos_cap_beta,
                              pos_cap_quantile=args.pos_cap_quantile)
    print(f"  negative-cloud threshold τ        : "
          f"{getattr(model, '_neg_threshold', None)}")
    _a = getattr(model, '_tau_coverage', None)
    _b = getattr(model, '_tau_pos_cap', None)
    _bind = getattr(model, '_tau_binding', None)
    if _a is not None:
        print(f"    τ terms:  coverage a={_a:.6f}"
              + (f"   pos-cap β·b={args.pos_cap_beta:.2f}×{_b:.6f}"
                 f"={args.pos_cap_beta*_b:.6f}" if _b is not None else "   pos-cap OFF")
              + f"   ->  binding: {_bind}")
        if _bind == 'pos_cap':
            print(f"    (train positives approach the negative cloud down to "
                  f"b={_b:.6f}; τ capped below it so such points ABSTAIN "
                  f"instead of being silently called negative)")

    # Optional per-run τ override (stricter NEG declarations -> more abstention).
    # Does NOT touch the checkpoint or other datasets.
    if args.tau_quantile is not None:
        neg_tr = train_emb[train_lab == 0]
        tau_new = _neg_nn_quantile(neg_tr, args.tau_quantile)
        print(f"  --tau_quantile {args.tau_quantile}: overriding τ "
              f"{float(model._neg_threshold):.6f} -> {tau_new:.6f}")
        model._neg_threshold = tau_new

    # --- Predict on the eval split (full deployment rule) ---
    print(f"  Evaluating on '{args.split}' split...")
    all_labels, all_scores, all_preds, all_abstained = [], [], [], []
    all_emb, all_s, all_ood = [], [], []  # for the FN "why" proof + τ sweep
    model.eval()
    with torch.no_grad():
        for batch_data in tqdm(eval_loader, desc="  predicting", ncols=80, leave=False):
            images, labels, _ = unpack_batch(batch_data, device)
            images = images.to(device)
            pred = model.predict(images)
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(np.atleast_1d(pred['scores'].cpu().numpy()))
            all_preds.extend(np.atleast_1d(pred['prediction'].cpu().numpy()))
            all_abstained.extend(np.atleast_1d(pred['abstained'].cpu().numpy()))
            all_emb.append(pred['embeddings'].float().cpu())
            all_s.append(np.atleast_1d(pred['distance_to_nearest'].cpu().numpy()))
            all_ood.extend(np.atleast_1d(pred['ood'].cpu().numpy()))

    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    all_preds = np.array(all_preds)
    all_abstained = np.array(all_abstained)
    all_ood = np.array(all_ood).astype(bool)
    ev_emb = torch.cat(all_emb)           # (N, D) in eval order == eval_samples order
    s_arr  = np.concatenate(all_s)        # signed dist to nearest ball (R - d)

    # --- Optional zero-omission guard (kNN veto): never declare NEG when a
    # positive is among the k nearest TRAIN neighbours -> abstain instead.
    n_vetoed, n_vetoed_pos = 0, 0
    if args.knn_veto > 0:
        neg_call = (all_preds == 0) & (~all_abstained.astype(bool))
        idx = np.where(neg_call)[0]
        if len(idx):
            k = min(args.knn_veto, train_emb.shape[0])
            cand = ev_emb[idx].to(device)
            has_pos = np.zeros(len(idx), dtype=bool)
            for i in range(0, len(idx), 2048):
                d = torch.cdist(cand[i:i + 2048], train_emb)
                _, nn_i = torch.topk(d, k=k, dim=1, largest=False)
                has_pos[i:i + 2048] = (train_lab[nn_i] == 1).any(dim=1).cpu().numpy()
            veto_idx = idx[has_pos]
            all_abstained[veto_idx] = True
            n_vetoed = len(veto_idx)
            n_vetoed_pos = int((all_labels[veto_idx] == 1).sum())
            print(f"  kNN veto (k={k}): {n_vetoed} NEG declaration(s) -> ABSTAIN "
                  f"({n_vetoed_pos} true positive(s) rescued, "
                  f"{n_vetoed - n_vetoed_pos} true negative(s) now abstain)")

    metrics = compute_all_metrics(all_labels, all_scores, all_preds, all_abstained)

    fnr = metrics.get('fnr', 1.0)
    fpr = metrics.get('fpr', 1.0)
    f1 = metrics.get('f1', 0.0)
    abst = metrics.get('abstention_rate', float(all_abstained.mean()))
    sel_fnr = metrics.get('non_abstained_fnr', fnr)   # FNR among COVERED (non-abstained)
    sel_fpr = metrics.get('non_abstained_fpr', fpr)   # FPR among COVERED (non-abstained)
    coverage = 1.0 - abst
    # "Operational" for an abstention system: zero omission among covered samples,
    # low false-alarm among covered, with reasonable coverage.
    operational = bool(sel_fnr == 0.0 and sel_fpr < 0.1)

    # Raw counts (numbers, not just percentages)
    n_pos = int((all_labels == 1).sum())
    n_neg = int((all_labels == 0).sum())
    n_abst = int(all_abstained.sum())
    # among NON-abstained
    keep = ~all_abstained.astype(bool)
    yk, pk = all_labels[keep], all_preds[keep]
    tp = int(((yk == 1) & (pk == 1)).sum())
    fn = int(((yk == 1) & (pk == 0)).sum())   # missed positives (real omissions)
    fp = int(((yk == 0) & (pk == 1)).sum())
    tn = int(((yk == 0) & (pk == 0)).sum())
    abst_pos = int(((all_labels == 1) & all_abstained.astype(bool)).sum())
    abst_neg = int(((all_labels == 0) & all_abstained.astype(bool)).sum())

    print("\n" + "=" * 60)
    print(f"  Dataset: {args.dataset}   Split: {args.split}   N={len(all_labels)}")
    print(f"  Positives: {n_pos}   Negatives: {n_neg}")
    print("-" * 60)
    print(f"  COUNTS (non-abstained):  TP={tp}  FN={fn}  FP={fp}  TN={tn}")
    print(f"  ABSTAINED: {n_abst}  (positives={abst_pos}, negatives={abst_neg})")
    print("-" * 60)
    print(f"  RAW (all samples)        FNR: {fnr:.6f}  FPR: {fpr:.6f}  F1: {f1:.6f}")
    print(f"  SELECTIVE (covered only) FNR: {sel_fnr:.6f}  FPR: {sel_fpr:.6f}")
    print(f"     -> missed positives (FN) = {fn} / {n_pos - abst_pos} covered positives")
    print(f"     -> false alarms   (FP) = {fp} / {n_neg - abst_neg} covered negatives")
    print(f"  Coverage: {coverage:.4f}   Abstention: {abst:.4f}")
    print(f"  Operational (selective FNR=0 & FPR<0.1): {'✅ YES' if operational else '❌ NO'}")
    print("=" * 60)

    # ================================================================
    #  FN DIAGNOSIS + SAVE REAL FN IMAGES
    #  A non-abstained FN is a positive that was predicted NEGATIVE. By the
    #  decision rule that can ONLY happen when the positive's embedding sits
    #  inside the training NEGATIVE cloud (d_neg <= tau) — i.e. it genuinely
    #  looks like a known negative. We prove it here per-FN and save the images.
    # ================================================================
    import shutil
    fn_mask_full = ((all_labels == 1) & (all_preds == 0) &
                    (~all_abstained.astype(bool)))
    fn_idx = np.where(fn_mask_full)[0]
    fn_report = []

    print("\n" + "=" * 60)
    print(f"  FN DIAGNOSIS — missed positives (non-abstained): {len(fn_idx)}")
    if len(fn_idx) == 0:
        print("  ✅ No FN — zero omission on this split. Nothing to save.")
    else:
        neg_ref = getattr(model, '_train_neg_embeddings', None)
        tau     = getattr(model, '_neg_threshold', None)
        fn_e    = ev_emb[fn_idx].to(device)

        # distance to the nearest training NEGATIVE (the "looks like a negative" proof)
        if neg_ref is not None and len(neg_ref) > 0:
            d_neg = torch.cdist(fn_e, neg_ref).min(dim=1).values.cpu().numpy()
        else:
            d_neg = np.full(len(fn_idx), np.nan)

        # k-NN label make-up in TRAIN embedding space (deep in negatives vs. gap)
        knn = min(args.knn, train_emb.shape[0])
        Dtr = torch.cdist(fn_e, train_emb)
        dvals, nn_idx = torch.topk(Dtr, k=knn, dim=1, largest=False)
        nn_lab = train_lab[nn_idx].cpu().numpy()
        nn_d   = dvals.cpu().numpy()

        save_dir = Path(args.save_fn_dir) if args.save_fn_dir else (
            ckpt_path.parent / f"fn_images_{args.split}")
        save_dir.mkdir(parents=True, exist_ok=True)

        for rank, i in enumerate(fn_idx):
            pct_neg = 100.0 * float((nn_lab[rank] == 0).mean())
            dn      = float(d_neg[rank])
            s_out   = float(s_arr[i])                 # R - d to nearest ball (<0 = outside)
            in_neg  = (tau is not None) and (dn <= float(tau))
            if pct_neg >= 70:
                verdict = ("INSIDE negative cloud (looks like a known negative "
                           "-> unrecoverable at rule level)")
            elif pct_neg >= 30:
                verdict = ("mixed boundary neighbourhood "
                           "(tighter τ / kNN veto would abstain it)")
            else:
                verdict = ("sits among POSITIVES yet within τ of a negative "
                           "-> RECOVERABLE (--knn_veto abstains it)")
            print(f"    FN #{rank}: d_to_nearest_NEG={dn:.4f} (tau={float(tau):.4f}"
                  f"{' -> inside' if in_neg else ''}), "
                  f"dist_outside_ball={-s_out:.4f}, "
                  f"{pct_neg:.0f}% of {knn} nearest TRAIN neighbours are NEGATIVE "
                  f"-> {verdict}")

            # copy the real image
            sample = eval_samples[i]
            src = sample.get('img_path')
            saved = None
            if src and Path(src).exists():
                ext = Path(src).suffix or ".jpg"
                dst = save_dir / (f"FN{rank:03d}_pctneg{int(pct_neg):02d}"
                                  f"_dneg{dn:.3f}{ext}")
                try:
                    shutil.copy(src, dst)
                    saved = str(dst)
                except Exception as e:
                    print(f"        ⚠️ could not copy {src}: {e}")
            else:
                print(f"        ⚠️ no img_path for FN #{rank} (src={src})")

            fn_report.append({
                'rank': rank, 'eval_index': int(i),
                'img_path': src, 'saved_to': saved,
                'label': 1, 'predicted': 'NEGATIVE',
                'd_to_nearest_neg': dn, 'tau': float(tau) if tau is not None else None,
                'inside_negative_cloud': bool(in_neg),
                'dist_outside_ball': -s_out,
                'pct_negative_neighbours': pct_neg,
                'nearest_neighbour_dist': float(nn_d[rank, 0]),
                'verdict': verdict,
            })

        # actionable summary: what τ would abstain ALL current FNs
        if np.isfinite(d_neg).all() and tau is not None:
            print(f"  -> to abstain ALL {len(fn_idx)} FNs via τ alone: "
                  f"τ < {float(d_neg.min()):.4f} (current τ = {float(tau):.4f}). "
                  f"Use --tau_sweep to see the abstention cost.")

        print(f"  🖼️  Saved {len([r for r in fn_report if r['saved_to']])} "
              f"FN image(s) to: {save_dir}")
        with open(save_dir / "fn_report.json", 'w') as f:
            json.dump(fn_report, f, indent=2)
        print(f"  📝 Per-FN diagnosis: {save_dir / 'fn_report.json'}")
    print("=" * 60)

    # ================================================================
    #  τ SWEEP — the deployment-rule frontier (diagnostic only).
    #  Shows, for each τ quantile, how many positives would still be silently
    #  called NEG (FN) and what the abstention cost is. Ball membership and OOD
    #  are τ-independent, so this is exact without re-embedding.
    # ================================================================
    if args.tau_sweep:
        neg_tr = train_emb[train_lab == 0]
        d_neg_all = _min_dist_chunked(ev_emb.to(device), neg_tr)
        inside_arr = all_preds.astype(bool)
        y_ = all_labels
        print("\n  τ SWEEP (kNN veto NOT applied):")
        print(f"  {'q':>5} {'τ':>10} {'FN':>4} {'FP':>4} {'abst+':>6} {'abst-':>6} "
              f"{'selFNR':>9} {'selFPR':>9} {'cover':>7}")
        for q_ in (0.85, 0.90, 0.95, 0.99):
            tau_q = _neg_nn_quantile(neg_tr, q_)
            pred_neg_ = (d_neg_all <= tau_q) & (~inside_arr) & (~all_ood)
            abst_ = (~inside_arr) & (~pred_neg_)
            fn_q = int(((y_ == 1) & pred_neg_).sum())
            fp_q = int(((y_ == 0) & inside_arr).sum())
            ap_ = int(((y_ == 1) & abst_).sum())
            an_ = int(((y_ == 0) & abst_).sum())
            cov_p = max(int((y_ == 1).sum()) - ap_, 1)
            cov_n = max(int((y_ == 0).sum()) - an_, 1)
            print(f"  {q_:>5.2f} {tau_q:>10.6f} {fn_q:>4} {fp_q:>4} {ap_:>6} {an_:>6} "
                  f"{fn_q / cov_p:>9.6f} {fp_q / cov_n:>9.6f} "
                  f"{1 - (ap_ + an_) / len(y_):>7.4f}")

    out = {
        'dataset': args.dataset, 'split': args.split, 'n': int(len(all_labels)),
        'n_pos': n_pos, 'n_neg': n_neg,
        'counts': {'TP': tp, 'FN': fn, 'FP': fp, 'TN': tn,
                   'abstained_pos': abst_pos, 'abstained_neg': abst_neg},
        'fnr': float(fnr), 'fpr': float(fpr), 'f1': float(f1),
        'selective_fnr': float(sel_fnr), 'selective_fpr': float(sel_fpr),
        'coverage': float(coverage),
        'abstention_rate': float(abst), 'operational': operational,
        'checkpoint': str(ckpt_path),
        'fn_diagnosis': fn_report,
        # --- CONFIG SNAPSHOT: makes every run self-documenting for the paper.
        # Prove all datasets used identical settings when writing the results table.
        'config_snapshot': {
            'seed': SEED,
            'backbone': FENCE_BACKBONE,
            'embedding_dim': FENCE_EMBEDDING_DIM,
            'batch_size': BATCH_SIZE,
            'margin': FENCE_RADIUS_MARGIN,
            'radius_quantile': FENCE_RADIUS_QUANTILE,
            'ood_threshold_quantile': FENCE_OOD_THRESHOLD_QUANTILE,
            'num_clusters_setting': FENCE_NUM_CLUSTERS,     # 0 = auto
            'max_clusters': FENCE_MAX_CLUSTERS,
            'min_cluster_size': FENCE_MIN_CLUSTER_SIZE,
            'spectral_norm_enabled': SPECTRAL_NORM_ENABLED,
            'checkpoint_has_spectral_norm': bool(ckpt_has_sn),
            'msc_temperature': MSC_TEMPERATURE,
            'msc_lambda': MSC_LAMBDA,
            'feasible_learning': {
                'epsilon': FL_EPSILON,
                'slack_enabled': FL_SLACK_ENABLED,
            },
            # deployment-rule overrides used for THIS eval run (None/0 = stock rule)
            'tau_quantile_override': args.tau_quantile,
            'knn_veto_k': args.knn_veto,
            'knn_veto_flipped': n_vetoed,
            'knn_veto_rescued_positives': n_vetoed_pos,
            # positive-capped tau rule (dual-sided): tau = min(a, beta*b)
            'tau_rule': {
                'pos_cap_enabled': args.pos_cap,
                'pos_cap_beta': args.pos_cap_beta,
                'pos_cap_quantile': args.pos_cap_quantile,
                'coverage_term_a': getattr(model, '_tau_coverage', None),
                'pos_cap_term_b': getattr(model, '_tau_pos_cap', None),
                'binding': getattr(model, '_tau_binding', None),
            },
            # actual fitted geometry (what the model ended up with)
            'geometry': {
                'n_main_balls': len(model.union_of_balls.balls),
                'n_personal_balls': len(model.quarantine_balls),
                'main_radii': [float(b.radius) for b in model.union_of_balls.balls],
                'neg_threshold_tau': (float(model._neg_threshold)
                                      if getattr(model, '_neg_threshold', None) is not None
                                      else None),
            },
            'verified_zero_fnr_at_finalize': state.get('verified', None),
        },
    }
    out_path = ckpt_path.parent / f"eval_{args.split}.json"
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"  📊 Saved: {out_path}")


if __name__ == "__main__":
    main()