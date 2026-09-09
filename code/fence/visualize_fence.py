"""
FENCE exact decision-space visualization: 2x2 grid over all four datasets.

Each panel plots the two quantities the FENCE decision is ACTUALLY made on, so
there is no projection distortion of any kind:

    x = signed margin to the nearest enclosure ball   s = R - d   (main + personal)
            x >= 0  <=>  point is INSIDE a ball            -> predicted POSITIVE
    y = signed margin to the negative cloud           tau - d_neg
            y >= 0  <=>  point is INSIDE the negative cloud -> predicted NEGATIVE

The FENCE rule maps to exact, axis-aligned regions (inside-ball wins):
    x >= 0                 -> POS      (right half-plane)
    x < 0  and  y >= 0     -> NEG      (top-left)
    x < 0  and  y < 0      -> ABSTAIN  (bottom-left)

Notes:
- Axes use a symmetric-log scale (linear near zero, logarithmic outside) so the
  dense core near the decision boundary is expanded while far outliers remain
  visible. Region membership is sign-based, so the regions stay exact.
- Abstentions that sat INSIDE the negative cloud but were vetoed by the OOD
  gate are drawn as gray hexagons ("abstained by OOD gate"); without this
  distinction they look like errors inside the NEGATIVE region.
- Every FP, FN, and abstention is always drawn; only bulk TP/TN are subsampled.

Usage:
    python code/fence/visualize_fence.py
    python code/fence/visualize_fence.py --split test --max_bulk 4000
    python code/fence/visualize_fence.py --datasets fasdd dfire
    python code/fence/visualize_fence.py --linear          # old linear axes
"""

import os
import sys
from pathlib import Path
os.environ['OMP_NUM_THREADS'] = '1'
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm

from fence.config import (
    DEVICE, BASE_DIR, FENCE_BACKBONE, FENCE_EMBEDDING_DIM, FENCE_CHECKPOINT_DIR,
    FENCE_NUM_CLUSTERS, FENCE_MAX_CLUSTERS, FENCE_MIN_CLUSTER_SIZE,
    FENCE_RADIUS_MARGIN, MSC_TEMPERATURE, MSC_LAMBDA, USE_AMP, SEED,
)
from fence.core.fence_model import FENCEModel
from fence.stage2_train import get_dataset_paths
from fence.training.feasible_trainer import unpack_batch
from fence.geometry.ball import Ball
from train_datasets.dataset_factory import create_dataset, create_dataloaders, auto_splits


# ---------------------------------------------------------------------------
# Palette and styling
# ---------------------------------------------------------------------------
PAL = {
    "tn":   "#4d9de0",   # blue
    "tp":   "#2ecc71",   # green
    "aneg": "#f4d03f",   # yellow
    "apos": "#e67e22",   # orange
    "fp":   "#9b59b6",   # purple
    "fn":   "#e74c3c",   # red
    "ood":  "#7f8c8d",   # gray (OOD-gated abstentions)
}
DATASET_TITLES = {
    "fasdd": "FASDD",
    "dfire": "D-Fire",
    "ustc_smokers": "USTC SmokeRS",
    "flame": "FLAME",
}
ALL_CATS = ['tp', 'tn', 'fp', 'fn', 'apos', 'aneg', 'apos_ood', 'aneg_ood']
RARE_CATS = ['fp', 'fn', 'apos', 'aneg', 'apos_ood', 'aneg_ood']

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 120,
})


# ---------------------------------------------------------------------------
# Model loading (same pattern as eval_final_model.py)
# ---------------------------------------------------------------------------
def restore_union(model, union_dict, device):
    union = model.union_of_balls
    union.balls = []
    for bd in union_dict.get('balls', []):
        center = torch.tensor(bd['center'], device=device)
        union.balls.append(Ball(center, radius=bd['radius'], margin=bd['margin']))
    union.margin = union_dict.get('margin', union.margin)
    union._feature_dim = union_dict.get('feature_dim', None)
    return union


def restore_quarantine(model, qballs, device):
    model.quarantine_balls = []
    for bd in qballs or []:
        center = torch.tensor(bd['center'], device=device)
        model.quarantine_balls.append(Ball(center, radius=bd['radius'], margin=bd['margin']))
    return model.quarantine_balls


def load_finalized_model(ckpt_path, device):
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
    return model, state


@torch.no_grad()
def extract_embeddings(model, loader, device, desc="  extracting"):
    model.eval()
    embs, labs = [], []
    for batch_data in tqdm(loader, desc=desc, ncols=80, leave=False):
        images, labels, _ = unpack_batch(batch_data, device)
        images = images.to(device)
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            z = model.get_embeddings(images)
        embs.append(z.float())
        labs.append(labels.to(device))
    return torch.cat(embs), torch.cat(labs)


# ---------------------------------------------------------------------------
# Decision quantities on embeddings, chunked to bound memory
# ---------------------------------------------------------------------------
@torch.no_grad()
def quarantine_margins(model, z, chunk=2048):
    q = model.quarantine_balls
    if not q:
        return torch.full((len(z),), float('-inf'), device=z.device)
    C = torch.stack([b.center for b in q]).float()
    R = torch.tensor([b.radius for b in q], device=z.device, dtype=torch.float32)
    out = []
    for i in range(0, len(z), chunk):
        d = torch.cdist(z[i:i + chunk].float(), C)
        out.append((R.unsqueeze(0) - d).max(dim=1).values)
    return torch.cat(out)


@torch.no_grad()
def neg_cloud_margins(model, z, chunk=2048):
    neg = model._train_neg_embeddings
    tau = float(model._neg_threshold)
    out = []
    for i in range(0, len(z), chunk):
        d = torch.cdist(z[i:i + chunk].float(), neg).min(dim=1).values
        out.append(tau - d)
    return torch.cat(out)


@torch.no_grad()
def ood_flags(model, z, chunk=512):
    out = []
    for i in range(0, len(z), chunk):
        out.append(model._is_ood(z[i:i + chunk].float()))
    return torch.cat(out)


@torch.no_grad()
def compute_decisions(model, z):
    """Full deployment decision rule on embeddings z (identical to eval_all.py)."""
    s_main = model.union_of_balls.distance_to_nearest(z)      # R - d, >=0 inside
    s_q = quarantine_margins(model, z)
    x = torch.maximum(s_main, s_q)                            # enclosure margin
    y = neg_cloud_margins(model, z)                           # tau - d_neg
    ood = ood_flags(model, z)

    inside = x >= 0
    conf_neg = y >= 0
    pred_negative = conf_neg & (~inside) & (~ood)
    abstain = (~inside) & (~pred_negative)

    return {
        'x': x.cpu().numpy(),
        'y': y.cpu().numpy(),
        'conf_neg': conf_neg.cpu().numpy(),
        'ood': ood.cpu().numpy(),
        'pred': inside.int().cpu().numpy(),
        'abstain': abstain.cpu().numpy(),
    }


def outcome_categories(labels, dec):
    """Outcome category per point. Abstentions are split by cause: points that
    sat inside the negative cloud but were vetoed by the OOD gate get their own
    category (apos_ood / aneg_ood)."""
    cat = np.full(len(labels), None, dtype=object)
    pos = labels == 1
    neg = ~pos
    abst = dec['abstain'].astype(bool)
    conf = dec['conf_neg'].astype(bool)
    cat[pos & abst & conf] = 'apos_ood'
    cat[neg & abst & conf] = 'aneg_ood'
    cat[pos & abst & ~conf] = 'apos'
    cat[neg & abst & ~conf] = 'aneg'
    decided_pos = (dec['pred'] == 1) & (~abst)
    decided_neg = (dec['pred'] == 0) & (~abst)
    cat[pos & decided_pos] = 'tp'
    cat[neg & decided_pos] = 'fp'
    cat[pos & decided_neg] = 'fn'
    cat[neg & decided_neg] = 'tn'
    return cat


def subsample_bulk(cat, max_bulk, seed=SEED):
    """Keep ALL rare points (fp, fn, abstentions); subsample only tp/tn."""
    rng = np.random.RandomState(seed)
    keep = np.zeros(len(cat), dtype=bool)
    keep[np.isin(cat, RARE_CATS)] = True
    for c in ['tp', 'tn']:
        idx = np.where(cat == c)[0]
        if len(idx) > max_bulk:
            idx = rng.choice(idx, size=max_bulk, replace=False)
        keep[idx] = True
    return keep


# ---------------------------------------------------------------------------
# Per-dataset pipeline: returns everything a panel needs, frees the model
# ---------------------------------------------------------------------------
def prepare_cell(dataset, split, ckpt_override, device):
    ckpt_path = Path(ckpt_override) if ckpt_override else (
        FENCE_CHECKPOINT_DIR / dataset / FENCE_BACKBONE / "stage2" / "final_model.pt")
    print(f"\n[{dataset}] checkpoint: {ckpt_path}")
    if not ckpt_path.exists():
        print(f"  Not found, skipping {dataset}.")
        return None

    model, state = load_finalized_model(ckpt_path, device)
    print(f"  main balls: {len(model.union_of_balls)}  "
          f"personal balls: {len(model.quarantine_balls)}  "
          f"verified at finalize: {state.get('verified', 'N/A')}")

    image_dir, ann_dir = get_dataset_paths(dataset)
    ds = create_dataset(dataset, image_dir, ann_dir)
    splits = auto_splits(ds)

    use_split = split
    eval_samples = splits.get(use_split)
    if not eval_samples:
        for fallback in ['test', 'val', 'cal']:
            if splits.get(fallback):
                print(f"  '{use_split}' empty for {dataset}; using '{fallback}'.")
                use_split = fallback
                eval_samples = splits[fallback]
                break
    if not eval_samples:
        print(f"  No held-out split for {dataset}, skipping.")
        return None

    loaders = create_dataloaders(
        {'ref': splits['train'], 'eval': eval_samples}, test_mode=False)

    print("  rebuilding OOD + negative-cloud reference from train embeddings...")
    train_emb, train_lab = extract_embeddings(model, loaders['ref'], device,
                                              desc=f"  {dataset} train")
    model.store_ood_reference(train_emb, train_lab)
    tau = float(model._neg_threshold)

    z, lab = extract_embeddings(model, loaders['eval'], device,
                                desc=f"  {dataset} {use_split}")
    dec = compute_decisions(model, z)
    labels = lab.cpu().numpy().astype(int)
    cat = outcome_categories(labels, dec)

    counts = {c: int((cat == c).sum()) for c in ALL_CATS}
    print(f"  tau={tau:.6f}  outcomes: {counts}")

    # free GPU memory before the next dataset
    del model, train_emb, train_lab, z, lab
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {'dec': dec, 'cat': cat, 'counts': counts, 'split': use_split, 'tau': tau}


# ---------------------------------------------------------------------------
# Panel drawing
# ---------------------------------------------------------------------------
def _linthresh(v):
    """Symmetric-log linear half-width: 20th percentile of nonzero magnitudes."""
    a = np.abs(v)
    a = a[a > 0]
    if len(a) == 0:
        return 1e-3
    return float(max(np.quantile(a, 0.2), 1e-6))


def draw_decision_panel(ax, cell, max_bulk, linear=False):
    dec, cat, counts = cell['dec'], cell['cat'], cell['counts']
    keep = subsample_bulk(cat, max_bulk)
    x, y, ck = dec['x'][keep], dec['y'][keep], cat[keep]

    xpad = 0.05 * (x.max() - x.min() + 1e-9)
    ypad = 0.05 * (y.max() - y.min() + 1e-9)
    xlim = (x.min() - xpad, x.max() + xpad)
    ylim = (y.min() - ypad, y.max() + ypad)

    # region tints: POS right half, NEG top-left, ABSTAIN bottom-left
    ax.axvspan(0, xlim[1], color=PAL['tp'], alpha=0.05, zorder=0)
    ax.fill_between([xlim[0], 0], 0, ylim[1], color=PAL['tn'], alpha=0.07, zorder=0)
    ax.fill_between([xlim[0], 0], ylim[0], 0, color=PAL['aneg'], alpha=0.10, zorder=0)
    ax.axvline(0, color='#444444', lw=1.2, zorder=1)
    ax.axhline(0, color='#444444', lw=1.2, zorder=1)

    style = {
        'tn':      dict(marker='o', s=5,   c=PAL['tn'],   alpha=0.40, edge='none'),
        'tp':      dict(marker='o', s=7,   c=PAL['tp'],   alpha=0.50, edge='none'),
        'aneg_ood': dict(marker='h', s=30, c=PAL['ood'],  alpha=0.95, edge='#2f3a3a'),
        'apos_ood': dict(marker='h', s=40, c=PAL['ood'],  alpha=0.95, edge='#2f3a3a'),
        'aneg':    dict(marker='D', s=26,  c=PAL['aneg'], alpha=0.95, edge='#9a7d0a'),
        'apos':    dict(marker='^', s=36,  c=PAL['apos'], alpha=0.95, edge='#784212'),
        'fp':      dict(marker='s', s=45,  c=PAL['fp'],   alpha=0.95, edge='#4a235a'),
        'fn':      dict(marker='*', s=180, c=PAL['fn'],   alpha=1.0,  edge='#7b241c'),
    }
    for name in ['tn', 'tp', 'aneg_ood', 'apos_ood', 'aneg', 'apos', 'fp', 'fn']:
        m = ck == name
        if m.any():
            st = style[name]
            ax.scatter(x[m], y[m], marker=st['marker'], s=st['s'], c=st['c'],
                       alpha=st['alpha'], edgecolors=st['edge'], linewidths=0.7,
                       rasterized=(name in ['tn', 'tp']), zorder=3)

    ax.text(0.985, 0.03, 'POSITIVE', transform=ax.transAxes, ha='right',
            color='#127a3e', weight='bold', fontsize=9)
    ax.text(0.015, 0.97, 'NEGATIVE', transform=ax.transAxes, va='top',
            color='#1f5d8c', weight='bold', fontsize=9)
    ax.text(0.015, 0.03, 'ABSTAIN', transform=ax.transAxes,
            color='#9a7d0a', weight='bold', fontsize=9)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if not linear:
        ax.set_xscale('symlog', linthresh=_linthresh(dec['x']))
        ax.set_yscale('symlog', linthresh=_linthresh(dec['y']))
    ax.set_xlabel("signed margin to nearest ball   $s = R - d$")
    ax.set_ylabel("signed margin to negative cloud   $\\tau - d_{neg}$")


def cell_title(name, cell):
    c = cell['counts']
    abst = c['apos'] + c['aneg'] + c['apos_ood'] + c['aneg_ood']
    return (f"{DATASET_TITLES.get(name, name)} ({cell['split']})\n"
            f"TP {c['tp']}   TN {c['tn']}   FP {c['fp']}   "
            f"FN {c['fn']}   abstain {abst}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="FENCE exact decision-space grid over all datasets")
    ap.add_argument("--datasets", nargs="+",
                    default=["fasdd", "dfire", "ustc_smokers", "flame"],
                    choices=["fasdd", "dfire", "ustc_smokers", "flame"],
                    help="Datasets to include (default: all four)")
    ap.add_argument("--split", default="test", choices=["val", "cal", "test"],
                    help="Held-out split to plot (falls back per dataset if empty)")
    ap.add_argument("--max_bulk", type=int, default=5000,
                    help="Max TP/TN points per panel (rare outcomes always drawn)")
    ap.add_argument("--linear", action="store_true",
                    help="Use linear axes instead of the default symmetric-log axes")
    ap.add_argument("--out", default=None,
                    help="Output path without extension "
                         "(default: results/figures/fence_decision_grid_<backbone>_<split>)")
    for ds_name in ["fasdd", "dfire", "ustc_smokers", "flame"]:
        ap.add_argument(f"--ckpt_{ds_name}", default=None,
                        help=f"Override checkpoint path for {ds_name}")
    args = ap.parse_args()

    device = DEVICE

    # --- compute every cell first (models are loaded and freed one by one) ---
    cells = {}
    for name in args.datasets:
        cell = prepare_cell(name, args.split, getattr(args, f"ckpt_{name}"), device)
        if cell is not None:
            cells[name] = cell

    if not cells:
        print("No datasets could be prepared. Check checkpoints.")
        return

    # --- figure ---
    n = len(cells)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(7.2 * ncols, 6.2 * nrows),
                             squeeze=False)
    for idx, (name, cell) in enumerate(cells.items()):
        ax = axes[idx // ncols][idx % ncols]
        draw_decision_panel(ax, cell, args.max_bulk, linear=args.linear)
        ax.set_title(cell_title(name, cell))
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis('off')

    # one shared legend
    legend_handles = [
        Line2D([], [], marker='o', ls='', markersize=6, color=PAL['tp'],
               alpha=0.6, label='true positives'),
        Line2D([], [], marker='o', ls='', markersize=6, color=PAL['tn'],
               alpha=0.6, label='true negatives'),
        Line2D([], [], marker='s', ls='', markersize=7, markerfacecolor=PAL['fp'],
               markeredgecolor='#4a235a', label='false positives'),
        Line2D([], [], marker='*', ls='', markersize=13, markerfacecolor=PAL['fn'],
               markeredgecolor='#7b241c', label='false negatives'),
        Line2D([], [], marker='^', ls='', markersize=7, markerfacecolor=PAL['apos'],
               markeredgecolor='#784212', label='abstained positives'),
        Line2D([], [], marker='D', ls='', markersize=6, markerfacecolor=PAL['aneg'],
               markeredgecolor='#9a7d0a', label='abstained negatives'),
        Line2D([], [], marker='h', ls='', markersize=7, markerfacecolor=PAL['ood'],
               markeredgecolor='#2f3a3a', label='abstained by OOD gate'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=4,
               frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(f"FENCE exact decision space   |   backbone: {FENCE_BACKBONE}",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0.045, 1, 0.96])

    out_base = Path(args.out) if args.out else (
        BASE_DIR / "results" / "figures" /
        f"fence_decision_grid_{FENCE_BACKBONE}_{args.split}")
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_base) + ".png", dpi=300, bbox_inches='tight')
    fig.savefig(str(out_base) + ".pdf", bbox_inches='tight')
    print(f"\nSaved: {out_base}.png")
    print(f"Saved: {out_base}.pdf")

    print("\nSuggested caption:")
    print("Exact decision space of FENCE on the held-out split of each benchmark.")
    print("Every point is plotted by its signed margin to the nearest enclosure")
    print("ball (x axis, positive means inside a ball) and its signed margin to the")
    print("negative cloud (y axis, positive means inside the cloud), so each point")
    print("lies in the region that matches the deployed decision rule with no")
    print("projection distortion: right half-plane is predicted positive, top-left")
    print("is predicted negative, bottom-left is abstained for human review. Axes")
    print("use a symmetric-log scale to expand the dense boundary region. Gray")
    print("hexagons are inputs that lay inside the negative cloud but were routed")
    print("to abstention by the out-of-distribution gate.")


if __name__ == "__main__":
    main()
