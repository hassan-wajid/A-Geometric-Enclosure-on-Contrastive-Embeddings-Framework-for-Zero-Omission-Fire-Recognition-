"""
FENCE TRAINING ABLATIONS — the ones that need Stage-1 retraining.

Separate from eval_ablations.py (which is eval-only). This trains SHORT Stage-1
variants IN MEMORY and evaluates them. It NEVER writes a model checkpoint and
NEVER touches your original Stage-1/Stage-2 checkpoints — only JSON results are
saved under results/ablations_train/.

Skips fasdd (largest dataset) by default. Trains on ViT by default (the headline
backbone); add ConvNeXt with --backbones if wanted.

Stage-1 representation variants (all trained for --epochs, same budget → fair):
    raw_clip          : NO training — frozen pretrained backbone features
    reference_full    : your EXISTING 150-epoch checkpoint (no training, from disk)
    msc               : SupCon + compactness (β=0.5)   [the method]
    contrastive_only  : SupCon only (β=0)  — mean-shift/compactness OFF
    cross_entropy     : BCE on the classifier logits (no contrastive)

Each variant -> extract embeddings -> fit union-of-balls -> abstain rule -> metrics.

Run:
    python code/fence/eval_train_ablations.py                     # ustc,dfire,flame × ViT
    python code/fence/eval_train_ablations.py --epochs 30
    python code/fence/eval_train_ablations.py --datasets dfire --backbones vit_large_patch14_224
"""

import os
import sys
from pathlib import Path
os.environ['OMP_NUM_THREADS'] = '1'
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from fence.config import (
    DEVICE, FENCE_NUM_CLUSTERS, FENCE_MAX_CLUSTERS, FENCE_MIN_CLUSTER_SIZE,
    FENCE_RADIUS_MARGIN, MSC_TEMPERATURE, MSC_LAMBDA, USE_AMP,
    FENCE_WEIGHT_DECAY, SPECTRAL_NORM_ENABLED,
)
from fence.core.fence_model import FENCEModel
from fence.stage1_pretrain import stage1_loss, WarmupCosineWithRestarts
from fence.stage2_train import get_dataset_paths
from train_datasets.dataset_factory import create_dataset, create_dataloaders, auto_splits

# reuse the eval-side machinery so ablations use the identical decision rule
from fence.eval_ablations import (
    BACKBONES, stage1_ckpt, load_model, extract,
    fit_geometry, build_neg_ref, ood_flags, decide, metrics,
)

REPO = Path(__file__).parent.parent.parent
OUT_DIR = REPO / "results" / "ablations_train"
TRAIN_DATASETS = ["ustc_smokers", "dfire", "flame"]      # NOT fasdd


def fresh_model(backbone, dim, device):
    """A new FENCEModel with pretrained backbone (NOT loaded from any checkpoint)."""
    return FENCEModel(backbone_name=backbone, embedding_dim=dim,
                      num_balls=FENCE_NUM_CLUSTERS, max_balls=FENCE_MAX_CLUSTERS,
                      min_cluster_size=FENCE_MIN_CLUSTER_SIZE, margin=FENCE_RADIUS_MARGIN,
                      temperature=MSC_TEMPERATURE, lambda_class=MSC_LAMBDA,
                      use_mean_shift=True, spectral_norm=SPECTRAL_NORM_ENABLED,
                      device=device)


def train_variant(model, train_samples, device, epochs, loss_mode, beta):
    """Short in-memory Stage-1 training (faithful loss + schedule). No checkpoint."""
    loader = create_dataloaders({"train": train_samples})["train"]   # 2-view, augmented
    bb, hd = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (bb if "clip_model" in n else hd).append(p)
    groups = []
    if bb: groups.append({"params": bb, "lr_scale": 0.1})
    if hd: groups.append({"params": hd, "lr_scale": 1.0})
    opt = optim.AdamW(groups, lr=1e-6, weight_decay=FENCE_WEIGHT_DECAY)
    sched = WarmupCosineWithRestarts(optimizer=opt, warmup_epochs=5, peak_lr=5e-4,
                                     min_lr=1e-6, restart_epochs=20, total_epochs=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    bce = nn.BCEWithLogitsLoss()

    model.train()
    for ep in range(epochs):
        sched.step(ep)
        run = 0.0; nb = 0
        from tqdm import tqdm
        for batch in tqdm(loader, desc=f"  {loss_mode} ep{ep+1}/{epochs}", ncols=80, leave=False):
            if len(batch) == 4:
                i1, i2, lab, _ = batch
            else:
                i1, lab = batch; i2 = i1
            imgs = torch.cat([i1, i2], 0).to(device)
            labc = torch.cat([lab, lab], 0).to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                out = model(imgs)
            emb, logits = out["embeddings"], out["logits"]
            if loss_mode == "cross_entropy":
                loss = bce(logits.squeeze(), labc.float())
            else:                                    # msc / contrastive_only
                loss, _, _ = stage1_loss(emb, labc, temperature=0.1, compactness_beta=beta)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            run += loss.item(); nb += 1
        print(f"  {loss_mode} epoch {ep+1}/{epochs}  loss={run/max(nb,1):.4f}")
    model.eval()
    return model


@torch.no_grad()
@torch.no_grad()
def separation(emb, lab, device, cap=2000):
    """Embedding quality: mean within-class cosine minus between-class cosine.
    Higher = fire and non-fire clusters are further apart (cleaner space)."""
    e = torch.nn.functional.normalize(emb.to(device).float(), dim=1)
    y = lab.to(device)
    pos, neg = e[y == 1], e[y == 0]
    g = torch.Generator(device=e.device).manual_seed(0)
    if len(pos) > cap:
        pos = pos[torch.randperm(len(pos), generator=g, device=e.device)[:cap]]
    if len(neg) > cap:
        neg = neg[torch.randperm(len(neg), generator=g, device=e.device)[:cap]]
    intra = 0.5 * ((pos @ pos.T).mean().item() + (neg @ neg.T).mean().item())
    inter = (pos @ neg.T).mean().item()
    return {"intra_cos": round(intra, 4), "inter_cos": round(inter, 4),
            "separation": round(intra - inter, 4)}


def fpr_at_matched_coverage(tr_emb, tr_lab, te_emb, te_lab, geom, ood, device,
                            targets=(0.90, 0.95, 0.99)):
    """Sweep the abstention knob (τ) to hit each target coverage, then report the
    selective FPR THERE. Equalizes 'how much each model refused' so the false-alarm
    comparison is fair (no model looks good just by abstaining more)."""
    y = te_lab.numpy()
    curve = []
    for q in (0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 0.999):
        nref, tau, _ = build_neg_ref(tr_emb, tr_lab, device, neg_q=q)
        pred, abst = decide(te_emb, geom, nref, tau, ood, tr_emb, tr_lab, device)
        curve.append(metrics(y, pred, abst))
    out = {}
    for t in targets:
        best = min(curve, key=lambda m: abs(m["coverage"] - t))
        out[f"cov{t}"] = {"fpr": best["selective_fpr"], "fnr": best["selective_fnr"],
                          "actual_coverage": best["coverage"]}
    return out


def eval_model(model, tr_s, te_s, device, K):
    tr_emb, tr_lab = extract(model, tr_s, device, "train")
    te_emb, te_lab = extract(model, te_s, device, "test")
    geom = fit_geometry(tr_emb, tr_lab, device, K=K, margin=FENCE_RADIUS_MARGIN)
    neg_ref, tau, _ = build_neg_ref(tr_emb, tr_lab, device, neg_q=0.99)
    ood = ood_flags(te_emb, tr_emb, device)
    pred, abst = decide(te_emb, geom, neg_ref, tau, ood, tr_emb, tr_lab, device)
    m = metrics(te_lab.numpy(), pred, abst)
    m["n_personal"] = geom["n_personal"]; m["tau"] = tau
    # --- the two metrics that expose the BCE-vs-contrastive difference ---
    m["stage1_separation"] = separation(te_emb, te_lab, device)
    m["fpr_at_matched_coverage"] = fpr_at_matched_coverage(
        tr_emb, tr_lab, te_emb, te_lab, geom, ood, device)
    return m


def run_cell(dataset, backbone, dim, device, epochs, K):
    print(f"\n{'='*70}\n  TRAIN-ABLATION CELL: {dataset} / {backbone}\n{'='*70}")
    ds = create_dataset(dataset, *get_dataset_paths(dataset))
    splits = auto_splits(ds)
    tr_s, te_s = splits["train"], splits.get("test") or splits["val"]
    R = {"cell": {"dataset": dataset, "backbone": backbone, "epochs": epochs}}

    # raw CLIP (no training)
    m = fresh_model(backbone, dim, device)
    R["raw_clip"] = eval_model(m, tr_s, te_s, device, K); del m

    # existing full checkpoint (no training)
    s1 = stage1_ckpt(backbone, dataset)
    if s1.exists():
        m, _ = load_model(backbone, dim, s1, device)
        R["reference_full_150ep"] = eval_model(m, tr_s, te_s, device, K); del m

    # trained variants (fresh model each; nothing saved)
    for name, mode, beta in [("msc", "msc", 0.5),
                             ("contrastive_only", "msc", 0.0),
                             ("cross_entropy", "cross_entropy", 0.0)]:
        m = fresh_model(backbone, dim, device)
        m = train_variant(m, tr_s, device, epochs, mode, beta)
        R[name] = eval_model(m, tr_s, te_s, device, K)
        del m
        if device == "cuda":
            torch.cuda.empty_cache()
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=TRAIN_DATASETS)
    ap.add_argument("--backbones", nargs="*", default=["vit_large_patch14_224"])
    ap.add_argument("--epochs", type=int, default=40,
                    help="Stage-1 epochs per trained variant (fixed budget for fairness)")
    ap.add_argument("--k", type=int, default=2, help="ball count for geometry fit")
    ap.add_argument("--out", default=None,
                    help="output folder name under results/ (default: ablations_train). "
                         "Use a new name to avoid overwriting old results.")
    args = ap.parse_args()

    global OUT_DIR
    if args.out:
        OUT_DIR = REPO / "results" / args.out
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Saving to: {OUT_DIR.resolve()}")
    allres = {}
    for bk in args.backbones:
        for dsname in args.datasets:
            if dsname == "fasdd":
                print("  skipping fasdd (too big for training ablations)"); continue
            key = f"{dsname}__{bk}"
            try:
                R = run_cell(dsname, bk, BACKBONES[bk], DEVICE, args.epochs, args.k)
            except Exception as e:
                import traceback; traceback.print_exc(); R = {"error": str(e)}
            allres[key] = R
            with open(OUT_DIR / f"train_ablation_{key}.json", "w") as f:
                json.dump(R, f, indent=2)
            print(f"  saved {OUT_DIR / f'train_ablation_{key}.json'}")

    with open(OUT_DIR / "train_ablations_ALL.json", "w") as f:
        json.dump(allres, f, indent=2)
    print(f"\n✅ done. combined: {OUT_DIR / 'train_ablations_ALL.json'}")


if __name__ == "__main__":
    main()