"""
Stage 1 diagnostic: measure how well the learned embeddings separate the two
classes. SupCon's absolute value has a high floor under class imbalance, so
Stage 1 should be judged by embedding *separability*, not by the loss number.

Usage:
    python code/fence/eval_stage1_embeddings.py --dataset ustc_smokers
"""

import os
import sys
from pathlib import Path
os.environ['OMP_NUM_THREADS'] = '1'
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
import argparse

from fence.config import (
    DEVICE, FENCE_BACKBONE, FENCE_EMBEDDING_DIM, FENCE_CHECKPOINT_DIR,
    SPECTRAL_NORM_ENABLED, MSC_TEMPERATURE, MSC_LAMBDA
)
from fence.core.fence_model import FENCEModel
from fence.stage1_pretrain import get_dataset_paths
from train_datasets.dataset_factory import create_dataset, create_dataloaders


@torch.no_grad()
def extract(model, loader, device):
    model.eval()
    embs, labs = [], []
    for batch in loader:
        v1 = batch[0].to(device)          # (v1, v2, label, idx) — v1==v2 for val transform
        y = batch[2] if len(batch) == 4 else batch[1]
        with torch.cuda.amp.autocast(enabled=True):
            z = model.get_embeddings(v1)
        embs.append(z.float().cpu())
        labs.append(y.cpu())
    return torch.cat(embs), torch.cat(labs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ustc_smokers")
    ap.add_argument("--ckpt", default=None, help="path to pretrained.pt (default: standard location)")
    args = ap.parse_args()

    device = DEVICE
    ckpt_path = Path(args.ckpt) if args.ckpt else (
        FENCE_CHECKPOINT_DIR / args.dataset / FENCE_BACKBONE / "stage1" / "pretrained.pt")
    print(f"Loading checkpoint: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = state['model_state_dict']

    # The checkpoint's saved 'config' doesn't record whether spectral_norm was
    # applied, and spectral_norm renames params (weight -> weight_orig/_u/_v).
    # Detect it directly from the state dict so the model we build here always
    # matches whatever checkpoint we're given, regardless of current config.py.
    ckpt_has_sn = any('weight_orig' in k for k in sd.keys())
    if ckpt_has_sn != SPECTRAL_NORM_ENABLED:
        print(f"  Note: checkpoint was saved with spectral_norm={ckpt_has_sn} "
              f"(current config says {SPECTRAL_NORM_ENABLED}) -> using {ckpt_has_sn}")

    model = FENCEModel(
        backbone_name=FENCE_BACKBONE, embedding_dim=FENCE_EMBEDDING_DIM,
        num_balls=0, margin=0.1, temperature=MSC_TEMPERATURE,
        lambda_class=MSC_LAMBDA, use_mean_shift=True,
        spectral_norm=ckpt_has_sn, device=device,
    )
    model.load_state_dict(sd)

    image_dir, ann_dir = get_dataset_paths(args.dataset)
    ds = create_dataset(args.dataset, image_dir, ann_dir)
    splits = ds.split_samples(ds.parse_samples())
    loaders = create_dataloaders({'val': splits['val']}, test_mode=False)

    Z, y = extract(model, loaders['val'], device)
    y = y.long().numpy()
    Zt = F.normalize(Z, dim=1)

    # --- 1) intra- vs inter-class cosine similarity ---
    sim = (Zt @ Zt.T).numpy()
    N = len(y)
    eye = np.eye(N, dtype=bool)
    same = (y[:, None] == y[None, :]) & ~eye
    diff = (y[:, None] != y[None, :])
    print(f"\nVal samples: {N}  (pos={int((y == 1).sum())}, neg={int((y == 0).sum())})")
    print(f"  mean cosine  intra-class : {sim[same].mean():.4f}")
    print(f"  mean cosine  inter-class : {sim[diff].mean():.4f}")
    print(f"  separation (intra-inter) : {sim[same].mean() - sim[diff].mean():.4f}   (higher = better)")

    # --- 2) linear probe + kNN on the frozen embeddings ---
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import roc_auc_score, f1_score

    Zn = Zt.numpy()
    Xtr, Xte, ytr, yte = train_test_split(Zn, y, test_size=0.3, random_state=42, stratify=y)
    lr = LogisticRegression(max_iter=2000, class_weight='balanced').fit(Xtr, ytr)
    knn = KNeighborsClassifier(n_neighbors=5, metric='cosine').fit(Xtr, ytr)

    print()
    for name, clf in [("LinearProbe", lr), ("kNN-5     ", knn)]:
        p = clf.predict(Xte)
        try:
            s = clf.predict_proba(Xte)[:, 1]
            auc = roc_auc_score(yte, s)
        except Exception:
            auc = float('nan')
        print(f"  {name}  acc={(p == yte).mean():.4f}  f1={f1_score(yte, p):.4f}  auc={auc:.4f}")

    print("\nInterpretation: AUC > ~0.9 and clearly positive separation => Stage 1 is working,")
    print("and the SupCon ~3.65 is just the loss floor for this class imbalance.")


if __name__ == "__main__":
    main()