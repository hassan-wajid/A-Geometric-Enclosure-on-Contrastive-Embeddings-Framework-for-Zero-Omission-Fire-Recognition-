"""
COMPREHENSIVE FENCE evaluation over the full grid (4 datasets x 2 backbones).

For every cell it records, from the STORED checkpoints (no training):
  * Stage-1  : embedding separability (intra/inter cosine, LinearProbe, kNN AUC)
  * Stage-2  : enclosure geometry (main balls, radii, personal balls, tau, margin)
  * Final    : test-set counts (TP/FN/FP/TN + abstain) and FNR/FPR/coverage
  * Frontier : tau sweep (0.85/0.90/0.95/0.99)  -- default rule, no veto/pos-cap

Writes ONE json per cell:  <out_dir>/eval_<dataset>_<backbone>.json
plus a combined:           <out_dir>/ALL_results.json
plus a flat summary table: <out_dir>/summary_table.csv

Usage:
    python code/fence/eval_all.py
    python code/fence/eval_all.py --out_dir results/final_eval
    python code/fence/eval_all.py --datasets dfire flame --backbones vit_large_patch14_224
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
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from fence.config import DEVICE, MSC_TEMPERATURE, MSC_LAMBDA, USE_AMP
from fence.core.fence_model import FENCEModel
from fence.stage2_train import get_dataset_paths
from fence.geometry.ball import Ball
from train_datasets.dataset_factory import create_dataset, create_dataloaders, auto_splits

# ---- grid definition -------------------------------------------------------
BACKBONES = {                      # backbone_name -> embedding_dim
    "vit_large_patch14_224": 768,
    "convnext_v2_large": 1536,
}
DATASETS = ["ustc_smokers", "flame", "dfire", "fasdd"]

# The checkpoints live under: <root>/<backbone>/<dataset>/<backbone>/stageN/*.pt
DEFAULT_CKPT_ROOT = ("E:/WajidHM/Forest Fire Detection/"
                     "Zero-Omission-Decision-Boundary-Optimization-Methods-for-"
                     "High-Risk-Image-Recognition/checkpoints/fence/flame")

TAU_QS = (0.85, 0.90, 0.95, 0.99)


# ---- helpers ---------------------------------------------------------------
def cell_paths(root, backbone, dataset):
    base = Path(root) / backbone / dataset / backbone
    return base / "stage2" / "final_model.pt", base / "stage1" / "pretrained.pt"


def restore(model, state, device):
    u = model.union_of_balls
    u.balls = [Ball(torch.tensor(b["center"], device=device),
                    radius=b["radius"], margin=b["margin"])
               for b in state["union_of_balls"].get("balls", [])]
    u.margin = state["union_of_balls"].get("margin", u.margin)
    model.quarantine_balls = [
        Ball(torch.tensor(b["center"], device=device),
             radius=b["radius"], margin=b["margin"])
        for b in state.get("quarantine_balls", [])]


@torch.no_grad()
def extract(model, loader, device, desc):
    model.eval()
    E, Y = [], []
    for batch in tqdm(loader, desc=desc, ncols=80, leave=False):
        imgs = batch[0].to(device)
        y = batch[2] if len(batch) == 4 else batch[1]
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            z = model.get_embeddings(imgs)
        E.append(z.float().cpu()); Y.append(y.cpu())
    return torch.cat(E), torch.cat(Y)


def _neg_nn_quantile(neg, q, device, cap=4000):
    """Quantile of negative->nearest-other-negative distances."""
    neg = neg.to(device)
    if len(neg) > cap:
        neg = neg[torch.randperm(len(neg), device=device)[:cap]]
    D = torch.cdist(neg, neg)
    D.fill_diagonal_(float("inf"))
    return torch.quantile(D.min(dim=1).values, q).item()


@torch.no_grad()
def test_signals(model, emb, device, chunk=1024):
    """Per-test-point tau-INDEPENDENT signals: inside-ball, ood, d_to_nearest_neg."""
    inside_l, ood_l, dneg_l = [], [], []
    neg_ref = model._train_neg_embeddings
    for i in range(0, len(emb), chunk):
        e = emb[i:i + chunk].to(device)
        s = model.union_of_balls.distance_to_nearest(e)
        inside = s >= 0
        for b in getattr(model, "quarantine_balls", []):
            inside = inside | b.contains(e)
        dneg = torch.cdist(e, neg_ref).min(dim=1).values
        ood = model._is_ood(e)
        inside_l.append(inside.cpu().numpy())
        ood_l.append(ood.cpu().numpy())
        dneg_l.append(dneg.cpu().numpy())
    return (np.concatenate(inside_l).astype(bool),
            np.concatenate(ood_l).astype(bool),
            np.concatenate(dneg_l))


def decide(inside, ood, dneg, tau):
    """Apply the abstain-by-default rule from tau-independent signals."""
    conf_neg = dneg <= tau
    pred_neg = conf_neg & (~inside) & (~ood)
    abst = (~inside) & (~pred_neg)
    pred = inside.astype(int)          # predicted POSITIVE iff inside a ball
    return pred, abst


def counts(y, pred, abst):
    keep = ~abst
    yk, pk = y[keep], pred[keep]
    tp = int(((yk == 1) & (pk == 1)).sum())
    fn = int(((yk == 1) & (pk == 0)).sum())
    fp = int(((yk == 0) & (pk == 1)).sum())
    tn = int(((yk == 0) & (pk == 0)).sum())
    ap = int(((y == 1) & abst).sum())
    an = int(((y == 0) & abst).sum())
    cov_p = max((y == 1).sum() - ap, 1)
    cov_n = max((y == 0).sum() - an, 1)
    return dict(
        TP=tp, FN=fn, FP=fp, TN=tn, abstain_pos=ap, abstain_neg=an,
        selective_fnr=fn / cov_p, selective_fpr=fp / cov_n,
        coverage=float(keep.mean()), abstention=float(abst.mean()),
        raw_fnr=float(((y == 1) & (pred == 0)).sum() / max((y == 1).sum(), 1)),
        raw_fpr=float(((y == 0) & (pred == 1)).sum() / max((y == 0).sum(), 1)),
    )


def stage1_diag(emb, lab, cap=3000):
    """Embedding separability on a balanced subsample (representation quality)."""
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import roc_auc_score, f1_score
    y = lab.long().numpy()
    Z = emb
    if len(Z) > cap:                                   # balanced subsample
        idx = []
        rng = np.random.default_rng(0)
        for c in (0, 1):
            ci = np.where(y == c)[0]
            idx.append(rng.choice(ci, min(cap // 2, len(ci)), replace=False))
        idx = np.concatenate(idx)
        Z, y = Z[idx], y[idx]
    Zt = F.normalize(Z, dim=1)
    sim = (Zt @ Zt.T).numpy()
    N = len(y); eye = np.eye(N, dtype=bool)
    same = (y[:, None] == y[None, :]) & ~eye
    diff = (y[:, None] != y[None, :])
    out = dict(n=int(N), n_pos=int((y == 1).sum()), n_neg=int((y == 0).sum()),
               intra_cos=float(sim[same].mean()), inter_cos=float(sim[diff].mean()),
               separation=float(sim[same].mean() - sim[diff].mean()))
    Zn = Zt.numpy()
    Xtr, Xte, ytr, yte = train_test_split(Zn, y, test_size=0.3, random_state=42, stratify=y)
    for name, clf in [("linear_probe", LogisticRegression(max_iter=2000, class_weight="balanced")),
                      ("knn5", KNeighborsClassifier(n_neighbors=5, metric="cosine"))]:
        clf.fit(Xtr, ytr); p = clf.predict(Xte)
        try:
            auc = float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))
        except Exception:
            auc = float("nan")
        out[name] = dict(acc=float((p == yte).mean()), f1=float(f1_score(yte, p)), auc=auc)
    return out


def read_json(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


# ---- per-cell evaluation ---------------------------------------------------
def eval_cell(dataset, backbone, dim, root, device):
    final_p, stage1_p = cell_paths(root, backbone, dataset)
    if not final_p.exists():
        return {"error": f"missing checkpoint: {final_p}"}

    state = torch.load(final_p, map_location=device, weights_only=False)
    sd = state["model_state_dict"]
    sn = any("weight_orig" in k for k in sd)
    model = FENCEModel(backbone_name=backbone, embedding_dim=dim,
                       num_balls=0, margin=0.1, temperature=MSC_TEMPERATURE,
                       lambda_class=MSC_LAMBDA, use_mean_shift=True,
                       spectral_norm=sn, device=device)
    model.load_state_dict(sd)
    restore(model, state, device)

    ds = create_dataset(dataset, *get_dataset_paths(dataset))
    splits = auto_splits(ds)
    test_samples = splits["test"] or splits["val"]
    # clean transforms (NON-'train' keys avoid augmentation + shuffle)
    loaders = create_dataloaders({"ref": splits["train"], "test": test_samples},
                                 test_mode=False)

    tr_emb, tr_lab = extract(model, loaders["ref"], device, f"{dataset}/{backbone} train")
    model.store_ood_reference(tr_emb.to(device), tr_lab.to(device))   # default rule (pos-cap OFF)
    te_emb, te_lab = extract(model, loaders["test"], device, f"{dataset}/{backbone} test")
    y = te_lab.numpy().astype(int)

    inside, ood, dneg = test_signals(model, te_emb, device)
    tau0 = float(model._neg_threshold)
    pred, abst = decide(inside, ood, dneg, tau0)
    main = counts(y, pred, abst)

    # tau sweep (inside/ood are tau-independent -> exact, no re-embedding)
    neg_tr = tr_emb[tr_lab == 0]
    sweep = []
    for q in TAU_QS:
        tq = _neg_nn_quantile(neg_tr, q, device)
        p_, a_ = decide(inside, ood, dneg, tq)
        c = counts(y, p_, a_)
        sweep.append({"q": q, "tau": tq, "FN": c["FN"], "FP": c["FP"],
                      "abstain_pos": c["abstain_pos"], "abstain_neg": c["abstain_neg"],
                      "selective_fnr": c["selective_fnr"], "selective_fpr": c["selective_fpr"],
                      "coverage": c["coverage"]})

    geom = {
        "n_main_balls": len(model.union_of_balls.balls),
        "n_personal_balls": len(model.quarantine_balls),
        "main_radii": [float(b.radius) for b in model.union_of_balls.balls],
        "margin": float(model.union_of_balls.margin),
        "tau_default": tau0,
        "tau_coverage_term": getattr(model, "_tau_coverage", None),
        "verified_zero_fnr_at_finalize": state.get("verified", None),
    }

    s1 = stage1_diag(tr_emb, tr_lab)
    s2_summary = read_json(final_p.parent / "stage2_summary.json")
    s1_history = read_json(stage1_p.parent / "history.json")

    return {
        "dataset": dataset, "backbone": backbone, "embedding_dim": dim,
        "n_test": int(len(y)), "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum()),
        "spectral_norm": bool(sn),
        "stage1_diagnostics": s1,
        "stage1_epochs": (len(s1_history.get("loss", [])) if s1_history else None),
        "stage2_geometry": geom,
        "stage2_summary": s2_summary,
        "final_eval_default_tau": main,
        "tau_sweep": sweep,
        "checkpoint": str(final_p),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_root", default=DEFAULT_CKPT_ROOT)
    ap.add_argument("--out_dir", default="results/final_eval")
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--backbones", nargs="+", default=list(BACKBONES.keys()))
    args = ap.parse_args()
    device = DEVICE

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing results to: {out_dir.resolve()}")

    all_results, rows = [], []
    for dataset in args.datasets:
        for backbone in args.backbones:
            dim = BACKBONES[backbone]
            tag = f"{dataset}_{backbone}"
            print(f"\n{'='*70}\n  {tag}\n{'='*70}")
            try:
                res = eval_cell(dataset, backbone, dim, args.ckpt_root, device)
            except Exception as e:
                res = {"dataset": dataset, "backbone": backbone, "error": repr(e)}
                print(f"  ERROR: {e}")
            with open(out_dir / f"eval_{tag}.json", "w") as f:
                json.dump(res, f, indent=2)
            all_results.append(res)

            if "error" not in res:
                m = res["final_eval_default_tau"]; g = res["stage2_geometry"]
                s1 = res["stage1_diagnostics"]
                print(f"  TP={m['TP']} FN={m['FN']} FP={m['FP']} TN={m['TN']} "
                      f"abst(p/n)={m['abstain_pos']}/{m['abstain_neg']}  "
                      f"selFNR={m['selective_fnr']:.4f} selFPR={m['selective_fpr']:.4f} "
                      f"cov={m['coverage']:.4f}")
                print(f"  balls={g['n_main_balls']} personal={g['n_personal_balls']} "
                      f"tau={g['tau_default']:.5f}  s1_sep={s1['separation']:.4f} "
                      f"s1_knnAUC={s1['knn5']['auc']:.4f}")
                rows.append((dataset, backbone, m, g, s1))

    with open(out_dir / "ALL_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # flat CSV summary
    with open(out_dir / "summary_table.csv", "w") as f:
        f.write("dataset,backbone,n_test,pos,neg,TP,FN,FP,TN,abst_pos,abst_neg,"
                "sel_FNR,sel_FPR,coverage,main_balls,personal_balls,tau,"
                "s1_separation,s1_knn_auc\n")
        for r in all_results:
            if "error" in r:
                continue
            m, g, s1 = r["final_eval_default_tau"], r["stage2_geometry"], r["stage1_diagnostics"]
            f.write(f"{r['dataset']},{r['backbone']},{r['n_test']},{r['n_pos']},{r['n_neg']},"
                    f"{m['TP']},{m['FN']},{m['FP']},{m['TN']},{m['abstain_pos']},{m['abstain_neg']},"
                    f"{m['selective_fnr']:.6f},{m['selective_fpr']:.6f},{m['coverage']:.6f},"
                    f"{g['n_main_balls']},{g['n_personal_balls']},{g['tau_default']:.6f},"
                    f"{s1['separation']:.6f},{s1['knn5']['auc']:.6f}\n")

    print(f"\n{'='*70}\n  DONE. Per-cell JSON + ALL_results.json + summary_table.csv in {out_dir}\n{'='*70}")


if __name__ == "__main__":
    main()