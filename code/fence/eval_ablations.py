"""
FENCE ABLATION SUITE — one file, all ablations, results to JSON.

SAFETY: this script NEVER writes a model .pt checkpoint. All geometry variants
are re-fit IN MEMORY from cached embeddings; only JSON results (and an optional
embedding cache) are written under results/ablations/. Your original Stage-1 /
Stage-2 checkpoints are never modified.

What it runs per cell (dataset × backbone), in order:

  TIER 1 (core novelty)
    T1a  personal (quarantine) balls  ON  vs OFF        [eval-only]
    T1b  abstain-by-default           vs hard classifier [eval-only]
    T1c  full model (Stage-2 head)    vs geometry-only (Stage-1 head, no primal-dual)
    T1d  radius rule: dual-quantile / max / positive-only / fixed
    T1e  union-of-balls               vs single ball (K=1)

  TIER 3 (component contributions)
    T3a  kNN veto  OFF vs ON
    T3b  OOD gate  ON  vs OFF
    T3c  neg-cloud τ vs plain distance threshold
    T3d  positive-capped τ OFF vs ON

  TIER 2 (hyperparameter sensitivity)
    T2a  τ sweep  0.85 / 0.90 / 0.95 / 0.99
    T2b  number of balls K = 1 / 2 / 4 / 6 / 8
    T2c  margin m = 0.05 / 0.1 / 0.2

Run:
    python code/fence/eval_ablations.py                 # all 8 cells, all ablations
    python code/fence/eval_ablations.py --quick         # ustc_smokers + dfire only
    python code/fence/eval_ablations.py --datasets dfire --backbones vit_large_patch14_224
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
from tqdm import tqdm
from sklearn.cluster import KMeans

from fence.config import (
    DEVICE, FENCE_EMBEDDING_DIM, FENCE_NUM_CLUSTERS, FENCE_MAX_CLUSTERS,
    FENCE_MIN_CLUSTER_SIZE, FENCE_RADIUS_MARGIN, MSC_TEMPERATURE, MSC_LAMBDA,
    USE_AMP, SEED, FENCE_RADIUS_QUANTILE,
)
from fence.core.fence_model import FENCEModel
from fence.stage2_train import get_dataset_paths
from train_datasets.dataset_factory import create_dataset, create_dataloaders, auto_splits

# ---- checkpoint layout (from your `find` output) -----------------------------
CKPT_ROOT = Path(r"E:\WajidHM\Forest Fire Detection"
                 r"\Zero-Omission-Decision-Boundary-Optimization-Methods-for-High-Risk-Image-Recognition"
                 r"\checkpoints\fence\flame")
BACKBONES = {"vit_large_patch14_224": 768, "convnext_v2_large": 1536}
DATASETS  = ["ustc_smokers", "flame", "dfire", "fasdd"]

REPO = Path(__file__).parent.parent.parent
OUT_DIR = REPO / "results" / "ablations"
CACHE_DIR = OUT_DIR / "emb_cache"


def final_ckpt(backbone, dataset):
    return CKPT_ROOT / backbone / dataset / backbone / "stage2" / "final_model.pt"

def stage1_ckpt(backbone, dataset):
    return CKPT_ROOT / backbone / dataset / backbone / "stage1" / "pretrained.pt"


# ============================ distance helpers ================================
def _min_dist(A, B, chunk=1024):
    """min distance of each row of A to set B (row-chunked)."""
    out = []
    for i in range(0, len(A), chunk):
        out.append(torch.cdist(A[i:i + chunk], B).min(dim=1).values)
    return torch.cat(out)


def _knn_labels(A, B, B_lab, k, chunk=1024):
    """For each row of A, labels of its k nearest neighbours in B (row-chunked)."""
    out = []
    for i in range(0, len(A), chunk):
        d = torch.cdist(A[i:i + chunk], B)
        idx = torch.topk(d, k=min(k, B.shape[0]), dim=1, largest=False).indices
        out.append(B_lab[idx])
    return torch.cat(out)


def _kth_nn_self(B, k, chunk=1024):
    """k-th NN distance of each point in B to the rest (self excluded), chunked."""
    out = []
    for i in range(0, len(B), chunk):
        d = torch.cdist(B[i:i + chunk], B)
        rows = torch.arange(d.shape[0], device=d.device)
        d[rows, i + rows] = float('inf')
        out.append(torch.topk(d, k=min(k, B.shape[0]), dim=1, largest=False).values[:, -1])
    return torch.cat(out)


# ============================ model / embeddings ==============================
def load_model(backbone, dim, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = state["model_state_dict"]
    sn = any("weight_orig" in k for k in sd)
    model = FENCEModel(backbone_name=backbone, embedding_dim=dim,
                       num_balls=FENCE_NUM_CLUSTERS, max_balls=FENCE_MAX_CLUSTERS,
                       min_cluster_size=FENCE_MIN_CLUSTER_SIZE, margin=FENCE_RADIUS_MARGIN,
                       temperature=MSC_TEMPERATURE, lambda_class=MSC_LAMBDA,
                       use_mean_shift=True, spectral_norm=sn, device=device)
    model.load_state_dict(sd)
    model.eval()
    return model, state


@torch.no_grad()
def extract(model, samples, device, desc):
    loader = create_dataloaders({"eval": samples}, test_mode=False)["eval"]
    E, Y = [], []
    for batch in tqdm(loader, desc=desc, ncols=80, leave=False):
        imgs = batch[0].to(device)
        y = batch[2] if len(batch) == 4 else batch[1]
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            z = model.get_embeddings(imgs)
        E.append(z.float().cpu()); Y.append(y.cpu())
    return torch.cat(E), torch.cat(Y)


def cached_extract(model, samples, device, tag):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    f = CACHE_DIR / f"{tag}.pt"
    if f.exists():
        d = torch.load(f, map_location="cpu")
        return d["E"], d["Y"]
    E, Y = extract(model, samples, device, tag)
    torch.save({"E": E, "Y": Y}, f)
    return E, Y


# ============================ geometry fitting ================================
def geometry_from_checkpoint(state, device):
    """Original deployed geometry (centers/radii + personal balls) as tensors."""
    u = state["union_of_balls"]
    centers = torch.tensor(np.array([b["center"] for b in u.get("balls", [])]),
                           dtype=torch.float32, device=device)
    radii = torch.tensor([b["radius"] for b in u.get("balls", [])],
                         dtype=torch.float32, device=device)
    q = state.get("quarantine_balls", []) or []
    if q:
        pc = torch.tensor(np.array([b["center"] for b in q]),
                          dtype=torch.float32, device=device)
        pr = torch.tensor([b["radius"] for b in q], dtype=torch.float32, device=device)
    else:
        pc = torch.zeros((0, centers.shape[1]), device=device)
        pr = torch.zeros((0,), device=device)
    margin = u.get("margin", FENCE_RADIUS_MARGIN)
    return dict(centers=centers, radii=radii, pcenters=pc, pradii=pr, margin=margin)


def fit_geometry(train_emb, train_lab, device, K, margin,
                 radius_rule="dual_quantile", use_personal=True,
                 deploy_q=0.95, neg_safe_q=0.01):
    """Re-fit union-of-balls IN MEMORY (no checkpoint written)."""
    pos = train_emb[train_lab == 1].to(device)
    neg = train_emb[train_lab == 0].to(device)
    Kf = max(1, min(K, len(pos)))
    km = KMeans(n_clusters=Kf, random_state=SEED, n_init=10).fit(pos.cpu().numpy())
    centers = torch.tensor(km.cluster_centers_, dtype=torch.float32, device=device)
    assign = torch.tensor(km.labels_, device=device)

    radii = torch.full((Kf,), margin, device=device)
    for j in range(Kf):
        m = assign == j
        if not m.any():
            continue
        pd = torch.norm(pos[m] - centers[j], dim=1)
        if radius_rule == "max":
            radii[j] = pd.max()
        elif radius_rule == "positive_only":
            radii[j] = torch.quantile(pd, deploy_q)
        elif radius_rule == "fixed":
            radii[j] = margin
        else:  # dual_quantile (the method)
            r_pos = torch.quantile(pd, deploy_q)
            if len(neg) > 0:
                nd = torch.norm(neg - centers[j], dim=1)
                r_neg = torch.quantile(nd, neg_safe_q) - margin
                radii[j] = torch.clamp(torch.minimum(r_pos, r_neg), min=1e-6)
            else:
                radii[j] = r_pos

    # personal balls for uncovered positives
    pc_list, pr_list = [], []
    if use_personal:
        s = (radii[None, :] - torch.cdist(pos, centers)).max(dim=1).values
        unc = torch.where(s < 0)[0]
        for idx in unc:
            p = pos[idx]
            if len(neg) > 0:
                r = min(margin, _min_dist(p[None, :], neg).item())
            else:
                r = margin
            pc_list.append(p); pr_list.append(r)
    if pc_list:
        pc = torch.stack(pc_list); pr = torch.tensor(pr_list, device=device)
    else:
        pc = torch.zeros((0, centers.shape[1]), device=device)
        pr = torch.zeros((0,), device=device)
    return dict(centers=centers, radii=radii, pcenters=pc, pradii=pr, margin=margin,
                n_personal=len(pr_list), n_main=Kf)


# ============================ negative cloud / OOD ============================
def build_neg_ref(train_emb, train_lab, device, neg_q=0.99,
                  pos_cap=False, pos_cap_beta=0.9, pos_cap_q=0.01):
    neg = train_emb[train_lab == 0].to(device)
    pos = train_emb[train_lab == 1].to(device)
    sample = neg
    if len(neg) > 4000:
        g = torch.Generator(device=neg.device).manual_seed(SEED)
        idx = torch.randperm(len(neg), generator=g, device=neg.device)[:4000]
        sample = neg[idx]
    D = torch.cdist(sample, sample); D.fill_diagonal_(float('inf'))
    a = torch.quantile(D.min(dim=1).values, neg_q).item()
    tau, binding = a, "coverage"
    if pos_cap and len(pos) > 0:
        b_q = torch.quantile(_min_dist(pos, neg), pos_cap_q).item()
        if pos_cap_beta * b_q < tau:
            tau, binding = pos_cap_beta * b_q, "pos_cap"
    return neg, tau, binding


def ood_flags(test_emb, train_emb, device, k=5):
    thr = torch.quantile(_kth_nn_self(train_emb.to(device), k), 0.99).item()
    scores = []
    tr = train_emb.to(device)
    for i in range(0, len(test_emb), 1024):
        d = torch.cdist(test_emb[i:i + 1024].to(device), tr)
        scores.append(torch.topk(d, k=min(k, tr.shape[0]), dim=1, largest=False).values[:, -1])
    return (torch.cat(scores) > thr)


# ============================ decision rule ==================================
def decide(test_emb, geom, neg_ref, tau, ood_mask, train_emb, train_lab, device,
           use_personal=True, hard_classifier=False, use_ood=True,
           tau_mode="neg_cloud", knn_veto_k=0):
    x = test_emb.to(device)
    m = geom["margin"]
    s = (geom["radii"][None, :] - torch.cdist(x, geom["centers"])).max(dim=1).values
    inside = s >= 0
    if use_personal and len(geom["pradii"]) > 0:
        dp = torch.cdist(x, geom["pcenters"])
        inside = inside | (dp <= geom["pradii"][None, :]).any(dim=1)

    if hard_classifier:                      # no abstention at all
        return inside.int().cpu().numpy(), np.zeros(len(x), dtype=bool)

    if tau_mode == "threshold":
        conf_neg = s < -m                    # plain distance-from-ball rule
    else:
        d_neg = _min_dist(x, neg_ref)
        conf_neg = d_neg <= tau

    ood = ood_mask.to(device) if use_ood else torch.zeros(len(x), dtype=torch.bool, device=device)
    predict_negative = conf_neg & (~inside) & (~ood)

    if knn_veto_k > 0:
        neg_call = torch.where(predict_negative)[0]
        if len(neg_call):
            lab = _knn_labels(x[neg_call], train_emb.to(device), train_lab.to(device),
                              knn_veto_k)
            has_pos = (lab == 1).any(dim=1)
            predict_negative[neg_call[has_pos]] = False

    abst = (~inside) & (~predict_negative)
    return inside.int().cpu().numpy(), abst.cpu().numpy()


# ============================ metrics ========================================
def metrics(y, pred, abst):
    y = y.astype(int); pred = pred.astype(int); abst = np.asarray(abst, bool)
    keep = ~abst
    tp = int(((y == 1) & (pred == 1) & keep).sum())
    fn = int(((y == 1) & (pred == 0) & keep).sum())
    fp = int(((y == 0) & (pred == 1) & keep).sum())
    tn = int(((y == 0) & (pred == 0) & keep).sum())
    ap = int(((y == 1) & abst).sum()); an = int(((y == 0) & abst).sum())
    npos = int((y == 1).sum()); nneg = int((y == 0).sum())
    cov_p, cov_n = npos - ap, nneg - an
    return dict(
        TP=tp, FN=fn, FP=fp, TN=tn, abst_pos=ap, abst_neg=an,
        selective_fnr=round(fn / max(cov_p, 1), 6),
        selective_fpr=round(fp / max(cov_n, 1), 6),
        raw_fnr=round((fn + ap) / max(npos, 1), 6),
        raw_fpr=round(fp / max(nneg, 1), 6),
        coverage=round(1 - (ap + an) / max(npos + nneg, 1), 6),
        abstention=round((ap + an) / max(npos + nneg, 1), 6),
        operational=bool(fn == 0 and (fp / max(cov_n, 1)) < 0.1),
    )


# ============================ per-cell driver ================================
def run_cell(dataset, backbone, dim, device):
    fck = final_ckpt(backbone, dataset)
    if not fck.exists():
        return {"error": f"missing {fck}"}
    print(f"\n{'='*70}\n  CELL: {dataset} / {backbone}\n{'='*70}")

    model, state = load_model(backbone, dim, fck, device)
    ds = create_dataset(dataset, *get_dataset_paths(dataset))
    splits = auto_splits(ds)
    tr_s, te_s = splits["train"], splits.get("test") or splits["val"]

    tr_emb, tr_lab = cached_extract(model, tr_s, device, f"{backbone}_{dataset}_train")
    te_emb, te_lab = cached_extract(model, te_s, device, f"{backbone}_{dataset}_test")
    y = te_lab.numpy()

    # shared references (default config)
    neg_ref, tau0, bind0 = build_neg_ref(tr_emb, tr_lab, device, neg_q=0.99)
    ood = ood_flags(te_emb, tr_emb, device)
    geo_main = geometry_from_checkpoint(state, device)

    def ev(geom, **kw):
        pred, abst = decide(te_emb, geom, neg_ref, kw.pop("tau", tau0), ood,
                            tr_emb, tr_lab, device, **kw)
        return metrics(y, pred, abst)

    R = {"cell": {"dataset": dataset, "backbone": backbone,
                  "n_train": len(tr_s), "n_test": len(te_s),
                  "n_main_balls": int(len(geo_main["radii"])),
                  "n_personal": int(len(geo_main["pradii"])),
                  "tau_default": tau0, "tau_binding": bind0}}

    # ---------------- TIER 1 ----------------
    R["T1a_personal_ON"]  = ev(geo_main, use_personal=True)
    R["T1a_personal_OFF"] = ev(geo_main, use_personal=False)
    R["T1b_abstain"]      = ev(geo_main)
    R["T1b_hard_no_abstain"] = ev(geo_main, hard_classifier=True)

    # geometry-only: Stage-1 head, fit geometry, no primal-dual
    s1 = stage1_ckpt(backbone, dataset)
    if s1.exists():
        m1, _ = load_model(backbone, dim, s1, device)   # stage1 has model_state_dict
        e1_tr, l1_tr = cached_extract(m1, tr_s, device, f"{backbone}_{dataset}_train_s1")
        e1_te, l1_te = cached_extract(m1, te_s, device, f"{backbone}_{dataset}_test_s1")
        nref1, tau1, _ = build_neg_ref(e1_tr, l1_tr, device, neg_q=0.99)
        ood1 = ood_flags(e1_te, e1_tr, device)
        g1 = fit_geometry(e1_tr, l1_tr, device, K=max(2, len(geo_main["radii"])),
                          margin=FENCE_RADIUS_MARGIN)
        p, a = decide(e1_te, g1, nref1, tau1, ood1, e1_tr, l1_tr, device)
        R["T1c_geometry_only_stage1"] = metrics(l1_te.numpy(), p, a)
        del m1
    R["T1c_full_stage2"] = ev(geo_main)

    for rule in ("dual_quantile", "max", "positive_only", "fixed"):
        g = fit_geometry(tr_emb, tr_lab, device, K=max(2, len(geo_main["radii"])),
                         margin=FENCE_RADIUS_MARGIN, radius_rule=rule)
        R[f"T1d_radius_{rule}"] = ev(g)

    g_single = fit_geometry(tr_emb, tr_lab, device, K=1, margin=FENCE_RADIUS_MARGIN)
    R["T1e_single_ball_K1"] = ev(g_single)

    # ---------------- TIER 3 ----------------
    R["T3a_veto_OFF"] = ev(geo_main, knn_veto_k=0)
    R["T3a_veto_ON"]  = ev(geo_main, knn_veto_k=15)
    R["T3b_ood_ON"]   = ev(geo_main, use_ood=True)
    R["T3b_ood_OFF"]  = ev(geo_main, use_ood=False)
    R["T3c_tau_negcloud"]  = ev(geo_main, tau_mode="neg_cloud")
    R["T3c_tau_threshold"] = ev(geo_main, tau_mode="threshold")
    R["T3d_poscap_OFF"] = ev(geo_main)
    nref_pc, tau_pc, bind_pc = build_neg_ref(tr_emb, tr_lab, device, neg_q=0.99, pos_cap=True)
    _neg_save = neg_ref
    neg_ref = nref_pc
    R["T3d_poscap_ON"] = {**ev(geo_main, tau=tau_pc), "tau": tau_pc, "binding": bind_pc}
    neg_ref = _neg_save

    # ---------------- TIER 2 ----------------
    sweep = {}
    for q in (0.85, 0.90, 0.95, 0.99):
        nref_q, tau_q, _ = build_neg_ref(tr_emb, tr_lab, device, neg_q=q)
        _s = neg_ref; neg_ref = nref_q
        sweep[f"q{q}"] = {**ev(geo_main, tau=tau_q), "tau": tau_q}
        neg_ref = _s
    R["T2a_tau_sweep"] = sweep

    kres = {}
    for K in (1, 2, 4, 6, 8):
        g = fit_geometry(tr_emb, tr_lab, device, K=K, margin=FENCE_RADIUS_MARGIN)
        kres[f"K{K}"] = {**ev(g), "n_personal": g["n_personal"]}
    R["T2b_num_balls"] = kres

    mres = {}
    for mm in (0.05, 0.1, 0.2):
        g = fit_geometry(tr_emb, tr_lab, device, K=max(2, len(geo_main["radii"])), margin=mm)
        mres[f"m{mm}"] = {**ev(g), "n_personal": g["n_personal"]}
    R["T2c_margin"] = mres

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--backbones", nargs="*", default=None)
    ap.add_argument("--quick", action="store_true",
                    help="only ustc_smokers + dfire (hard+easy representative)")
    args = ap.parse_args()

    datasets = args.datasets or (["ustc_smokers", "dfire"] if args.quick else DATASETS)
    backbones = args.backbones or list(BACKBONES.keys())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Saving all ablation results to: {OUT_DIR.resolve()}")
    allres = {}
    for bk in backbones:
        for dsname in datasets:
            key = f"{dsname}__{bk}"
            try:
                R = run_cell(dsname, bk, BACKBONES[bk], DEVICE)
            except Exception as e:
                import traceback; traceback.print_exc()
                R = {"error": str(e)}
            allres[key] = R
            with open(OUT_DIR / f"ablation_{key}.json", "w") as f:
                json.dump(R, f, indent=2)
            print(f"  saved {OUT_DIR / f'ablation_{key}.json'}")

    with open(OUT_DIR / "ablations_ALL.json", "w") as f:
        json.dump(allres, f, indent=2)
    print(f"\n✅ done. combined: {OUT_DIR / 'ablations_ALL.json'}")


if __name__ == "__main__":
    main()