"""
FENCE Grad-CAM visualization (ViT-L/14 CLIP backbone only).

For each dataset, draws one grid figure of fire-only (TP) samples:
    rows    = fire samples
    col 1   = original image
    col 2   = Grad-CAM with the classifier-logit target
                (which pixels the model reads as fire evidence)
    col 3   = Grad-CAM with the enclosure-margin target s = R - d
                (which pixels pull the embedding into or out of the balls)

The backbone is frozen after Stage 1, so the maps show what the Stage 1
representation responds to; the two targets connect that representation to the
classifier score and to the Stage 2 geometry respectively.

Use --n_grids K to produce K separate grids per dataset with disjoint
samples, so you can pick the best-looking one.

Outputs PNG + PDF per grid in results/figures/.

Usage:
    python code/fence/visualize_gradcam.py
    python code/fence/visualize_gradcam.py --n_grids 3
    python code/fence/visualize_gradcam.py --datasets fasdd dfire --n_samples 8
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
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from fence.config import (
    DEVICE, BASE_DIR, FENCE_BACKBONE, SEED, USE_AMP,
)
from fence.stage2_train import get_dataset_paths
from fence.visualize_fence import (
    load_finalized_model, extract_embeddings, compute_decisions,
    outcome_categories,
)
from train_datasets.dataset_factory import create_dataset, create_dataloaders, auto_splits


DATASET_TITLES = {
    "fasdd": "FASDD",
    "dfire": "D-Fire",
    "ustc_smokers": "USTC SmokeRS",
    "flame": "FLAME",
}
OUTCOME_LABELS = {
    'tp': 'TP', 'tn': 'TN', 'fp': 'FP', 'fn': 'FN',
    'apos': 'abstained pos', 'aneg': 'abstained neg',
    'apos_ood': 'abstained pos (OOD)', 'aneg_ood': 'abstained neg (OOD)',
}
RARE_ORDER = ['fp', 'fn', 'apos', 'aneg', 'apos_ood', 'aneg_ood']

# Eval transform (must match dataset_factory.py: ImageNet-normalized 224x224)
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Grad-CAM for the SentenceTransformer CLIP ViT tower
# ---------------------------------------------------------------------------
class ClipViTGradCAM:
    """Minimal Grad-CAM on the last encoder layer's layer_norm1 of the HF CLIP
    vision tower inside the SentenceTransformer backbone."""

    def __init__(self, model):
        fb = model.backbone.backbone                 # FENCEBackbone
        if getattr(fb, 'backbone_type', None) != 'vit':
            raise RuntimeError(
                f"Grad-CAM script supports the ViT (CLIP) backbone only, "
                f"got: {getattr(fb, 'backbone_type', '?')}")
        vm = fb._vision_model                        # HF CLIPVisionTransformer
        self.target_layer = vm.encoder.layers[-1].layer_norm1

        # Gradient checkpointing was enabled for training; it prevents the
        # activation hooks from capturing tensors. Disable it for CAM.
        try:
            fb.clip_model[0].model.gradient_checkpointing_disable()
            print("  gradient checkpointing disabled for CAM")
        except Exception as e:
            print(f"  note: could not disable gradient checkpointing: {e}")

        self._act = None
        self._grad = None
        self._hook = self.target_layer.register_forward_hook(self._fwd_hook)

    def _fwd_hook(self, module, inputs, output):
        output.requires_grad_(True)
        self._act = output
        output.register_hook(self._bwd_hook)

    def _bwd_hook(self, grad):
        self._grad = grad

    def close(self):
        self._hook.remove()

    def _tokens_to_cam(self):
        # act/grad: (1, 257, C) with CLS at index 0 -> (256, C) -> 16x16
        act = self._act[0, 1:, :]
        grad = self._grad[0, 1:, :]
        weights = grad.mean(dim=0)                          # (C,)
        cam = torch.relu((act * weights).sum(dim=-1))       # (256,)
        cam = cam.reshape(16, 16)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-9)
        cam = F.interpolate(cam[None, None], size=(224, 224),
                            mode='bicubic', align_corners=False)
        return cam[0, 0].detach().cpu().numpy()

    def compute(self, model, x, target):
        """One forward + backward for a single image. target in {'logit', 'margin'}."""
        model.zero_grad(set_to_none=True)
        x = x.clone().requires_grad_(True)
        with torch.enable_grad():
            out = model.backbone(x)                          # embeddings, logits
            if target == 'logit':
                scalar = out['logits'].float().sum()
            elif target == 'margin':
                z = out['embeddings'].float()
                s_main = model.union_of_balls.distance_to_nearest(z)
                if len(model.quarantine_balls) > 0:
                    C = torch.stack([b.center for b in model.quarantine_balls]).float()
                    R = torch.tensor([b.radius for b in model.quarantine_balls],
                                     device=z.device, dtype=torch.float32)
                    s_q = (R.unsqueeze(0) - torch.cdist(z, C)).max(dim=1).values
                    scalar = torch.maximum(s_main, s_q).sum()
                else:
                    scalar = s_main.sum()
            else:
                raise ValueError(target)
            scalar.backward()
        return self._tokens_to_cam()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def overlay_cam(pil_img, cam, alpha=0.45):
    img = np.asarray(pil_img.resize((224, 224))).astype(np.float32) / 255.0
    heat = plt.get_cmap('turbo')(cam)[..., :3]
    blended = (1 - alpha) * img + alpha * heat
    return np.clip(blended, 0, 1)


def pick_fire_grids(cat, n_samples, n_grids, seed):
    """Pick n_grids disjoint groups of n_samples fire-only samples (TPs).

    Returns a list of n_grids lists of eval-split indices. Samples are
    drawn without replacement across grids, so every grid shows different
    images. Raises if there are not enough TPs."""
    need = n_samples * n_grids
    tp_idx = np.where(cat == 'tp')[0]
    if len(tp_idx) < need:
        raise RuntimeError(
            f"Only {len(tp_idx)} TP samples available, need {need} "
            f"({n_grids} grids x {n_samples}). Reduce --n_samples or --n_grids.")
    rng = np.random.RandomState(seed)
    tp_idx = tp_idx.copy()
    rng.shuffle(tp_idx)
    grids = []
    for g in range(n_grids):
        grids.append([int(i) for i in tp_idx[g * n_samples:(g + 1) * n_samples]])
    return grids


# ---------------------------------------------------------------------------
# Per-dataset pipeline
# ---------------------------------------------------------------------------
def run_dataset(name, args, device):
    from fence.config import FENCE_CHECKPOINT_DIR, FENCE_BACKBONE as BB
    ckpt_override = getattr(args, f"ckpt_{name}")
    ckpt_path = Path(ckpt_override) if ckpt_override else (
        FENCE_CHECKPOINT_DIR / name / BB / "stage2" / "final_model.pt")
    print(f"\n[{name}] checkpoint: {ckpt_path}")
    if not ckpt_path.exists():
        print(f"  Not found, skipping {name}.")
        return

    model, state = load_finalized_model(ckpt_path, device)
    if model.backbone.backbone.backbone_type != 'vit':
        print(f"  Backbone is not ViT ({model.backbone.backbone.backbone_type}); "
              f"Grad-CAM script is ViT-only. Skipping {name}.")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return

    # --- data + decisions (for outcome labels and rare-case sampling) ---
    image_dir, ann_dir = get_dataset_paths(name)
    ds = create_dataset(name, image_dir, ann_dir)
    splits = auto_splits(ds)
    use_split = args.split
    eval_samples = splits.get(use_split)
    if not eval_samples:
        for fallback in ['test', 'val', 'cal']:
            if splits.get(fallback):
                print(f"  '{use_split}' empty for {name}; using '{fallback}'.")
                use_split = fallback
                eval_samples = splits[fallback]
                break
    loaders = create_dataloaders(
        {'ref': splits['train'], 'eval': eval_samples}, test_mode=False)

    print("  rebuilding OOD + negative-cloud reference...")
    train_emb, train_lab = extract_embeddings(model, loaders['ref'], device,
                                              desc=f"  {name} train")
    model.store_ood_reference(train_emb, train_lab)

    z, lab = extract_embeddings(model, loaders['eval'], device,
                                desc=f"  {name} {use_split}")
    dec = compute_decisions(model, z)
    labels = lab.cpu().numpy().astype(int)
    cat = outcome_categories(labels, dec)

    # freeze params: CAM needs input grads only
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    grids = pick_fire_grids(cat, args.n_samples, args.n_grids, args.seed)

    cam_engine = ClipViTGradCAM(model)

    col_titles = ["input",
                  "Grad-CAM: classifier logit",
                  "Grad-CAM: enclosure margin $s$"]
    out_dir = Path(args.out_dir)
    for g, picks in enumerate(grids):
        print(f"  grid {g + 1}/{len(grids)} samples: {picks}")
        n = len(picks)
        fig, axes = plt.subplots(n, 3, figsize=(9.6, 3.15 * n), squeeze=False)
        for row, idx in enumerate(picks):
            sample = eval_samples[idx]
            pil_img = Image.open(sample['img_path']).convert('RGB')
            x = EVAL_TRANSFORM(pil_img).unsqueeze(0).to(device)

            cam_logit = cam_engine.compute(model, x, 'logit')
            cam_margin = cam_engine.compute(model, x, 'margin')

            outcome = OUTCOME_LABELS[cat[idx]]
            row_data = [
                np.asarray(pil_img.resize((224, 224))).astype(np.float32) / 255.0,
                overlay_cam(pil_img, cam_logit),
                overlay_cam(pil_img, cam_margin),
            ]
            for col in range(3):
                ax = axes[row][col]
                ax.imshow(row_data[col])
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    ax.set_title(col_titles[col], fontsize=10)
                if col == 0:
                    ax.set_ylabel(f"fire | {outcome}", fontsize=9)

        fig.suptitle(f"{DATASET_TITLES.get(name, name)} ({use_split})  |  "
                     f"Grad-CAM on {BB}, fire samples", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        suffix = f"_g{g + 1}" if len(grids) > 1 else ""
        out_base = out_dir / f"fence_gradcam_{name}_{use_split}{suffix}"
        fig.savefig(str(out_base) + ".png", dpi=300, bbox_inches='tight')
        fig.savefig(str(out_base) + ".pdf", bbox_inches='tight')
        print(f"  saved: {out_base}.png")
        plt.close(fig)

    cam_engine.close()
    del model, train_emb, train_lab, z, lab
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(
        description="FENCE Grad-CAM grids (ViT backbone), one figure per dataset")
    ap.add_argument("--datasets", nargs="+",
                    default=["fasdd", "dfire", "ustc_smokers", "flame"],
                    choices=["fasdd", "dfire", "ustc_smokers", "flame"])
    ap.add_argument("--split", default="test", choices=["val", "cal", "test"])
    ap.add_argument("--n_samples", type=int, default=5,
                    help="Fire samples (TPs) per grid")
    ap.add_argument("--n_grids", type=int, default=1,
                    help="Number of separate grids per dataset, each with "
                         "different samples (drawn without replacement)")
    ap.add_argument("--seed", type=int, default=SEED,
                    help="Random seed for sample selection (change for different samples)")
    ap.add_argument("--out_dir", default=str(BASE_DIR / "results" / "figures"))
    for ds_name in ["fasdd", "dfire", "ustc_smokers", "flame"]:
        ap.add_argument(f"--ckpt_{ds_name}", default=None)
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    device = DEVICE
    for name in args.datasets:
        run_dataset(name, args, device)
    print("\nDone. One Grad-CAM grid per dataset in:", args.out_dir)


if __name__ == "__main__":
    main()
