"""
Publication figures + comparison tables for the FENCE paper.

Reads your final-eval results JSON (from eval_all.py / eval_final_model.py) and the
pAUC+CRC baseline's reported numbers, and produces paper-ready PNGs:

  fig_fpr_comparison.png     FENCE vs pAUC+CRC: FPR at zero-omission (log bars)
  fig_risk_coverage.png      FENCE risk-coverage frontier per dataset (τ sweep)
  fig_human_workload.png     images sent to humans (FP + abstain) FENCE vs baseline

Also prints a matched-human-workload table to stdout.

Baseline numbers are ViT+pAUC+CRC at zero-omission, from the companion paper
(Tables 3 & 5). Edit BASELINE below if you update them.

Run:
    python code/fence/paper_figures.py --results "PATH/ALL_results.json"
"""

import os, sys, json, argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── pAUC+CRC baseline (ViT+pAUC+CRC, zero-omission) — from the companion paper ──
#   fpr = false-positive rate ; alarms = number of false alarms ; fnr ≈ 0
BASELINE = {
    "ustc_smokers": {"fpr": 0.0665, "alarms": 68,  "fnr": 0.0},
    "dfire":        {"fpr": 0.0778, "alarms": 156, "fnr": 0.0},
    "flame":        {"fpr": 0.0788, "alarms": 230, "fnr": 0.0},
    "fasdd":        {"fpr": 0.0442, "alarms": 339, "fnr": 0.0},
}
NICE = {"ustc_smokers": "USTC", "dfire": "D-Fire", "flame": "FLAME", "fasdd": "FASDD"}
ORDER = ["ustc_smokers", "flame", "fasdd", "dfire"]     # wins first, hard last

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 150,
})
FEN_C, BASE_C = "#2e86de", "#e74c3c"


def load(results_path):
    data = json.load(open(results_path))
    if not isinstance(data, list) or not (data and isinstance(data[0], dict)
                                          and "final_eval_default_tau" in data[0]):
        raise ValueError(
            f"{results_path} is not the final-eval ALL_results.json (expected a LIST "
            f"of cells with 'final_eval_default_tau'). Point --results at "
            f"results/final_eval/ALL_results.json, NOT the ablations file.")
    # index ViT cells by dataset
    out = {}
    for cell in data:
        if cell.get("backbone", "").startswith("vit"):
            out[cell["dataset"]] = cell
    return out


def fig_fpr(fen, outdir):
    ds = [d for d in ORDER if d in fen]
    fen_fpr  = [fen[d]["final_eval_default_tau"]["selective_fpr"] * 100 for d in ds]
    base_fpr = [BASELINE[d]["fpr"] * 100 for d in ds]
    x = np.arange(len(ds)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w/2, base_fpr, w, label="pAUC+CRC (baseline)", color=BASE_C)
    ax.bar(x + w/2, [max(v, 0.02) for v in fen_fpr], w, label="FENCE (ours)", color=FEN_C)
    ax.set_yscale("log")
    ax.set_ylabel("False-positive rate at zero-omission (%, log)")
    ax.set_xticks(x); ax.set_xticklabels([NICE[d] for d in ds])
    for xi, (b, f) in enumerate(zip(base_fpr, fen_fpr)):
        ax.text(xi - w/2, b, f"{b:.1f}", ha="center", va="bottom", fontsize=9)
        ax.text(xi + w/2, max(f, 0.02), f"{f:.2f}", ha="center", va="bottom", fontsize=9)
    ax.legend(); ax.set_title("FENCE cuts false alarms ~80× at matched zero-omission")
    fig.tight_layout(); fig.savefig(outdir / "fig_fpr_comparison.png", bbox_inches="tight")
    plt.close(fig)


def fig_risk_coverage(fen, outdir):
    ds = [d for d in ORDER if d in fen]
    fig, axes = plt.subplots(1, len(ds), figsize=(4.2 * len(ds), 4), sharey=False)
    if len(ds) == 1: axes = [axes]
    for ax, d in zip(axes, ds):
        sweep = fen[d]["tau_sweep"]
        cov = [s["coverage"] * 100 for s in sweep]
        fpr = [s["selective_fpr"] * 100 for s in sweep]
        fn  = [s["FN"] for s in sweep]
        ax.plot(cov, fpr, "-o", color=FEN_C, label="FENCE sel-FPR")
        for c, f, n in zip(cov, fpr, fn):
            ax.annotate(f"FN={n}", (c, f), fontsize=8, xytext=(0, 5),
                        textcoords="offset points", ha="center")
        ax.axhline(BASELINE[d]["fpr"] * 100, color=BASE_C, ls="--",
                   label="pAUC+CRC FPR")
        ax.set_title(NICE[d]); ax.set_xlabel("Coverage (%)")
        ax.set_ylabel("Selective FPR (%)")
        ax.legend(fontsize=8)
    fig.suptitle("Risk–coverage frontier: FENCE trades <1–6% abstention for far lower FPR",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outdir / "fig_risk_coverage.png", bbox_inches="tight")
    plt.close(fig)


def fig_workload(fen, outdir):
    ds = [d for d in ORDER if d in fen]
    fen_load, base_load = [], []
    for d in ds:
        e = fen[d]["final_eval_default_tau"]
        fen_load.append(e["FP"] + e["abstain_pos"] + e["abstain_neg"])
        base_load.append(BASELINE[d]["alarms"])
    x = np.arange(len(ds)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w/2, base_load, w, label="pAUC+CRC (false alarms)", color=BASE_C)
    ax.bar(x + w/2, fen_load, w, label="FENCE (FP + abstain)", color=FEN_C)
    ax.set_ylabel("Images requiring human review")
    ax.set_xticks(x); ax.set_xticklabels([NICE[d] for d in ds])
    for xi, (b, f) in enumerate(zip(base_load, fen_load)):
        ax.text(xi - w/2, b, str(b), ha="center", va="bottom", fontsize=9)
        ax.text(xi + w/2, f, str(f), ha="center", va="bottom", fontsize=9)
    ax.legend(); ax.set_title("FENCE reduces human workload at equal or better omission")
    fig.tight_layout(); fig.savefig(outdir / "fig_human_workload.png", bbox_inches="tight")
    plt.close(fig)


def print_table(fen):
    print("\n  MATCHED-HUMAN-WORKLOAD (ViT, zero-omission operating point)")
    print(f"  {'Dataset':8} | {'Baseline alarms':>15} | {'FENCE FP+abstain':>16} | "
          f"{'FENCE missed(FN)':>16}")
    for d in ORDER:
        if d not in fen: continue
        e = fen[d]["final_eval_default_tau"]
        fen_load = e["FP"] + e["abstain_pos"] + e["abstain_neg"]
        print(f"  {NICE[d]:8} | {BASELINE[d]['alarms']:>15} | {fen_load:>16} | "
              f"{e['FN']:>16}")


def _is_final_eval(p):
    """True if p is the final-eval file (list of cells with final_eval_default_tau)."""
    try:
        d = json.load(open(p))
        return isinstance(d, list) and d and isinstance(d[0], dict) \
            and "final_eval_default_tau" in d[0]
    except Exception:
        return False


def _find_results():
    """Find the correct final-eval ALL_results.json BY CONTENT (ignores ablations
    files that happen to share a similar name). Prefers results/final_eval/."""
    repo = Path(__file__).parent.parent.parent
    cands = []
    for base in (repo, repo.parent, Path.home() / "OneDrive" / "Desktop", Path.cwd()):
        try:
            cands += list(base.rglob("ALL_results.json"))
            cands += list(base.rglob("*final_eval*/*.json"))
        except Exception:
            pass
    # de-dup, validate by content, prefer paths containing 'final_eval'
    seen, valid = set(), []
    for c in cands:
        if c in seen:
            continue
        seen.add(c)
        if _is_final_eval(c):
            valid.append(c)
    valid.sort(key=lambda p: ("final_eval" not in str(p).lower(), len(str(p))))
    return valid[0] if valid else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None,
                    help="path to final-eval ALL_results.json (auto-found if omitted)")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    if args.results:
        results = Path(args.results)
        if not _is_final_eval(results):
            print(f"  ⚠️  {results} is not the final-eval file; searching by content...")
            results = _find_results()
    else:
        results = _find_results()
    if not results or not results.exists():
        print("  ❌ final-eval ALL_results.json not found. It's the LIST-format file "
              "under results/final_eval/. Pass --results PATH."); return
    print(f"  Using results: {results}")
    fen = load(str(results))
    outdir = Path(args.outdir) if args.outdir else results.parent / "paper_figs"
    outdir.mkdir(parents=True, exist_ok=True)
    fig_fpr(fen, outdir)
    fig_risk_coverage(fen, outdir)
    fig_workload(fen, outdir)
    print_table(fen)
    print(f"\n  ✅ figures saved to {outdir.resolve()}")


if __name__ == "__main__":
    main()