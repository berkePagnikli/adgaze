"""Compare pipeline runs with the ground truth produced by annotate.py.

    python scripts/evaluate.py                       # every gt/<video>.csv vs runs/<video>/frames.csv
    python scripts/evaluate.py --runs runs/yaw15     # another run folder

Matching: in every frame, GT person boxes are matched to tracked person boxes (Hungarian, IoU >= 0.5),
so the metrics do not depend on track IDs. GT rows labelled -1 (ignore) are skipped.

Metrics (positive class = LOOKING):
    coverage        share of GT person-frames that the tracker found (matched)
    accuracy / precision / recall / F1   frame level; an unmatched GT person-frame counts as "not looking"
    watch_gt_s / watch_pred_s / watch_abs_err_s   watch time per GT person, summed per video
Writes <runs>/evaluation.csv.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import GT_DIR, RUNS_DIR
from run_pipeline import resolve


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / np.maximum(area(a)[:, None] + area(b)[None, :] - inter, 1e-9)


def match(gt: pd.DataFrame, pred: pd.DataFrame, min_iou: float) -> pd.Series:
    """Predicted (filtered) looking state for every GT row; NaN where no tracked person matched."""
    out = pd.Series(np.nan, index=gt.index)
    pred_by_frame = {f: d for f, d in pred.groupby("frame")}
    for frame, g in gt.groupby("frame"):
        p = pred_by_frame.get(frame)
        if p is None:
            continue
        iou = iou_matrix(g[["x1", "y1", "x2", "y2"]].to_numpy(float), p[["px1", "py1", "px2", "py2"]].to_numpy(float))
        rows, cols = linear_sum_assignment(-iou)
        for r, c in zip(rows, cols):
            if iou[r, c] >= min_iou:
                out[g.index[r]] = p["looking"].iloc[c]
    return out


def evaluate_video(gt: pd.DataFrame, pred: pd.DataFrame, fps: float, min_iou: float) -> dict:
    gt = gt[gt["looking"] >= 0].copy()
    gt["pred"] = match(gt, pred, min_iou)
    y_true = gt["looking"].to_numpy(int)
    y_pred = gt["pred"].fillna(0).to_numpy(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    per_person = gt.assign(p=y_pred).groupby("gt_id").agg(gt_f=("looking", "sum"), pred_f=("p", "sum"))
    return {
        "gt_rows": len(gt),
        "coverage": gt["pred"].notna().mean(),
        "accuracy": (tp + tn) / len(gt) if len(gt) else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "watch_gt_s": per_person["gt_f"].sum() / fps,
        "watch_pred_s": per_person["pred_f"].sum() / fps,
        "watch_abs_err_s": (per_person["gt_f"] - per_person["pred_f"]).abs().sum() / fps,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=str(RUNS_DIR), help="folder with one sub-folder per video")
    ap.add_argument("--gt", default=str(GT_DIR), help="ground-truth folder")
    ap.add_argument("--min-iou", type=float, default=0.5)
    args = ap.parse_args()

    runs_dir, gt_dir = resolve(args.runs), resolve(args.gt)
    results = []
    for gt_csv in sorted(gt_dir.glob("*.csv")):
        run = runs_dir / gt_csv.stem
        if not (run / "frames.csv").exists():
            print(f"[evaluate] skip {gt_csv.stem}: no {run / 'frames.csv'} (run run_pipeline.py first)")
            continue
        fps = json.loads((run / "summary.json").read_text(encoding="utf-8"))["fps"]
        res = evaluate_video(pd.read_csv(gt_csv), pd.read_csv(run / "frames.csv"), fps, args.min_iou)
        results.append({"video": gt_csv.stem, **res})
    if not results:
        raise SystemExit("[evaluate] nothing to evaluate")

    df = pd.DataFrame(results)
    tp, fp, fn, tn = (df[k].sum() for k in ("tp", "fp", "fn", "tn"))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    total = {
        "video": "ALL", "gt_rows": df["gt_rows"].sum(),
        "coverage": (df["coverage"] * df["gt_rows"]).sum() / df["gt_rows"].sum(),
        "accuracy": (tp + tn) / df["gt_rows"].sum(), "precision": p, "recall": r,
        "f1": 2 * p * r / (p + r) if p + r else 0.0, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "watch_gt_s": df["watch_gt_s"].sum(), "watch_pred_s": df["watch_pred_s"].sum(),
        "watch_abs_err_s": df["watch_abs_err_s"].sum(),
    }
    df = pd.concat([df, pd.DataFrame([total])], ignore_index=True).round(3)
    df.to_csv(runs_dir / "evaluation.csv", index=False)
    cols = ["video", "gt_rows", "coverage", "accuracy", "precision", "recall", "f1",
            "watch_gt_s", "watch_pred_s", "watch_abs_err_s"]
    print(df[cols].to_string(index=False))
    print(f"-> {runs_dir / 'evaluation.csv'}")


if __name__ == "__main__":
    main()
