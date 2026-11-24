"""Semi-automatic ground-truth annotation tool (OpenCV GUI).

The person tracks of a pipeline run are shown on the video; you only mark *when* each person
starts / stops looking at the screen. Labels persist forward in time until changed, so a
person usually needs only a few key presses. If the video has no run yet, the pipeline is run
first automatically (tracks -> runs/<video_stem>). Works from the thesis root or from scripts/.

    python scripts/annotate.py --video data/man.mp4
    -> tracks: scripts/runs/man/frames.csv (created if missing)
    -> ground truth: scripts/gt/man.csv  (+ man.json with the editable change points)

Note: the change points in gt/<video>.json refer to the track IDs of the run they were made on.
A new pipeline run may number the tracks differently, so re-open an existing annotation only with
the run it was made on. gt/<video>.csv itself stores boxes and is independent of track IDs.

Keys
    mouse click   select the person (track) under the cursor      TAB  cycle selection
    l / 1         selected person LOOKING from this frame on
    n / 0         selected person NOT LOOKING from this frame on
    i / 9         selected person IGNORE (-1, excluded from metrics) from this frame on
    u             merge the selected track into the previously selected one (same real person)
    z             undo the last label change
    d / a         next / previous processed frame         f / b   +25 / -25 frames
    SPACE         play / pause                              w       save
    ESC / q       save and quit
"""

from __future__ import annotations

import argparse
import bisect
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import SCRIPTS_DIR
from run_pipeline import resolve

LABEL_COLORS = {1: (60, 200, 60), 0: (60, 60, 230), -1: (160, 160, 160), None: (0, 200, 230)}
LABEL_NAMES = {1: "LOOKING", 0: "NOT", -1: "IGNORE", None: "?"}


class Annotation:
    def __init__(self, default: int | None):
        self.changes: dict[int, list[tuple[int, int]]] = {}
        self.id_map: dict[int, int] = {}
        self.history: list[tuple[int, int]] = []
        self.default = default

    def set(self, tid: int, frame: int, label: int) -> None:
        cps = [c for c in self.changes.get(tid, []) if c[0] != frame]
        bisect.insort(cps, (frame, label))
        self.changes[tid] = cps
        self.history.append((tid, frame))

    def undo(self) -> None:
        if self.history:
            tid, frame = self.history.pop()
            self.changes[tid] = [c for c in self.changes.get(tid, []) if c[0] != frame]

    def label(self, tid: int, frame: int):
        cps = self.changes.get(tid, [])
        i = bisect.bisect_right(cps, (frame, 99)) - 1
        return cps[i][1] if i >= 0 else self.default

    def gt_id(self, tid: int) -> int:
        seen = set()
        while tid in self.id_map and tid not in seen:
            seen.add(tid)
            tid = self.id_map[tid]
        return tid

    def to_json(self) -> dict:
        return {"changes": {str(k): v for k, v in self.changes.items()},
                "id_map": {str(k): v for k, v in self.id_map.items()}, "default": self.default}

    @classmethod
    def from_json(cls, d: dict) -> "Annotation":
        a = cls(d.get("default"))
        a.changes = {int(k): [tuple(c) for c in v] for k, v in d.get("changes", {}).items()}
        a.id_map = {int(k): int(v) for k, v in d.get("id_map", {}).items()}
        return a


def export(ann: Annotation, pred: pd.DataFrame, out_csv: Path) -> int:
    rows = []
    for r in pred.itertuples(index=False):
        lab = ann.label(int(r.track_id), int(r.frame))
        if lab is None:
            continue
        rows.append(dict(video=r.video, frame=int(r.frame), gt_id=ann.gt_id(int(r.track_id)),
                         x1=r.px1, y1=r.py1, x2=r.px2, y2=r.py2, looking=int(lab)))
    pd.DataFrame(rows, columns=["video", "frame", "gt_id", "x1", "y1", "x2", "y2", "looking"]).to_csv(
        out_csv, index=False)
    return len(rows)


def prepare_run(video: Path, run_arg: str | None, overrides: list[str]) -> Path:
    """Return a run folder with person tracks of ``video``; runs the pipeline first if needed."""
    run_dir = resolve(run_arg or f"runs/{video.stem}")
    summary_path = run_dir / "summary.json"
    if (run_dir / "frames.csv").exists() and summary_path.exists():
        run_video = json.loads(summary_path.read_text(encoding="utf-8")).get("video")
        if run_video != video.stem:
            raise SystemExit(f"[annotate] {run_dir} contains the tracks of '{run_video}', not of '{video.stem}'.\n"
                             f"           Omit --run (-> runs/{video.stem}) or give a new folder name.")
        return run_dir
    print(f"[annotate] no person tracks in {run_dir} yet -> running the pipeline on {video.name} first ...")
    from pipeline.config import Config
    from pipeline.pipeline import Pipeline

    Pipeline(Config().update(overrides)).run(str(video), run_dir, save_video=False)
    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--run", help="run folder with frames.csv of this video (default runs/<video_stem>); "
                                  "created with the pipeline if it does not exist yet")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override used only when the run is created")
    ap.add_argument("--out", help="ground-truth csv (default gt/<video_stem>.csv)")
    ap.add_argument("--default", type=int, choices=[-1, 0, 1], default=None,
                    help="label used before the first change point (default: unlabelled -> not exported)")
    ap.add_argument("--scale", type=float, default=1.0, help="display scale")
    args = ap.parse_args()

    video = resolve(args.video)
    if not video.is_file():
        raise SystemExit(f"[annotate] video not found: {args.video} (looked in {Path.cwd()} and {SCRIPTS_DIR})")
    run_dir = prepare_run(video, args.run, args.set)
    pred = pd.read_csv(run_dir / "frames.csv")
    out_csv = resolve(args.out or f"gt/{video.stem}.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    side = out_csv.with_suffix(".json")
    ann = Annotation.from_json(json.loads(side.read_text(encoding="utf-8"))) if side.exists() else Annotation(args.default)
    if args.default is not None:
        ann.default = args.default
    print(f"[annotate] video: {video}\n[annotate] tracks: {run_dir}\n[annotate] ground truth -> {out_csv}")

    n_frames = json.loads((run_dir / "summary.json").read_text(encoding="utf-8")).get("n_frames", 0)
    frames = list(range(n_frames)) or sorted(pred["frame"].unique().tolist())
    if not frames:
        raise SystemExit(f"[annotate] {run_dir / 'frames.csv'} has no processed frames")
    by_frame = {f: d for f, d in pred.groupby("frame")}

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"[annotate] cannot open video: {video}")
    pos = {"i": 0, "sel": None, "prev_sel": None, "play": False, "click": None}
    cache = {"idx": None, "img": None, "next": 0}

    def read(fidx: int):
        if cache["idx"] == fidx:
            return cache["img"]
        if cache["next"] != fidx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, img = cap.read()
        cache.update(idx=fidx, img=img if ok else None, next=fidx + 1)
        return cache["img"]

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pos["click"] = (x / args.scale, y / args.scale)

    win = "annotate (h: help in console)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    print(__doc__)

    def save():
        side.write_text(json.dumps(ann.to_json(), indent=1), encoding="utf-8")
        n = export(ann, pred, out_csv)
        print(f"[annotate] saved {n} rows -> {out_csv}")

    while True:
        fidx = frames[pos["i"]]
        img = read(fidx)
        if img is None:
            img = np.zeros((480, 640, 3), np.uint8)
        vis = img.copy()
        d = by_frame.get(fidx)
        visible = [] if d is None else d[["track_id", "px1", "py1", "px2", "py2"]].to_numpy()
        if pos["click"] is not None and len(visible):
            cx, cy = pos["click"]
            inside = [(r[3] - r[1]) * (r[4] - r[2]) if r[1] <= cx <= r[3] and r[2] <= cy <= r[4] else np.inf
                      for r in visible]
            if np.isfinite(min(inside)):
                pos["prev_sel"], pos["sel"] = pos["sel"], int(visible[int(np.argmin(inside))][0])
            pos["click"] = None
        for tid, x1, y1, x2, y2 in visible:
            tid = int(tid)
            lab = ann.label(tid, fidx)
            col = LABEL_COLORS[lab]
            thick = 4 if tid == pos["sel"] else 2
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), col, thick)
            txt = f"T{tid}->P{ann.gt_id(tid)} {LABEL_NAMES[lab]}"
            cv2.putText(vis, txt, (int(x1), max(int(y1) - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        cv2.putText(vis, f"frame {fidx} ({pos['i'] + 1}/{len(frames)})  selected: {pos['sel']}  "
                         f"{'PLAY' if pos['play'] else 'PAUSE'}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
        if args.scale != 1.0:
            vis = cv2.resize(vis, None, fx=args.scale, fy=args.scale)
        cv2.imshow(win, vis)
        key = cv2.waitKey(30 if pos["play"] else 50) & 0xFF
        sel = pos["sel"]
        if key in (27, ord("q")):
            save()
            break
        elif key == ord(" "):
            pos["play"] = not pos["play"]
        elif key == ord("d"):
            pos["i"] = min(pos["i"] + 1, len(frames) - 1)
        elif key == ord("a"):
            pos["i"] = max(pos["i"] - 1, 0)
        elif key == ord("f"):
            pos["i"] = min(pos["i"] + 25, len(frames) - 1)
        elif key == ord("b"):
            pos["i"] = max(pos["i"] - 25, 0)
        elif key == 9 and len(visible):  # TAB
            ids = sorted(int(t) for t in visible[:, 0])
            pos["prev_sel"], pos["sel"] = sel, ids[(ids.index(sel) + 1) % len(ids)] if sel in ids else ids[0]
        elif key in (ord("l"), ord("1")) and sel is not None:
            ann.set(sel, fidx, 1)
        elif key in (ord("n"), ord("0")) and sel is not None:
            ann.set(sel, fidx, 0)
        elif key in (ord("i"), ord("9")) and sel is not None:
            ann.set(sel, fidx, -1)
        elif key == ord("u") and sel is not None and pos["prev_sel"] not in (None, sel):
            ann.id_map[sel] = ann.gt_id(pos["prev_sel"])
            print(f"[annotate] track {sel} -> person P{ann.id_map[sel]}")
        elif key == ord("z"):
            ann.undo()
        elif key == ord("w"):
            save()
        elif key == ord("h"):
            print(__doc__)
        if pos["play"]:
            if pos["i"] < len(frames) - 1:
                pos["i"] += 1
            else:
                pos["play"] = False
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
