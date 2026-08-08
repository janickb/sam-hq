"""Interactive HQ-SAM inference over sgdata .hdf5 frames.

For each object in each frame's embedded COCO annotations, click point
prompts on the image, run HQ-SAM, and compare the predicted mask against
the ground-truth segmentation (IoU). Results are written to a CSV.

Usage:
    python scripts/interactive_hdf5_inference.py \
        --hdf5 "D:/path/to/frames/*.hdf5" \
        --checkpoint pretrained_checkpoint/sam_hq_vit_l.pth \
        --model-type vit_l \
        --out results/interactive_run1

Frames are grouped into "scenes" by filename prefix before the first
underscore (e.g. "023701F8C" in "023701F8C_02-011.hdf5"); the trailing
number after the prefix is just that scene's image index. Use the
Prev/Next Scene buttons to jump straight to another scene's first frame.

Controls (point-collection window):
    left click   = positive point
    right click  = negative point
    enter        = run HQ-SAM on the collected points
    b            = run HQ-SAM using the ground-truth bbox as a box prompt instead (ignores points)
    a            = run automatic "segment everything" on the whole frame instead (no prompts,
                   matches each GT object to its best-overlapping auto mask), then moves to the next frame
    r            = reset points for this object
    s            = skip this object
    q            = quit the whole session (writes out what's collected so far)
    buttons      = "<< Prev Scene" / "Next Scene >>" jump to the adjacent scene's first frame

Controls (result window):
    any key / click = continue to the next object
"""

import argparse
import csv
import glob
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import pycocotools.mask as mask_util
import torch

from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry

from sgdata import reader as sg_reader


def frame_scene_id(path: str) -> str:
    """The hash prefix before the first '_' identifies the scene a frame
    belongs to; the trailing number (e.g. '02' in '023701F8C_02-011') is
    just that scene's image index."""
    return os.path.splitext(os.path.basename(path))[0].split("_", 1)[0]


def decode_gt_mask(segmentation) -> np.ndarray:
    rle = dict(segmentation)
    if isinstance(rle["counts"], str):
        rle["counts"] = rle["counts"].encode("ascii")
    return mask_util.decode(rle).astype(bool)


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return float(intersection / union) if union > 0 else 0.0


def collect_points(image: np.ndarray, title: str):
    """Blocking window for click-based point prompt collection.

    Returns (points, action) where points is a list of (x, y, label) and
    action is one of 'run', 'run_box', 'auto_frame', 'skip', 'quit',
    'prev_scene', 'next_scene'. 'run_box' means: run HQ-SAM using the
    ground-truth bbox as a box prompt, ignoring any points. 'auto_frame'
    means: run automatic "segment everything" on the whole frame instead.
    """
    points = []
    state = {"action": "run"}

    fig, ax = plt.subplots(figsize=(9, 9))
    fig.subplots_adjust(bottom=0.13)
    ax.imshow(image)
    ax.set_title(
        f"{title}\nleft=+point  right=-point  enter=run(points)  b=run(GT box)  a=segment-everything  r=reset  s=skip  q=quit",
        fontsize=10,
    )
    ax.axis("off")
    scat_pos = ax.scatter([], [], c="lime", marker="*", s=250, edgecolors="white", linewidths=1)
    scat_neg = ax.scatter([], [], c="red", marker="*", s=250, edgecolors="white", linewidths=1)

    def go_prev_scene(_event):
        state["action"] = "prev_scene"
        plt.close(fig)

    def go_next_scene(_event):
        state["action"] = "next_scene"
        plt.close(fig)

    ax_prev_scene = fig.add_axes([0.12, 0.02, 0.2, 0.055])
    ax_next_scene = fig.add_axes([0.68, 0.02, 0.2, 0.055])
    btn_prev_scene = Button(ax_prev_scene, "<< Prev Scene")
    btn_next_scene = Button(ax_next_scene, "Next Scene >>")
    btn_prev_scene.on_clicked(go_prev_scene)
    btn_next_scene.on_clicked(go_next_scene)

    def redraw():
        pos = np.array([p[:2] for p in points if p[2] == 1]) if any(p[2] == 1 for p in points) else np.empty((0, 2))
        neg = np.array([p[:2] for p in points if p[2] == 0]) if any(p[2] == 0 for p in points) else np.empty((0, 2))
        scat_pos.set_offsets(pos)
        scat_neg.set_offsets(neg)
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax or event.xdata is None:
            return
        if event.button == 1:
            points.append((event.xdata, event.ydata, 1))
            redraw()
        elif event.button == 3:
            points.append((event.xdata, event.ydata, 0))
            redraw()

    def on_key(event):
        if event.key == "enter":
            state["action"] = "run"
            plt.close(fig)
        elif event.key == "b":
            state["action"] = "run_box"
            plt.close(fig)
        elif event.key == "a":
            state["action"] = "auto_frame"
            plt.close(fig)
        elif event.key == "r":
            points.clear()
            redraw()
        elif event.key == "s":
            state["action"] = "skip"
            plt.close(fig)
        elif event.key == "q":
            state["action"] = "quit"
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

    return points, state["action"]


def show_result(image, pred_mask, gt_mask, points, iou, score, title, prompt_type="points"):
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(image)

    gt_overlay = np.zeros((*gt_mask.shape, 4))
    gt_overlay[gt_mask] = [1.0, 0.85, 0.0, 0.35]
    ax.imshow(gt_overlay)

    pred_overlay = np.zeros((*pred_mask.shape, 4))
    pred_overlay[pred_mask] = [30 / 255, 144 / 255, 255 / 255, 0.5]
    ax.imshow(pred_overlay)

    pos = np.array([p[:2] for p in points if p[2] == 1]) if any(p[2] == 1 for p in points) else np.empty((0, 2))
    neg = np.array([p[:2] for p in points if p[2] == 0]) if any(p[2] == 0 for p in points) else np.empty((0, 2))
    if len(pos):
        ax.scatter(pos[:, 0], pos[:, 1], c="lime", marker="*", s=250, edgecolors="white", linewidths=1)
    if len(neg):
        ax.scatter(neg[:, 0], neg[:, 1], c="red", marker="*", s=250, edgecolors="white", linewidths=1)

    ax.set_title(f"{title}\nprompt={prompt_type}  IoU={iou:.3f}  model_score={score:.3f}  (yellow=GT, blue=prediction)", fontsize=10)
    ax.axis("off")

    state = {"advance": False}

    def advance(_event):
        state["advance"] = True
        plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", advance)
    fig.canvas.mpl_connect("key_press_event", advance)
    plt.show()


def show_auto_result(image, auto_masks, gt_masks, best_ious, title):
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(image)

    rng = np.random.default_rng(0)
    overlay = np.zeros((*image.shape[:2], 4))
    for m in sorted(auto_masks, key=lambda m: m["area"], reverse=True):
        color = np.concatenate([rng.random(3), [0.5]])
        overlay[m["segmentation"]] = color
    ax.imshow(overlay)

    mean_iou = float(np.mean(best_ious)) if best_ious else 0.0
    ax.set_title(
        f"{title}\n{len(auto_masks)} auto masks vs {len(gt_masks)} GT objects  "
        f"mean best-IoU={mean_iou:.3f}  (colored=auto masks)",
        fontsize=10,
    )
    ax.axis("off")

    def advance(_event):
        plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", advance)
    fig.canvas.mpl_connect("key_press_event", advance)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hdf5", required=True, help="Path or glob pattern for .hdf5 frames")
    parser.add_argument("--checkpoint", default="pretrained_checkpoint/sam_hq_vit_l.pth")
    parser.add_argument("--model-type", default="vit_l", choices=["vit_b", "vit_l", "vit_h", "vit_tiny"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hq-token-only", action="store_true", help="Use HQ output only instead of SAM+HQ fusion")
    parser.add_argument("--multimask-output", action="store_true", help="Let SAM propose 3 masks and keep the highest-scoring one")
    parser.add_argument("--auto-points-per-side", type=int, default=32, help="Grid density for automatic 'segment everything' mode")
    parser.add_argument("--auto-pred-iou-thresh", type=float, default=0.88, help="Quality filter for automatic mode")
    parser.add_argument("--auto-stability-score-thresh", type=float, default=0.95, help="Stability filter for automatic mode")
    parser.add_argument("--out", default="results/interactive_hdf5", help="Output directory for annotated images + results.csv")
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        raise SystemExit(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Download a checkpoint per the README (Model Checkpoints section) and pass --checkpoint <path>."
        )

    os.makedirs(args.out, exist_ok=True)

    frame_paths = sorted(glob.glob(args.hdf5))
    if not frame_paths:
        raise SystemExit(f"No files matched: {args.hdf5}")

    scene_ids = [frame_scene_id(p) for p in frame_paths]
    unique_scenes = list(dict.fromkeys(scene_ids))
    scene_position = {scene: i for i, scene in enumerate(unique_scenes)}
    print(f"{len(frame_paths)} frames across {len(unique_scenes)} scenes")

    print(f"Loading {args.model_type} checkpoint from {args.checkpoint} on {args.device} ...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=args.device)
    predictor = SamPredictor(sam)
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=args.auto_points_per_side,
        pred_iou_thresh=args.auto_pred_iou_thresh,
        stability_score_thresh=args.auto_stability_score_thresh,
    )

    results = []
    cache = {"frame_idx": None, "image": None, "annotations": None}

    def load_frame(idx):
        if cache["frame_idx"] == idx:
            return cache["image"], cache["annotations"]
        path = frame_paths[idx]
        annotations = sg_reader.read_coco_annotations(path) or []
        if not annotations:
            print(f"note: {path} has no coco_annotations (run sgdata.backfill first?) or no objects")
        image = sg_reader.read_image(path)
        if image.shape[-1] == 4:
            image = image[..., :3]
        image = np.ascontiguousarray(image)
        predictor.set_image(image)
        cache.update(frame_idx=idx, image=image, annotations=annotations)
        return image, annotations

    frame_idx, obj_idx = 0, 0
    while 0 <= frame_idx < len(frame_paths):
        image, annotations = load_frame(frame_idx)
        frame_path = frame_paths[frame_idx]
        frame_name = os.path.basename(frame_path)
        scene_id = scene_ids[frame_idx]

        if obj_idx >= len(annotations):
            frame_idx += 1
            obj_idx = 0
            continue

        ann = annotations[obj_idx]
        gt_mask = decode_gt_mask(ann["segmentation"])
        title = (
            f"Scene {scene_position[scene_id] + 1}/{len(unique_scenes)}: {scene_id}\n"
            f"{frame_name}  object {obj_idx + 1}/{len(annotations)}"
        )

        points, action = collect_points(image, title)

        if action == "quit":
            break

        if action == "next_scene":
            j = frame_idx
            while j < len(frame_paths) and scene_ids[j] == scene_id:
                j += 1
            if j < len(frame_paths):
                frame_idx, obj_idx = j, 0
            else:
                print("Already at the last scene.")
            continue

        if action == "prev_scene":
            start = frame_idx
            while start > 0 and scene_ids[start - 1] == scene_id:
                start -= 1
            if start == 0:
                print("Already at the first scene.")
            else:
                prev_scene = scene_ids[start - 1]
                j = start - 1
                while j > 0 and scene_ids[j - 1] == prev_scene:
                    j -= 1
                frame_idx, obj_idx = j, 0
            continue

        if action == "skip" or (action == "run" and not points):
            obj_idx += 1
            continue

        if action == "auto_frame":
            auto_masks = mask_generator.generate(image)
            gt_masks = [decode_gt_mask(a["segmentation"]) for a in annotations]

            best_ious = []
            for gt_idx, gt_mask in enumerate(gt_masks):
                per_mask_ious = [compute_iou(m["segmentation"], gt_mask) for m in auto_masks]
                best_iou = max(per_mask_ious) if per_mask_ious else 0.0
                best_idx = int(np.argmax(per_mask_ious)) if per_mask_ious else None
                best_ious.append(best_iou)
                results.append(
                    {
                        "frame": frame_name,
                        "object_index": gt_idx,
                        "prompt_type": "auto",
                        "num_pos_points": 0,
                        "num_neg_points": 0,
                        "iou": best_iou,
                        "model_score": float(auto_masks[best_idx]["predicted_iou"]) if best_idx is not None else 0.0,
                        "gt_area": annotations[gt_idx].get("area"),
                    }
                )

            show_auto_result(image, auto_masks, gt_masks, best_ious, title)

            out_name = f"{os.path.splitext(frame_name)[0]}_auto.png"
            fig, ax = plt.subplots(figsize=(9, 9))
            ax.imshow(image)
            overlay = np.zeros((*image.shape[:2], 4))
            for m in auto_masks:
                overlay[m["segmentation"]] = np.concatenate([np.random.random(3), [0.5]])
            ax.imshow(overlay)
            ax.axis("off")
            fig.savefig(os.path.join(args.out, out_name), bbox_inches="tight", pad_inches=-0.1)
            plt.close(fig)

            obj_idx = len(annotations)
            continue

        if action == "run_box":
            x, y, w, h = ann["bbox"]
            box = np.array([x, y, x + w, y + h])
            point_coords, point_labels = None, None
            prompt_type = "box"
        else:
            box = None
            point_coords = np.array([[p[0], p[1]] for p in points])
            point_labels = np.array([p[2] for p in points])
            prompt_type = "points"

        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=args.multimask_output,
            hq_token_only=args.hq_token_only,
        )
        best = int(np.argmax(scores))
        pred_mask, score = masks[best], float(scores[best])
        iou = compute_iou(pred_mask, gt_mask)

        show_result(image, pred_mask, gt_mask, points, iou, score, title, prompt_type=prompt_type)

        out_name = f"{os.path.splitext(frame_name)[0]}_obj{obj_idx}.png"
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.imshow(image)
        overlay = np.zeros((*pred_mask.shape, 4))
        overlay[pred_mask] = [30 / 255, 144 / 255, 255 / 255, 0.5]
        ax.imshow(overlay)
        ax.axis("off")
        fig.savefig(os.path.join(args.out, out_name), bbox_inches="tight", pad_inches=-0.1)
        plt.close(fig)

        results.append(
            {
                "frame": frame_name,
                "object_index": obj_idx,
                "prompt_type": prompt_type,
                "num_pos_points": int(sum(1 for p in points if p[2] == 1)),
                "num_neg_points": int(sum(1 for p in points if p[2] == 0)),
                "iou": iou,
                "model_score": score,
                "gt_area": ann.get("area"),
            }
        )
        obj_idx += 1

    csv_path = os.path.join(args.out, "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "object_index", "prompt_type", "num_pos_points", "num_neg_points", "iou", "model_score", "gt_area"])
        writer.writeheader()
        writer.writerows(results)

    if results:
        ious = [r["iou"] for r in results]
        print(f"\n{len(results)} objects annotated. Mean IoU: {np.mean(ious):.3f}  Min: {np.min(ious):.3f}  Max: {np.max(ious):.3f}")
    print(f"Results written to {csv_path}")


if __name__ == "__main__":
    main()
