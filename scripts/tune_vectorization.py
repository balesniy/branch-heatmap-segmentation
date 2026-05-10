from __future__ import annotations

import argparse
import itertools
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from branch_segmentation.dataset import render_center_mask
from branch_segmentation.metrics import (
    apls_score,
    cldice_np,
    edge_coverage_f1,
    edge_f1,
    edge_instance_f1,
    graph_quality_score,
)
from branch_segmentation.postprocess import heatmap_to_graph


def parse_csv(value: str, cast):
    return [cast(item) for item in value.split(",") if item != ""]


def tree_id_from_name(name: str) -> int | None:
    match = re.match(r"tree_(\d+)_", name)
    return int(match.group(1)) if match else None


def load_coco_polylines(path: Path) -> dict[str, list[np.ndarray]]:
    coco = json.loads(path.read_text())
    image_by_id = {item["id"]: item["file_name"] for item in coco["images"]}
    out: dict[str, list[np.ndarray]] = defaultdict(list)
    for ann in coco["annotations"]:
        name = image_by_id.get(ann.get("image_id"))
        polyline = ann.get("polyline")
        if name is None or not isinstance(polyline, list) or len(polyline) < 2:
            continue
        arr = np.asarray(polyline, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] == 2:
            out[name].append(arr)
    return dict(out)


def select_images(
    images_dir: Path,
    heatmap_dir: Path,
    val_tree_ids: set[int] | None,
    limit: int | None,
) -> list[Path]:
    image_paths = []
    for image_path in sorted(images_dir.glob("*.png")):
        heatmap_path = heatmap_dir / f"{image_path.stem}.heatmap.png"
        if not heatmap_path.exists():
            continue
        tid = tree_id_from_name(image_path.name)
        if val_tree_ids is not None and tid not in val_tree_ids:
            continue
        image_paths.append(image_path)
    return image_paths[:limit] if limit else image_paths


def evaluate_params(
    params: dict,
    image_paths: list[Path],
    heatmap_dir: Path,
    gt_by_name: dict[str, list[np.ndarray]],
    compute_apls: bool = False,
) -> dict:
    rows = []
    for image_path in image_paths:
        heatmap_path = heatmap_dir / f"{image_path.stem}.heatmap.png"
        heatmap = cv2.imread(str(heatmap_path), cv2.IMREAD_GRAYSCALE)
        if heatmap is None:
            raise ValueError(f"Could not read heatmap {heatmap_path}")
        heatmap = heatmap.astype(np.float32) / 255.0
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image {image_path}")
        h, w = image.shape[:2]
        gt_polys = gt_by_name.get(image_path.name, [])
        gt_mask = render_center_mask(gt_polys, h, w, thickness=1, render_scale=2) > 0
        graph = heatmap_to_graph(heatmap, **params)
        pred_polys = [edge.points for edge in graph.edges]
        cl = cldice_np(graph.mask, gt_mask, radius=2)
        strict = edge_instance_f1(
            pred_polys,
            gt_polys,
            distance_tolerance=4.0,
            coverage_threshold=0.5,
            sample_step=4.0,
        )
        covered = edge_coverage_f1(
            pred_polys,
            gt_polys,
            distance_tolerance=4.0,
            coverage_threshold=0.5,
            sample_step=4.0,
        )
        apls = (
            apls_score(
                pred_polys,
                gt_polys,
                sample_step=20.0,
                snap_radius=8.0,
                min_path_length=40.0,
            )
            if compute_apls
            else 0.0
        )
        quick_score = float(
            0.45 * covered["edge_coverage_f1"]
            + 0.35 * cl["cldice@2px"]
            + 0.10 * covered["edge_coverage_precision"]
            + 0.10 * covered["edge_coverage_recall"]
        )
        rows.append(
            {
                "image": image_path.name,
                "gt_polylines": len(gt_polys),
                "pred_edges": len(pred_polys),
                "pred_nodes": len(graph.nodes),
                "pred_mask_pixels": int(graph.mask.sum()),
                "cldice@2px": cl["cldice@2px"],
                "strict_edge_f1@4px": strict["edge_f1"],
                "edge_coverage_f1@4px": covered["edge_coverage_f1"],
                "edge_coverage_precision@4px": covered["edge_coverage_precision"],
                "edge_coverage_recall@4px": covered["edge_coverage_recall"],
                "apls_like": apls,
                "score": (
                    graph_quality_score(
                        apls,
                        covered["edge_coverage_f1"],
                        cl["cldice@2px"],
                        0.0,
                        0.0,
                    )
                    if compute_apls
                    else quick_score
                ),
            }
        )
    mean = {}
    for key in [
        "pred_edges",
        "pred_mask_pixels",
        "cldice@2px",
        "strict_edge_f1@4px",
        "edge_coverage_f1@4px",
        "edge_coverage_precision@4px",
        "edge_coverage_recall@4px",
        "apls_like",
        "score",
    ]:
        mean[key] = float(np.mean([row[key] for row in rows])) if rows else 0.0
    return {"params": params, "mean": mean, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--heatmap-dir", required=True)
    parser.add_argument("--coco-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--val-tree-ids", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--low", default="0.30,0.35", help="comma-separated")
    parser.add_argument("--high", default="0.45,0.50")
    parser.add_argument("--min-object-size", default="4,8")
    parser.add_argument("--closing-radius", default="0")
    parser.add_argument("--centerline-mode", default="ridge_skeleton,ridge,skeleton")
    parser.add_argument("--ridge-nms-size", default="3")
    parser.add_argument("--min-len", default="5,8")
    parser.add_argument("--simplify-tol", default="0.75")
    parser.add_argument("--spur-min-length", default="5,10")
    parser.add_argument("--spur-min-p20-score", default="0.15,0.25")
    parser.add_argument("--edge-min-p20-score", default="0.0,0.15")
    parser.add_argument("--edge-max-low-score-fraction", default="1.0,0.75")
    parser.add_argument("--edge-keep-longer-than", type=float, default=80.0)
    parser.add_argument("--cycle-collapse-max-length", default="0,15")
    parser.add_argument("--gap-max", default="6")
    parser.add_argument("--angle-max", default="45")
    parser.add_argument("--min-bridge-score", default="0.15")
    parser.add_argument("--snap-radius", default="0,2")
    parser.add_argument("--apls-top-k", type=int, default=10)
    args = parser.parse_args()

    val_tree_ids = (
        set(parse_csv(args.val_tree_ids, int)) if args.val_tree_ids is not None else None
    )
    images = select_images(
        Path(args.images_dir),
        Path(args.heatmap_dir),
        val_tree_ids=val_tree_ids,
        limit=args.limit,
    )
    gt_by_name = load_coco_polylines(Path(args.coco_json))
    if not images:
        raise FileNotFoundError("No images with matching .heatmap.png files found")

    grids = {
        "low": parse_csv(args.low, float),
        "high": parse_csv(args.high, float),
        "min_object_size": parse_csv(args.min_object_size, int),
        "closing_radius": parse_csv(args.closing_radius, int),
        "centerline_mode": parse_csv(args.centerline_mode, str),
        "ridge_nms_size": parse_csv(args.ridge_nms_size, int),
        "min_len": parse_csv(args.min_len, float),
        "simplify_tol": parse_csv(args.simplify_tol, float),
        "spur_min_length": parse_csv(args.spur_min_length, float),
        "spur_min_p20_score": parse_csv(args.spur_min_p20_score, float),
        "edge_min_p20_score": parse_csv(args.edge_min_p20_score, float),
        "edge_max_low_score_fraction": parse_csv(args.edge_max_low_score_fraction, float),
        "cycle_collapse_max_length": parse_csv(args.cycle_collapse_max_length, float),
        "gap_max": parse_csv(args.gap_max, float),
        "angle_max": parse_csv(args.angle_max, float),
        "min_bridge_score": parse_csv(args.min_bridge_score, float),
        "snap_radius": parse_csv(args.snap_radius, int),
    }
    names = list(grids)
    trials = []
    total = 1
    for values in grids.values():
        total *= len(values)
    print(f"tuning {total} parameter sets on {len(images)} images")

    for idx, values in enumerate(itertools.product(*(grids[name] for name in names)), start=1):
        params = dict(zip(names, values))
        if params["low"] >= params["high"]:
            continue
        params["blur_sigma"] = 0.0
        params["bridge_gaps"] = True
        params["edge_keep_longer_than"] = args.edge_keep_longer_than
        result = evaluate_params(
            params,
            images,
            Path(args.heatmap_dir),
            gt_by_name,
            compute_apls=False,
        )
        trials.append(result)
        if idx % 25 == 0:
            best = max(trials, key=lambda item: item["mean"]["score"])
            print(
                f"{idx}/{total} best score={best['mean']['score']:.4f} "
                f"coverage_f1={best['mean']['edge_coverage_f1@4px']:.4f} "
                f"cldice={best['mean']['cldice@2px']:.4f}"
            )

    trials.sort(key=lambda item: item["mean"]["score"], reverse=True)
    top_with_apls = [
        evaluate_params(
            item["params"],
            images,
            Path(args.heatmap_dir),
            gt_by_name,
            compute_apls=True,
        )
        for item in trials[: max(1, args.apls_top_k)]
    ]
    top_with_apls.sort(key=lambda item: item["mean"]["score"], reverse=True)
    summary = {
        "images": [path.name for path in images],
        "best": top_with_apls[0],
        "best_quick": trials[0],
        "top_with_apls": top_with_apls,
        "trials": trials,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2))
    print(json.dumps({"best": trials[0]["params"], "mean": trials[0]["mean"]}, indent=2))


if __name__ == "__main__":
    main()
