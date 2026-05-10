from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from branch_segmentation.yolo_pose_graph import (
    PoseDetection,
    build_yolo_heatmap_topology,
    render_topology_overlay,
)


def detections_from_yolo_result(result, conf_threshold: float) -> list[PoseDetection]:
    detections = []
    if result.boxes is None or result.keypoints is None:
        return detections
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    keypoints = result.keypoints.xy.detach().cpu().numpy()
    for bbox, conf, cls, kpts in zip(boxes, confs, classes, keypoints):
        if float(conf) < conf_threshold or kpts.shape[0] < 2:
            continue
        detections.append(
            PoseDetection(
                bbox_xyxy=tuple(float(v) for v in bbox),
                keypoints_xy=(
                    (float(kpts[0, 0]), float(kpts[0, 1])),
                    (float(kpts[1, 0]), float(kpts[1, 1])),
                ),
                confidence=float(conf),
                class_id=int(cls),
            )
        )
    return detections


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo-checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--heatmap", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--cluster-radius", type=float, default=8.0)
    parser.add_argument("--no-recover-missing", action="store_true")
    args = parser.parse_args()

    image_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Could not read image {args.image}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    heatmap = cv2.imread(args.heatmap, cv2.IMREAD_GRAYSCALE)
    if heatmap is None:
        raise ValueError(f"Could not read heatmap {args.heatmap}")
    heatmap = heatmap.astype(np.float32) / 255.0

    model = YOLO(args.yolo_checkpoint)
    result = model.predict(
        source=args.image,
        imgsz=args.imgsz,
        conf=args.conf,
        verbose=False,
    )[0]
    detections = detections_from_yolo_result(result, conf_threshold=args.conf)
    graph = build_yolo_heatmap_topology(
        detections,
        heatmap,
        cluster_radius=args.cluster_radius,
        recover_missing=not args.no_recover_missing,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".graph.json").open("w", encoding="utf-8") as f:
        json.dump(graph.to_json_dict(), f, indent=2)
    overlay = render_topology_overlay(image_rgb, graph)
    cv2.imwrite(str(output.with_suffix(".graph.png")), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
