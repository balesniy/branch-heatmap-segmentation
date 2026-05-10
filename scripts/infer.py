from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2

from branch_segmentation import BranchHeatmapNet
from branch_segmentation.postprocess import heatmap_to_graph, render_graph_overlay


def _tta_variants(tensor: torch.Tensor, modes: list[str]) -> list[tuple[str, torch.Tensor]]:
    variants = []
    for mode in modes:
        if mode == "original":
            variants.append((mode, tensor))
        elif mode == "hflip":
            variants.append((mode, torch.flip(tensor, dims=[3])))
        elif mode == "vflip":
            variants.append((mode, torch.flip(tensor, dims=[2])))
        elif mode == "rot90":
            variants.append((mode, torch.rot90(tensor, k=1, dims=[2, 3])))
        elif mode == "rot180":
            variants.append((mode, torch.rot90(tensor, k=2, dims=[2, 3])))
        elif mode == "rot270":
            variants.append((mode, torch.rot90(tensor, k=3, dims=[2, 3])))
        else:
            raise ValueError(f"Unknown TTA mode: {mode}")
    return variants


def _invert_tta(pred: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "original":
        return pred
    if mode == "hflip":
        return torch.flip(pred, dims=[3])
    if mode == "vflip":
        return torch.flip(pred, dims=[2])
    if mode == "rot90":
        return torch.rot90(pred, k=3, dims=[2, 3])
    if mode == "rot180":
        return torch.rot90(pred, k=2, dims=[2, 3])
    if mode == "rot270":
        return torch.rot90(pred, k=1, dims=[2, 3])
    raise ValueError(f"Unknown TTA mode: {mode}")


def predict_heatmap(
    model: BranchHeatmapNet,
    tensor: torch.Tensor,
    tta_modes: list[str],
) -> np.ndarray:
    preds = []
    with torch.no_grad():
        for mode, augmented in _tta_variants(tensor, tta_modes):
            pred = model(augmented, return_side_outputs=False)
            preds.append(_invert_tta(pred, mode))
    return torch.stack(preds).mean(dim=0)[0, 0].cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--low-thr", type=float, default=0.35)
    parser.add_argument("--high-thr", type=float, default=0.5)
    parser.add_argument("--blur-sigma", type=float, default=0.5)
    parser.add_argument("--min-object-size", type=int, default=4)
    parser.add_argument("--closing-radius", type=int, default=0)
    parser.add_argument(
        "--centerline-mode",
        default="ridge_skeleton",
        choices=["skeleton", "ridge", "ridge_skeleton"],
    )
    parser.add_argument("--ridge-nms-size", type=int, default=3)
    parser.add_argument("--min-len", type=float, default=8.0)
    parser.add_argument("--simplify-tol", type=float, default=1.0)
    parser.add_argument("--spur-min-length", type=float, default=10.0)
    parser.add_argument("--spur-min-p20-score", type=float, default=0.25)
    parser.add_argument("--edge-min-p20-score", type=float, default=0.0)
    parser.add_argument("--edge-max-low-score-fraction", type=float, default=1.0)
    parser.add_argument("--edge-keep-longer-than", type=float, default=80.0)
    parser.add_argument("--cycle-collapse-max-length", type=float, default=15.0)
    parser.add_argument("--gap-max", type=float, default=6.0)
    parser.add_argument("--angle-max", type=float, default=40.0)
    parser.add_argument("--min-bridge-score", type=float, default=0.2)
    parser.add_argument("--snap-radius", type=int, default=2)
    parser.add_argument("--no-bridge-gaps", action="store_true")
    parser.add_argument(
        "--tta",
        nargs="+",
        default=["original", "hflip", "vflip"],
        choices=["original", "hflip", "vflip", "rot90", "rot180", "rot270"],
    )
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    h, w = cfg["data"]["image_size"]
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    model = BranchHeatmapNet(**cfg["model"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    image_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Could not read image {args.image}")
    orig_h, orig_w = image_bgr.shape[:2]
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    transform = Compose([Resize(h, w), Normalize(), ToTensorV2()])
    tensor = transform(image=image)["image"][None].to(device)

    pred = predict_heatmap(model, tensor, args.tta)
    pred = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    graph = heatmap_to_graph(
        pred,
        low=args.low_thr,
        high=args.high_thr,
        blur_sigma=args.blur_sigma,
        min_object_size=args.min_object_size,
        closing_radius=args.closing_radius,
        centerline_mode=args.centerline_mode,
        ridge_nms_size=args.ridge_nms_size,
        min_len=args.min_len,
        simplify_tol=args.simplify_tol,
        spur_min_length=args.spur_min_length,
        spur_min_p20_score=args.spur_min_p20_score,
        edge_min_p20_score=args.edge_min_p20_score,
        edge_max_low_score_fraction=args.edge_max_low_score_fraction,
        edge_keep_longer_than=args.edge_keep_longer_than,
        cycle_collapse_max_length=args.cycle_collapse_max_length,
        bridge_gaps=not args.no_bridge_gaps,
        gap_max=args.gap_max,
        angle_max=args.angle_max,
        min_bridge_score=args.min_bridge_score,
        snap_radius=args.snap_radius,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output.with_suffix(".heatmap.png")), (graph.heatmap * 255).astype(np.uint8))
    cv2.imwrite(str(output.with_suffix(".mask.png")), graph.mask.astype(np.uint8) * 255)
    cv2.imwrite(str(output.with_suffix(".skeleton.png")), graph.skeleton.astype(np.uint8) * 255)
    overlay = render_graph_overlay(image, graph)
    cv2.imwrite(str(output.with_suffix(".graph.png")), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    with output.with_suffix(".graph.json").open("w", encoding="utf-8") as f:
        json.dump(graph.to_json_dict(), f, indent=2)


if __name__ == "__main__":
    main()
