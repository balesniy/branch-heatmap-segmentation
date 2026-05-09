from __future__ import annotations

import json
import inspect
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


def read_polylines(path: Path) -> list[np.ndarray]:
    """Read branches from JSON.

    Supported keys: "branches", "polylines", or root-level list.
    A polyline is [[x, y], [x, y], ...].
    """
    data: Any = json.loads(path.read_text())
    items = data.get("branches", data.get("polylines")) if isinstance(data, dict) else data
    if items is None:
        raise ValueError(f"No 'branches' or 'polylines' key in {path}")
    polylines = []
    for points in items:
        arr = np.asarray(points, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] >= 2 and arr.shape[1] == 2:
            polylines.append(arr)
    return polylines


def flatten_polylines(polylines: list[np.ndarray]) -> tuple[list[tuple[float, float]], list[int]]:
    keypoints: list[tuple[float, float]] = []
    lengths = []
    for line in polylines:
        lengths.append(len(line))
        keypoints.extend((float(x), float(y)) for x, y in line)
    return keypoints, lengths


def unflatten_polylines(
    keypoints: list[tuple[float, float]], lengths: list[int]
) -> list[np.ndarray]:
    polylines = []
    offset = 0
    for n in lengths:
        pts = keypoints[offset : offset + n]
        offset += n
        if len(pts) >= 2:
            polylines.append(np.asarray(pts, dtype=np.float32))
    return polylines


def render_center_mask(
    polylines: list[np.ndarray],
    height: int,
    width: int,
    thickness: int = 1,
    render_scale: int = 4,
) -> np.ndarray:
    """Anti-aliased centerline render via high-resolution rasterization."""
    h2, w2 = height * render_scale, width * render_scale
    canvas = np.zeros((h2, w2), dtype=np.uint8)
    for line in polylines:
        pts = np.round(line * render_scale).astype(np.int32)
        if pts.shape[0] >= 2:
            cv2.polylines(
                canvas,
                [pts.reshape(-1, 1, 2)],
                isClosed=False,
                color=255,
                thickness=max(1, thickness * render_scale),
                lineType=cv2.LINE_AA,
            )
    soft = cv2.resize(canvas, (width, height), interpolation=cv2.INTER_AREA)
    return (soft > 127).astype(np.float32)


def make_gaussian_heatmap(center_mask: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    if center_mask.max() <= 0:
        return np.zeros_like(center_mask, dtype=np.float32)
    inv = (1.0 - center_mask).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
    heatmap = np.exp(-(dist * dist) / (2.0 * sigma * sigma))
    return heatmap.astype(np.float32)


def build_train_transform(image_size: tuple[int, int]) -> A.Compose:
    h, w = image_size
    if "std_range" in inspect.signature(A.GaussNoise).parameters:
        noise = A.GaussNoise(std_range=(0.02, 0.12), p=0.2)
    else:
        noise = A.GaussNoise(var_limit=(5.0, 30.0), p=0.2)
    return A.Compose(
        [
            A.LongestMaxSize(max_size=max(h, w)),
            A.PadIfNeeded(min_height=h, min_width=w, border_mode=cv2.BORDER_REFLECT_101),
            A.RandomCrop(height=h, width=w),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ElasticTransform(alpha=80, sigma=6, p=0.35),
            A.RandomBrightnessContrast(p=0.3),
            noise,
            A.Normalize(),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


def build_val_transform(image_size: tuple[int, int]) -> A.Compose:
    h, w = image_size
    return A.Compose(
        [
            A.Resize(height=h, width=w),
            A.Normalize(),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


class BranchPolylineDataset(Dataset):
    def __init__(
        self,
        images_dir: str | Path,
        annotations_dir: str | Path,
        image_size: tuple[int, int] = (512, 512),
        train: bool = True,
        gaussian_sigma: float = 2.0,
        line_thickness: int = 1,
        render_scale: int = 4,
    ):
        self.images_dir = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)
        self.image_paths = sorted(
            p for p in self.images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {self.images_dir}")
        self.transform = build_train_transform(image_size) if train else build_val_transform(image_size)
        self.gaussian_sigma = gaussian_sigma
        self.line_thickness = line_thickness
        self.render_scale = render_scale

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image_path = self.image_paths[index]
        ann_path = self.annotations_dir / f"{image_path.stem}.json"
        if not ann_path.exists():
            raise FileNotFoundError(f"Missing annotation for {image_path.name}: {ann_path}")

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        polylines = read_polylines(ann_path)
        keypoints, lengths = flatten_polylines(polylines)

        augmented = self.transform(image=image, keypoints=keypoints)
        aug_image = augmented["image"]
        aug_polylines = unflatten_polylines(augmented["keypoints"], lengths)

        height, width = int(aug_image.shape[1]), int(aug_image.shape[2])
        center = render_center_mask(
            aug_polylines,
            height=height,
            width=width,
            thickness=self.line_thickness,
            render_scale=self.render_scale,
        )
        heatmap = make_gaussian_heatmap(center, sigma=self.gaussian_sigma)

        return {
            "image": aug_image,
            "gaussian": torch.from_numpy(heatmap[None, ...]),
            "center": torch.from_numpy(center[None, ...]),
            "path": str(image_path),
        }
