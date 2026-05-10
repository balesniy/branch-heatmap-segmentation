from __future__ import annotations

import heapq
from math import hypot
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from skimage.morphology import dilation, disk, skeletonize


def _f1(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / max(precision + recall, 1e-6)


@torch.no_grad()
def binary_metrics(
    pred_prob: torch.Tensor, target: torch.Tensor, threshold: float = 0.5
) -> dict[str, float]:
    pred = pred_prob > threshold
    target = target > 0.5
    tp = (pred & target).sum().float()
    fp = (pred & ~target).sum().float()
    fn = (~pred & target).sum().float()
    precision = tp / (tp + fp).clamp(min=1.0)
    recall = tp / (tp + fn).clamp(min=1.0)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-6)
    return {
        "precision": float(precision.cpu()),
        "recall": float(recall.cpu()),
        "f1": float(f1.cpu()),
    }


def _dilate_torch(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    kernel = 2 * radius + 1
    dilated = F.max_pool2d(mask.float(), kernel_size=kernel, stride=1, padding=radius)
    return dilated > 0


@torch.no_grad()
def tolerant_binary_metrics(
    pred_prob: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    radius: int = 2,
) -> dict[str, float]:
    pred = pred_prob > threshold
    target = target > 0.5
    pred_support = _dilate_torch(pred, radius)
    target_support = _dilate_torch(target, radius)
    precision = ((pred & target_support).sum().float() / pred.sum().float().clamp(min=1.0))
    recall = ((target & pred_support).sum().float() / target.sum().float().clamp(min=1.0))
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-6)
    return {
        f"precision@{radius}px": float(precision.cpu()),
        f"recall@{radius}px": float(recall.cpu()),
        f"f1@{radius}px": float(f1.cpu()),
    }


def cldice_np(pred_mask: np.ndarray, gt_mask: np.ndarray, radius: int = 2) -> dict[str, float]:
    pred = pred_mask > 0
    gt = gt_mask > 0
    pred_skel = skeletonize(pred)
    gt_skel = skeletonize(gt)
    footprint = disk(radius)
    pred_support = dilation(pred, footprint)
    gt_support = dilation(gt, footprint)
    cl_precision = (pred_skel & gt_support).sum() / max(int(pred_skel.sum()), 1)
    cl_recall = (gt_skel & pred_support).sum() / max(int(gt_skel.sum()), 1)
    return {
        f"cl_precision@{radius}px": float(cl_precision),
        f"cl_recall@{radius}px": float(cl_recall),
        f"cldice@{radius}px": _f1(float(cl_precision), float(cl_recall)),
    }


@torch.no_grad()
def cldice(
    pred_prob: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    radius: int = 2,
) -> dict[str, float]:
    pred_np = (pred_prob.detach().cpu().numpy() > threshold).astype(np.uint8)
    target_np = (target.detach().cpu().numpy() > 0.5).astype(np.uint8)
    totals = {
        f"cl_precision@{radius}px": 0.0,
        f"cl_recall@{radius}px": 0.0,
        f"cldice@{radius}px": 0.0,
    }
    count = max(1, pred_np.shape[0])
    for pred_mask, gt_mask in zip(pred_np[:, 0], target_np[:, 0]):
        values = cldice_np(pred_mask, gt_mask, radius=radius)
        for key, value in values.items():
            totals[key] += value
    return {key: value / count for key, value in totals.items()}


@torch.no_grad()
def centerline_metrics(
    pred_prob: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    tolerance_radius: int = 2,
) -> dict[str, float]:
    metrics = binary_metrics(pred_prob, target, threshold=threshold)
    metrics.update(
        tolerant_binary_metrics(
            pred_prob,
            target,
            threshold=threshold,
            radius=tolerance_radius,
        )
    )
    metrics.update(cldice(pred_prob, target, threshold=threshold, radius=1))
    if tolerance_radius != 1:
        metrics.update(cldice(pred_prob, target, threshold=threshold, radius=tolerance_radius))
    return metrics


def sample_polyline(polyline: Iterable[Iterable[float]], step: float = 5.0) -> np.ndarray:
    points = np.asarray(list(polyline), dtype=np.float32)
    if len(points) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if len(points) == 1:
        return points

    sampled = [points[0]]
    for a, b in zip(points[:-1], points[1:]):
        length = float(np.linalg.norm(b - a))
        if length == 0:
            continue
        n = max(1, int(np.floor(length / step)))
        for i in range(1, n + 1):
            sampled.append(a + (b - a) * min(1.0, i * step / length))
    if not np.allclose(sampled[-1], points[-1]):
        sampled.append(points[-1])
    return np.asarray(sampled, dtype=np.float32)


def _min_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if len(source) == 0:
        return np.zeros((0,), dtype=np.float32)
    if len(target) == 0:
        return np.full((len(source),), np.inf, dtype=np.float32)
    distances, _ = cKDTree(target).query(source, k=1)
    return np.asarray(distances, dtype=np.float32)


def _concat_samples(samples: list[np.ndarray]) -> np.ndarray:
    non_empty = [sample for sample in samples if len(sample)]
    if not non_empty:
        return np.zeros((0, 2), dtype=np.float32)
    return np.concatenate(non_empty, axis=0).astype(np.float32)


def polyline_geometry_metrics(
    pred_polyline: Iterable[Iterable[float]],
    gt_polyline: Iterable[Iterable[float]],
    sample_step: float = 2.0,
) -> dict[str, float]:
    pred = sample_polyline(pred_polyline, step=sample_step)
    gt = sample_polyline(gt_polyline, step=sample_step)
    pred_to_gt = _min_distances(pred, gt)
    gt_to_pred = _min_distances(gt, pred)
    all_dist = np.concatenate([pred_to_gt, gt_to_pred])
    pred_len = polyline_length(pred)
    gt_len = polyline_length(gt)
    return {
        "chamfer": float(all_dist.mean()) if len(all_dist) else float("inf"),
        "hausdorff95": float(np.percentile(all_dist, 95)) if len(all_dist) else float("inf"),
        "relative_length_error": float(abs(pred_len - gt_len) / max(gt_len, 1e-6)),
    }


def polyline_length(polyline: Iterable[Iterable[float]]) -> float:
    points = np.asarray(list(polyline), dtype=np.float32)
    if len(points) < 2:
        return 0.0
    delta = np.diff(points, axis=0)
    return float(np.hypot(delta[:, 0], delta[:, 1]).sum())


def coverage(source: np.ndarray, target: np.ndarray, tolerance: float) -> float:
    if len(source) == 0:
        return 0.0
    return float(np.mean(_min_distances(source, target) <= tolerance))


def edge_instance_f1(
    pred_polylines: list[Iterable[Iterable[float]]],
    gt_polylines: list[Iterable[Iterable[float]]],
    distance_tolerance: float = 3.0,
    coverage_threshold: float = 0.75,
    sample_step: float = 2.0,
) -> dict[str, float]:
    pred_samples = [sample_polyline(line, step=sample_step) for line in pred_polylines]
    gt_samples = [sample_polyline(line, step=sample_step) for line in gt_polylines]
    candidates = []
    for pred_idx, pred in enumerate(pred_samples):
        for gt_idx, gt in enumerate(gt_samples):
            gt_cov = coverage(gt, pred, distance_tolerance)
            pred_cov = coverage(pred, gt, distance_tolerance)
            if gt_cov >= coverage_threshold and pred_cov >= coverage_threshold:
                score = 0.5 * (gt_cov + pred_cov)
                candidates.append((-score, pred_idx, gt_idx))
    candidates.sort()

    matched_pred: set[int] = set()
    matched_gt: set[int] = set()
    for _, pred_idx, gt_idx in candidates:
        if pred_idx in matched_pred or gt_idx in matched_gt:
            continue
        matched_pred.add(pred_idx)
        matched_gt.add(gt_idx)

    tp = len(matched_pred)
    precision = tp / max(len(pred_polylines), 1)
    recall = tp / max(len(gt_polylines), 1)
    return {
        "edge_precision": float(precision),
        "edge_recall": float(recall),
        "edge_f1": _f1(precision, recall),
    }


def edge_f1(
    pred_polylines: list[Iterable[Iterable[float]]],
    gt_polylines: list[Iterable[Iterable[float]]],
    distance_tolerance: float = 3.0,
    coverage_threshold: float | None = None,
    sample_step: float = 1.0,
) -> dict[str, float]:
    """Length-weighted buffer coverage between predicted and GT graph geometry.

    This intentionally ignores one-to-one edge ids. If one long branch is split into
    several predicted segments, it still scores well as long as its sampled length is
    covered inside ``distance_tolerance``.
    """
    pred_samples = [sample_polyline(line, step=sample_step) for line in pred_polylines]
    gt_samples = [sample_polyline(line, step=sample_step) for line in gt_polylines]
    pred_points = _concat_samples(pred_samples)
    gt_points = _concat_samples(gt_samples)

    if len(pred_points) == 0 and len(gt_points) == 0:
        precision = recall = 1.0
    elif len(pred_points) == 0 or len(gt_points) == 0:
        precision = recall = 0.0
    else:
        pred_to_gt, _ = cKDTree(gt_points).query(
            pred_points,
            k=1,
            distance_upper_bound=distance_tolerance,
        )
        gt_to_pred, _ = cKDTree(pred_points).query(
            gt_points,
            k=1,
            distance_upper_bound=distance_tolerance,
        )
        precision = float(np.mean(pred_to_gt <= distance_tolerance))
        recall = float(np.mean(gt_to_pred <= distance_tolerance))
    return {
        "edge_precision": float(precision),
        "edge_recall": float(recall),
        "edge_f1": _f1(float(precision), float(recall)),
    }


def edge_coverage_f1(
    pred_polylines: list[Iterable[Iterable[float]]],
    gt_polylines: list[Iterable[Iterable[float]]],
    distance_tolerance: float = 3.0,
    coverage_threshold: float = 0.75,
    sample_step: float = 1.0,
) -> dict[str, float]:
    values = edge_f1(
        pred_polylines,
        gt_polylines,
        distance_tolerance=distance_tolerance,
        coverage_threshold=coverage_threshold,
        sample_step=sample_step,
    )
    return {
        "edge_coverage_precision": values["edge_precision"],
        "edge_coverage_recall": values["edge_recall"],
        "edge_coverage_f1": values["edge_f1"],
    }


def point_f1(
    pred_points: Iterable[Iterable[float]],
    gt_points: Iterable[Iterable[float]],
    radius: float = 5.0,
    prefix: str = "point",
) -> dict[str, float]:
    pred = np.asarray(list(pred_points), dtype=np.float32)
    gt = np.asarray(list(gt_points), dtype=np.float32)
    if len(pred) == 0 and len(gt) == 0:
        return {
            f"{prefix}_precision": 1.0,
            f"{prefix}_recall": 1.0,
            f"{prefix}_f1": 1.0,
        }
    candidates = []
    for pred_idx, pred_xy in enumerate(pred):
        for gt_idx, gt_xy in enumerate(gt):
            distance = float(np.linalg.norm(pred_xy - gt_xy))
            if distance <= radius:
                candidates.append((distance, pred_idx, gt_idx))
    candidates.sort()

    matched_pred: set[int] = set()
    matched_gt: set[int] = set()
    for _, pred_idx, gt_idx in candidates:
        if pred_idx in matched_pred or gt_idx in matched_gt:
            continue
        matched_pred.add(pred_idx)
        matched_gt.add(gt_idx)

    tp = len(matched_pred)
    precision = tp / max(len(pred), 1)
    recall = tp / max(len(gt), 1)
    return {
        f"{prefix}_precision": float(precision),
        f"{prefix}_recall": float(recall),
        f"{prefix}_f1": _f1(precision, recall),
    }


def node_f1(pred_graph, gt_graph, kind: str, radius: float = 5.0) -> dict[str, float]:
    pred_points = [node.xy for node in pred_graph.nodes if node.kind == kind]
    gt_points = [node.xy for node in gt_graph.nodes if node.kind == kind]
    return point_f1(pred_points, gt_points, radius=radius, prefix=kind)


def _merge_node(
    nodes: list[tuple[float, float]],
    xy: tuple[float, float],
    merge_radius: float,
) -> int:
    for idx, node_xy in enumerate(nodes):
        if hypot(xy[0] - node_xy[0], xy[1] - node_xy[1]) <= merge_radius:
            return idx
    nodes.append(xy)
    return len(nodes) - 1


def _build_sampled_graph(
    polylines: list[Iterable[Iterable[float]]],
    sample_step: float,
    merge_radius: float,
) -> tuple[list[tuple[float, float]], list[list[tuple[int, float]]]]:
    nodes: list[tuple[float, float]] = []
    adjacency: list[list[tuple[int, float]]] = []

    def ensure_node(xy: tuple[float, float]) -> int:
        idx = _merge_node(nodes, xy, merge_radius)
        while len(adjacency) <= idx:
            adjacency.append([])
        return idx

    for polyline in polylines:
        sampled = sample_polyline(polyline, step=sample_step)
        if len(sampled) < 2:
            continue
        ids = [ensure_node((float(x), float(y))) for x, y in sampled]
        for left, right in zip(ids[:-1], ids[1:]):
            if left == right:
                continue
            a, b = nodes[left], nodes[right]
            weight = hypot(a[0] - b[0], a[1] - b[1])
            adjacency[left].append((right, weight))
            adjacency[right].append((left, weight))
    return nodes, adjacency


def _nearest_node(
    xy: tuple[float, float],
    nodes: list[tuple[float, float]],
    radius: float,
) -> int | None:
    best_idx = None
    best_distance = float("inf")
    for idx, node_xy in enumerate(nodes):
        distance = hypot(xy[0] - node_xy[0], xy[1] - node_xy[1])
        if distance < best_distance:
            best_idx = idx
            best_distance = distance
    return best_idx if best_distance <= radius else None


def _dijkstra(adjacency: list[list[tuple[int, float]]], source: int) -> list[float]:
    dist = [float("inf")] * len(adjacency)
    dist[source] = 0.0
    queue = [(0.0, source)]
    while queue:
        cur_dist, node = heapq.heappop(queue)
        if cur_dist > dist[node]:
            continue
        for nb, weight in adjacency[node]:
            next_dist = cur_dist + weight
            if next_dist < dist[nb]:
                dist[nb] = next_dist
                heapq.heappush(queue, (next_dist, nb))
    return dist


def apls_score(
    pred_polylines: list[Iterable[Iterable[float]]],
    gt_polylines: list[Iterable[Iterable[float]]],
    sample_step: float = 10.0,
    snap_radius: float = 4.0,
    min_path_length: float = 20.0,
) -> float:
    gt_nodes, gt_adj = _build_sampled_graph(
        gt_polylines,
        sample_step=sample_step,
        merge_radius=max(1.0, snap_radius * 0.25),
    )
    pred_nodes, pred_adj = _build_sampled_graph(
        pred_polylines,
        sample_step=sample_step,
        merge_radius=max(1.0, snap_radius * 0.25),
    )
    if len(gt_nodes) < 2:
        return 1.0 if len(pred_nodes) < 2 else 0.0

    pred_snaps = [_nearest_node(xy, pred_nodes, snap_radius) for xy in gt_nodes]
    errors = []
    gt_all_dist = [_dijkstra(gt_adj, source) for source in range(len(gt_nodes))]
    pred_dist_cache: dict[int, list[float]] = {}

    for i in range(len(gt_nodes)):
        for j in range(i + 1, len(gt_nodes)):
            d_gt = gt_all_dist[i][j]
            if not np.isfinite(d_gt) or d_gt < min_path_length:
                continue
            pred_i = pred_snaps[i]
            pred_j = pred_snaps[j]
            if pred_i is None or pred_j is None:
                errors.append(1.0)
                continue
            if pred_i not in pred_dist_cache:
                pred_dist_cache[pred_i] = _dijkstra(pred_adj, pred_i)
            d_pred = pred_dist_cache[pred_i][pred_j]
            if not np.isfinite(d_pred):
                errors.append(1.0)
                continue
            errors.append(min(1.0, abs(d_gt - d_pred) / max(d_gt, 1e-6)))

    if not errors:
        return 0.0
    return float(1.0 - np.mean(errors))


def graph_quality_score(
    apls: float,
    edge_f1_value: float,
    cldice_at_2px: float,
    junction_f1: float,
    endpoint_f1: float,
) -> float:
    return float(
        0.40 * apls
        + 0.25 * edge_f1_value
        + 0.20 * cldice_at_2px
        + 0.10 * junction_f1
        + 0.05 * endpoint_f1
    )
