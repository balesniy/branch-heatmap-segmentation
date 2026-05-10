from __future__ import annotations

from dataclasses import asdict, dataclass
from math import hypot
from typing import Iterable

import cv2
import numpy as np
from scipy.spatial import cKDTree
from skimage.graph import route_through_array

from branch_segmentation.metrics import sample_polyline
from branch_segmentation.postprocess import heatmap_to_graph


@dataclass(frozen=True)
class PoseDetection:
    bbox_xyxy: tuple[float, float, float, float]
    keypoints_xy: tuple[tuple[float, float], tuple[float, float]]
    confidence: float
    class_id: int


@dataclass(frozen=True)
class TopologyNode:
    id: int
    xy: tuple[float, float]
    confidence: float
    size: int


@dataclass(frozen=True)
class TopologyEdge:
    id: int
    start: int
    end: int
    points: list[tuple[float, float]]
    source: str
    confidence: float
    mean_prob: float
    min_prob: float
    p20_prob: float
    bad_ratio: float
    length: float
    euclidean_length: float
    tortuosity: float


@dataclass(frozen=True)
class TopologyGraph:
    nodes: list[TopologyNode]
    edges: list[TopologyEdge]

    @property
    def polylines(self) -> list[list[tuple[float, float]]]:
        return [edge.points for edge in self.edges]

    def to_json_dict(self) -> dict:
        return {
            "nodes": [asdict(node) for node in self.nodes],
            "edges": [asdict(edge) for edge in self.edges],
        }


class _DisjointSet:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, idx: int) -> int:
        parent = self.parent[idx]
        if parent != idx:
            self.parent[idx] = self.find(parent)
        return self.parent[idx]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _polyline_length(points: Iterable[Iterable[float]]) -> float:
    arr = np.asarray(list(points), dtype=np.float32)
    if len(arr) < 2:
        return 0.0
    delta = np.diff(arr, axis=0)
    return float(np.hypot(delta[:, 0], delta[:, 1]).sum())


def _line_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-6:
        return np.linalg.norm(points - a[None], axis=1)
    t = np.clip(((points - a[None]) @ ab) / denom, 0.0, 1.0)
    projection = a[None] + t[:, None] * ab[None]
    return np.linalg.norm(points - projection, axis=1)


def _edge_stats(
    points: list[tuple[float, float]],
    heatmap: np.ndarray,
    low_prob: float,
) -> tuple[float, float, float, float, float, float]:
    if len(points) < 2:
        return 0.0, 0.0, 0.0, 1.0, 0.0, 0.0
    sampled = sample_polyline(points, step=1.0)
    xs = np.clip(np.rint(sampled[:, 0]).astype(np.intp), 0, heatmap.shape[1] - 1)
    ys = np.clip(np.rint(sampled[:, 1]).astype(np.intp), 0, heatmap.shape[0] - 1)
    probs = heatmap[ys, xs].astype(np.float32)
    length = _polyline_length(points)
    euclidean = hypot(points[-1][0] - points[0][0], points[-1][1] - points[0][1])
    return (
        float(probs.mean()) if len(probs) else 0.0,
        float(probs.min()) if len(probs) else 0.0,
        float(np.percentile(probs, 20)) if len(probs) else 0.0,
        float(np.mean(probs < low_prob)) if len(probs) else 1.0,
        length,
        float(length / max(euclidean, 1e-6)),
    )


def shortest_heatmap_path(
    heatmap: np.ndarray,
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    bbox_xyxy: tuple[float, float, float, float] | None = None,
    pad: int = 24,
    cost_weight: float = 8.0,
    line_band: float = 16.0,
) -> list[tuple[float, float]]:
    height, width = heatmap.shape
    if bbox_xyxy is None:
        x0 = min(start_xy[0], end_xy[0]) - pad
        y0 = min(start_xy[1], end_xy[1]) - pad
        x1 = max(start_xy[0], end_xy[0]) + pad
        y1 = max(start_xy[1], end_xy[1]) + pad
    else:
        x0, y0, x1, y1 = bbox_xyxy
        x0 -= pad
        y0 -= pad
        x1 += pad
        y1 += pad
    x0i = max(0, int(np.floor(x0)))
    y0i = max(0, int(np.floor(y0)))
    x1i = min(width, int(np.ceil(x1)))
    y1i = min(height, int(np.ceil(y1)))
    if x1i - x0i < 2 or y1i - y0i < 2:
        return [start_xy, end_xy]

    crop = np.clip(heatmap[y0i:y1i, x0i:x1i].astype(np.float32), 0.0, 1.0)
    cost = 1.0 + cost_weight * np.square(1.0 - crop)

    yy, xx = np.mgrid[y0i:y1i, x0i:x1i]
    coords = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    distances = _line_distance(
        coords,
        np.asarray(start_xy, dtype=np.float32),
        np.asarray(end_xy, dtype=np.float32),
    ).reshape(crop.shape)
    cost = cost + np.clip((distances - line_band) / max(line_band, 1e-6), 0.0, None) * 8.0

    start_yx = (
        int(np.clip(round(start_xy[1]) - y0i, 0, crop.shape[0] - 1)),
        int(np.clip(round(start_xy[0]) - x0i, 0, crop.shape[1] - 1)),
    )
    end_yx = (
        int(np.clip(round(end_xy[1]) - y0i, 0, crop.shape[0] - 1)),
        int(np.clip(round(end_xy[0]) - x0i, 0, crop.shape[1] - 1)),
    )
    try:
        path_yx, _ = route_through_array(
            cost,
            start=start_yx,
            end=end_yx,
            fully_connected=True,
            geometric=True,
        )
    except Exception:
        return [start_xy, end_xy]
    return [(float(x + x0i), float(y + y0i)) for y, x in path_yx]


def refine_pose_detections(
    detections: list[PoseDetection],
    heatmap: np.ndarray,
    min_mean_prob: float = 0.25,
    max_bad_ratio: float = 0.60,
    max_tortuosity: float = 4.0,
    low_prob: float = 0.30,
) -> list[TopologyEdge]:
    edges: list[TopologyEdge] = []
    for det in detections:
        start_xy, end_xy = det.keypoints_xy
        points = shortest_heatmap_path(
            heatmap,
            start_xy=start_xy,
            end_xy=end_xy,
            bbox_xyxy=det.bbox_xyxy,
        )
        mean_prob, min_prob, p20_prob, bad_ratio, length, tortuosity = _edge_stats(
            points,
            heatmap,
            low_prob=low_prob,
        )
        if mean_prob < min_mean_prob or bad_ratio > max_bad_ratio or tortuosity > max_tortuosity:
            continue
        edges.append(
            TopologyEdge(
                id=len(edges),
                start=-1,
                end=-1,
                points=points,
                source="yolo_pose",
                confidence=float(det.confidence),
                mean_prob=mean_prob,
                min_prob=min_prob,
                p20_prob=p20_prob,
                bad_ratio=bad_ratio,
                length=length,
                euclidean_length=hypot(end_xy[0] - start_xy[0], end_xy[1] - start_xy[1]),
                tortuosity=tortuosity,
            )
        )
    return edges


def build_topology_graph(
    candidate_edges: list[TopologyEdge],
    cluster_radius: float = 8.0,
) -> TopologyGraph:
    endpoints: list[tuple[float, float]] = []
    endpoint_edge_refs: list[tuple[int, str]] = []
    for edge_idx, edge in enumerate(candidate_edges):
        if len(edge.points) < 2:
            continue
        endpoint_edge_refs.append((edge_idx, "start"))
        endpoints.append(edge.points[0])
        endpoint_edge_refs.append((edge_idx, "end"))
        endpoints.append(edge.points[-1])

    if not endpoints:
        return TopologyGraph(nodes=[], edges=[])

    dsu = _DisjointSet(len(endpoints))
    points = np.asarray(endpoints, dtype=np.float32)
    pairs = cKDTree(points).query_pairs(r=cluster_radius)
    for left, right in pairs:
        dsu.union(left, right)

    groups: dict[int, list[int]] = {}
    for idx in range(len(endpoints)):
        groups.setdefault(dsu.find(idx), []).append(idx)

    endpoint_to_node: dict[int, int] = {}
    nodes: list[TopologyNode] = []
    for group in groups.values():
        arr = points[group]
        node_id = len(nodes)
        for idx in group:
            endpoint_to_node[idx] = node_id
        nodes.append(
            TopologyNode(
                id=node_id,
                xy=(float(arr[:, 0].mean()), float(arr[:, 1].mean())),
                confidence=1.0,
                size=len(group),
            )
        )

    edges = []
    for endpoint_idx, (edge_idx, side) in enumerate(endpoint_edge_refs):
        edge = candidate_edges[edge_idx]
        start_node = endpoint_to_node[endpoint_idx] if side == "start" else None
        end_node = endpoint_to_node[endpoint_idx] if side == "end" else None
        existing_idx = next(
            (
                i
                for i, (idx, prev_side) in enumerate(endpoint_edge_refs[:endpoint_idx])
                if idx == edge_idx and prev_side != side
            ),
            None,
        )
        if existing_idx is None:
            continue
        if side == "start":
            other = endpoint_to_node[existing_idx]
            start_node = endpoint_to_node[endpoint_idx]
            end_node = other
        else:
            other = endpoint_to_node[existing_idx]
            start_node = other
            end_node = endpoint_to_node[endpoint_idx]
        if start_node == end_node:
            continue
        edges.append(
            TopologyEdge(
                id=len(edges),
                start=int(start_node),
                end=int(end_node),
                points=edge.points,
                source=edge.source,
                confidence=edge.confidence,
                mean_prob=edge.mean_prob,
                min_prob=edge.min_prob,
                p20_prob=edge.p20_prob,
                bad_ratio=edge.bad_ratio,
                length=edge.length,
                euclidean_length=edge.euclidean_length,
                tortuosity=edge.tortuosity,
            )
        )

    return merge_duplicate_edges(TopologyGraph(nodes=nodes, edges=edges))


def merge_duplicate_edges(graph: TopologyGraph) -> TopologyGraph:
    best: dict[tuple[int, int], TopologyEdge] = {}
    for edge in graph.edges:
        key = tuple(sorted((edge.start, edge.end)))
        prev = best.get(key)
        score = edge.confidence * max(edge.mean_prob, 1e-6)
        prev_score = prev.confidence * max(prev.mean_prob, 1e-6) if prev is not None else -1.0
        if prev is None or score > prev_score:
            best[key] = edge
    edges = [
        TopologyEdge(
            id=i,
            start=edge.start,
            end=edge.end,
            points=edge.points,
            source=edge.source,
            confidence=edge.confidence,
            mean_prob=edge.mean_prob,
            min_prob=edge.min_prob,
            p20_prob=edge.p20_prob,
            bad_ratio=edge.bad_ratio,
            length=edge.length,
            euclidean_length=edge.euclidean_length,
            tortuosity=edge.tortuosity,
        )
        for i, edge in enumerate(best.values())
    ]
    return TopologyGraph(nodes=graph.nodes, edges=edges)


def prune_low_confidence_dangling_edges(
    graph: TopologyGraph,
    min_dangling_confidence: float = 0.20,
    min_dangling_mean_prob: float = 0.25,
    min_dangling_length: float = 12.0,
) -> TopologyGraph:
    degree = {node.id: 0 for node in graph.nodes}
    for edge in graph.edges:
        degree[edge.start] = degree.get(edge.start, 0) + 1
        degree[edge.end] = degree.get(edge.end, 0) + 1

    kept = []
    for edge in graph.edges:
        dangling = degree.get(edge.start, 0) <= 1 or degree.get(edge.end, 0) <= 1
        weak = edge.confidence < min_dangling_confidence or edge.mean_prob < min_dangling_mean_prob
        short = edge.length < min_dangling_length
        if dangling and weak and short:
            continue
        kept.append(edge)
    return TopologyGraph(nodes=graph.nodes, edges=[
        TopologyEdge(
            id=i,
            start=edge.start,
            end=edge.end,
            points=edge.points,
            source=edge.source,
            confidence=edge.confidence,
            mean_prob=edge.mean_prob,
            min_prob=edge.min_prob,
            p20_prob=edge.p20_prob,
            bad_ratio=edge.bad_ratio,
            length=edge.length,
            euclidean_length=edge.euclidean_length,
            tortuosity=edge.tortuosity,
        )
        for i, edge in enumerate(kept)
    ])


def recover_heatmap_short_edges(
    graph: TopologyGraph,
    heatmap: np.ndarray,
    max_length: float = 80.0,
    min_p20_score: float = 0.35,
    min_distance_to_existing: float = 5.0,
) -> list[TopologyEdge]:
    heatmap_graph = heatmap_to_graph(heatmap, blur_sigma=0.0)
    existing_samples = [
        sample_polyline(edge.points, step=2.0)
        for edge in graph.edges
        if len(edge.points) >= 2
    ]
    existing_points = np.concatenate(existing_samples, axis=0) if existing_samples else None
    tree = cKDTree(existing_points) if existing_points is not None and len(existing_points) else None
    recovered = []
    for edge in heatmap_graph.edges:
        if edge.length > max_length or edge.p20_score < min_p20_score:
            continue
        sampled = sample_polyline(edge.points, step=2.0)
        if tree is not None and len(sampled):
            distances, _ = tree.query(sampled, k=1)
            if float(np.mean(distances > min_distance_to_existing)) < 0.75:
                continue
        mean_prob, min_prob, p20_prob, bad_ratio, length, tortuosity = _edge_stats(
            edge.points,
            heatmap,
            low_prob=0.30,
        )
        recovered.append(
            TopologyEdge(
                id=-1,
                start=-1,
                end=-1,
                points=edge.points,
                source="heatmap_recovery",
                confidence=mean_prob,
                mean_prob=mean_prob,
                min_prob=min_prob,
                p20_prob=p20_prob,
                bad_ratio=bad_ratio,
                length=length,
                euclidean_length=hypot(
                    edge.points[-1][0] - edge.points[0][0],
                    edge.points[-1][1] - edge.points[0][1],
                ),
                tortuosity=tortuosity,
            )
        )
    return recovered


def build_yolo_heatmap_topology(
    detections: list[PoseDetection],
    heatmap: np.ndarray,
    cluster_radius: float = 8.0,
    recover_missing: bool = True,
    min_mean_prob: float = 0.25,
    max_bad_ratio: float = 0.60,
    max_tortuosity: float = 4.0,
    low_prob: float = 0.30,
) -> TopologyGraph:
    refined = refine_pose_detections(
        detections,
        heatmap,
        min_mean_prob=min_mean_prob,
        max_bad_ratio=max_bad_ratio,
        max_tortuosity=max_tortuosity,
        low_prob=low_prob,
    )
    graph = build_topology_graph(refined, cluster_radius=cluster_radius)
    graph = prune_low_confidence_dangling_edges(graph)
    if recover_missing:
        recovered = recover_heatmap_short_edges(graph, heatmap)
        if recovered:
            graph = build_topology_graph(graph.edges + recovered, cluster_radius=cluster_radius)
            graph = prune_low_confidence_dangling_edges(graph)
    return graph


def render_topology_overlay(
    image_rgb: np.ndarray,
    graph: TopologyGraph,
    line_color: tuple[int, int, int] = (255, 64, 0),
    node_color: tuple[int, int, int] = (0, 255, 255),
) -> np.ndarray:
    overlay = image_rgb.copy()
    for edge in graph.edges:
        pts = np.rint(np.asarray(edge.points, dtype=np.float32)).astype(np.int32)
        if len(pts) >= 2:
            cv2.polylines(overlay, [pts], False, line_color, 2, lineType=cv2.LINE_AA)
    for node in graph.nodes:
        cv2.circle(
            overlay,
            (int(round(node.xy[0])), int(round(node.xy[1]))),
            3,
            node_color,
            -1,
            lineType=cv2.LINE_AA,
        )
    return overlay
