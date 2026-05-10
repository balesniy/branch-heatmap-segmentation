from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from math import hypot

import networkx as nx
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import apply_hysteresis_threshold
from skimage.measure import approximate_polygon
from skimage.morphology import binary_closing, disk, remove_small_objects, skeletonize


NEIGHBORS_8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


@dataclass(frozen=True)
class GraphNode:
    id: int
    xy: tuple[float, float]
    kind: str
    confidence: float
    size: int


@dataclass(frozen=True)
class GraphEdge:
    id: int
    start: int | None
    end: int | None
    points: list[tuple[float, float]]
    length: float
    mean_score: float
    median_score: float
    p20_score: float
    min_score: float
    low_score_fraction: float


@dataclass(frozen=True)
class HeatmapGraph:
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    polylines: list[list[tuple[float, float]]]
    heatmap: np.ndarray
    mask: np.ndarray
    skeleton: np.ndarray

    def to_json_dict(self) -> dict:
        return {
            "nodes": [asdict(node) for node in self.nodes],
            "edges": [asdict(edge) for edge in self.edges],
            "polylines": self.polylines,
        }


def _neighbors(point: tuple[int, int], skel: np.ndarray) -> list[tuple[int, int]]:
    y, x = point
    height, width = skel.shape
    out = []
    for dy, dx in NEIGHBORS_8:
        yy, xx = y + dy, x + dx
        if 0 <= yy < height and 0 <= xx < width and skel[yy, xx]:
            out.append((yy, xx))
    return out


def _edge_key(
    a: tuple[int, int], b: tuple[int, int]
) -> tuple[tuple[int, int], tuple[int, int]]:
    return tuple(sorted((a, b)))


def _degree_map(skel: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    return ndi.convolve(skel.astype(np.uint8), kernel, mode="constant", cval=0)


def ridge_nms(heatmap: np.ndarray, threshold: float = 0.2, nms_size: int = 3) -> np.ndarray:
    """Keep only local heatmap maxima."""
    local_max = ndi.maximum_filter(heatmap, size=nms_size, mode="nearest") == heatmap
    local_min = ndi.minimum_filter(heatmap, size=nms_size, mode="nearest")
    non_flat = heatmap > local_min
    return (heatmap > threshold) & local_max & non_flat


def _remove_small_objects(mask: np.ndarray, min_size: int) -> np.ndarray:
    try:
        return remove_small_objects(mask, max_size=max(0, min_size - 1), connectivity=2)
    except TypeError:
        return remove_small_objects(mask, min_size=min_size, connectivity=2)


def _polyline_length(points_xy: np.ndarray) -> float:
    if len(points_xy) < 2:
        return 0.0
    delta = np.diff(points_xy, axis=0)
    return float(np.hypot(delta[:, 0], delta[:, 1]).sum())


def _line_sample_scores(points_yx: list[tuple[int, int]], heatmap: np.ndarray) -> np.ndarray:
    if not points_yx:
        return np.zeros((0,), dtype=np.float32)
    yy = np.array([p[0] for p in points_yx], dtype=np.intp)
    xx = np.array([p[1] for p in points_yx], dtype=np.intp)
    return heatmap[yy, xx].astype(np.float32)


def _node_kind(component_degrees: np.ndarray) -> str:
    if np.any(component_degrees >= 3):
        return "junction"
    if np.any(component_degrees == 1):
        return "endpoint"
    return "isolated"


def _build_nodes(
    skel: np.ndarray, heatmap: np.ndarray, degree: np.ndarray
) -> tuple[list[GraphNode], np.ndarray, dict[int, int]]:
    key_mask = skel & (degree != 2)
    labels, label_count = ndi.label(key_mask, structure=np.ones((3, 3), dtype=np.uint8))
    nodes: list[GraphNode] = []
    label_to_node: dict[int, int] = {}

    for label_id in range(1, label_count + 1):
        ys, xs = np.nonzero(labels == label_id)
        if len(ys) == 0:
            continue
        node_id = len(nodes)
        label_to_node[label_id] = node_id
        scores = heatmap[ys, xs]
        xy = (float(xs.mean()), float(ys.mean()))
        kind = _node_kind(degree[ys, xs])
        nodes.append(
            GraphNode(
                id=node_id,
                xy=xy,
                kind=kind,
                confidence=float(scores.mean()) if len(scores) else 0.0,
                size=int(len(ys)),
            )
        )
    return nodes, labels, label_to_node


def _make_edge(
    edge_id: int,
    start: int | None,
    end: int | None,
    points_yx: list[tuple[int, int]],
    heatmap: np.ndarray,
    simplify_tol: float | None,
) -> GraphEdge:
    points_xy = np.array(points_yx, dtype=np.float32)[:, ::-1]
    if simplify_tol is not None and len(points_xy) >= 3:
        points_xy = approximate_polygon(points_xy, tolerance=simplify_tol).astype(
            np.float32
        )
    scores = _line_sample_scores(points_yx, heatmap)
    return GraphEdge(
        id=edge_id,
        start=start,
        end=end,
        points=[(float(x), float(y)) for x, y in points_xy],
        length=_polyline_length(points_xy),
        mean_score=float(scores.mean()) if len(scores) else 0.0,
        median_score=float(np.median(scores)) if len(scores) else 0.0,
        p20_score=float(np.percentile(scores, 20)) if len(scores) else 0.0,
        min_score=float(scores.min()) if len(scores) else 0.0,
        low_score_fraction=float(np.mean(scores < 0.2)) if len(scores) else 0.0,
    )


def _renumber_edges(edges: list[GraphEdge]) -> list[GraphEdge]:
    return [
        GraphEdge(
            i,
            edge.start,
            edge.end,
            edge.points,
            edge.length,
            edge.mean_score,
            edge.median_score,
            edge.p20_score,
            edge.min_score,
            edge.low_score_fraction,
        )
        for i, edge in enumerate(edges)
    ]


def _with_edges(graph: HeatmapGraph, edges: list[GraphEdge]) -> HeatmapGraph:
    edges = _renumber_edges(edges)
    return HeatmapGraph(
        nodes=graph.nodes,
        edges=edges,
        polylines=[edge.points for edge in edges],
        heatmap=graph.heatmap,
        mask=graph.mask,
        skeleton=graph.skeleton,
    )


def skeleton_to_graph(
    skel: np.ndarray,
    heatmap: np.ndarray,
    min_len: float = 5.0,
    simplify_tol: float | None = 1.0,
) -> HeatmapGraph:
    skel = skel.astype(bool)
    degree = _degree_map(skel)
    nodes, key_labels, label_to_node = _build_nodes(skel, heatmap, degree)
    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    edges: list[GraphEdge] = []

    for label_id, start_node in label_to_node.items():
        component_pixels = list(zip(*np.nonzero(key_labels == label_id)))
        for start in component_pixels:
            for nb in _neighbors(start, skel):
                if key_labels[nb] == label_id:
                    continue
                first_key = _edge_key(start, nb)
                if first_key in visited:
                    continue

                visited.add(first_key)
                points = [start, nb]
                prev, cur = start, nb
                end_node: int | None = None

                while True:
                    cur_label = int(key_labels[cur])
                    if cur_label > 0:
                        end_node = label_to_node[cur_label]
                        break
                    next_pixels = [pix for pix in _neighbors(cur, skel) if pix != prev]
                    if not next_pixels:
                        break
                    next_pixels.sort(key=lambda pix: float(heatmap[pix]), reverse=True)
                    nxt = next_pixels[0]
                    ek = _edge_key(cur, nxt)
                    if ek in visited:
                        break
                    visited.add(ek)
                    points.append(nxt)
                    prev, cur = cur, nxt

                edge = _make_edge(
                    len(edges),
                    start_node,
                    end_node,
                    points,
                    heatmap,
                    simplify_tol,
                )
                if edge.length >= min_len:
                    edges.append(edge)

    # Closed loops have no endpoints or junctions, so trace any unvisited degree-2 cycle.
    for start in zip(*np.nonzero(skel & (degree == 2))):
        ns = _neighbors(start, skel)
        if not ns:
            continue
        if all(_edge_key(start, nb) in visited for nb in ns):
            continue
        points = [start]
        prev, cur = start, ns[0]
        visited.add(_edge_key(prev, cur))
        while cur != start:
            points.append(cur)
            next_pixels = [pix for pix in _neighbors(cur, skel) if pix != prev]
            if not next_pixels:
                break
            nxt = next_pixels[0]
            ek = _edge_key(cur, nxt)
            if ek in visited and nxt != start:
                break
            visited.add(ek)
            prev, cur = cur, nxt
        edge = _make_edge(len(edges), None, None, points, heatmap, simplify_tol)
        if edge.length >= min_len:
            edges.append(edge)

    return HeatmapGraph(
        nodes=nodes,
        edges=edges,
        polylines=[edge.points for edge in edges],
        heatmap=heatmap,
        mask=skel,
        skeleton=skel,
    )


def _endpoint_tangent(edge: GraphEdge, node_id: int) -> tuple[float, float] | None:
    if len(edge.points) < 2:
        return None
    if edge.start == node_id:
        a, b = edge.points[0], edge.points[min(3, len(edge.points) - 1)]
    elif edge.end == node_id:
        a, b = edge.points[-1], edge.points[max(0, len(edge.points) - 4)]
    else:
        return None
    dx, dy = b[0] - a[0], b[1] - a[1]
    norm = hypot(dx, dy)
    if norm == 0:
        return None
    return dx / norm, dy / norm


def _angle_degrees(a: tuple[float, float], b: tuple[float, float]) -> float:
    dot = max(-1.0, min(1.0, a[0] * b[0] + a[1] * b[1]))
    return float(np.degrees(np.arccos(dot)))


def prune_short_low_score_spurs(
    graph: HeatmapGraph,
    min_length: float = 12.0,
    min_p20_score: float = 0.25,
) -> HeatmapGraph:
    endpoint_nodes = {node.id for node in graph.nodes if node.kind == "endpoint"}
    kept = []
    for edge in graph.edges:
        touches_endpoint = edge.start in endpoint_nodes or edge.end in endpoint_nodes
        should_prune = (
            touches_endpoint
            and edge.length < min_length
            and edge.p20_score < min_p20_score
        )
        if not should_prune:
            kept.append(edge)
    return _with_edges(graph, kept)


def filter_weak_edges(
    graph: HeatmapGraph,
    min_length: float = 5.0,
    min_p20_score: float = 0.0,
    max_low_score_fraction: float = 1.0,
    keep_longer_than: float = 80.0,
) -> HeatmapGraph:
    kept = []
    for edge in graph.edges:
        if edge.length < min_length:
            continue
        score_ok = (
            edge.p20_score >= min_p20_score
            and edge.low_score_fraction <= max_low_score_fraction
        )
        if score_ok or edge.length >= keep_longer_than:
            kept.append(edge)
    return _with_edges(graph, kept)


def _draw_line_scores(
    a: tuple[float, float], b: tuple[float, float], heatmap: np.ndarray
) -> np.ndarray:
    distance = max(1, int(round(hypot(b[0] - a[0], b[1] - a[1]))))
    xs = np.linspace(a[0], b[0], distance + 1)
    ys = np.linspace(a[1], b[1], distance + 1)
    xi = np.clip(np.rint(xs).astype(np.intp), 0, heatmap.shape[1] - 1)
    yi = np.clip(np.rint(ys).astype(np.intp), 0, heatmap.shape[0] - 1)
    return heatmap[yi, xi]


def bridge_endpoint_gaps(
    graph: HeatmapGraph,
    gap_max: float = 6.0,
    angle_max: float = 40.0,
    min_bridge_score: float = 0.2,
) -> HeatmapGraph:
    endpoint_nodes = [node for node in graph.nodes if node.kind == "endpoint"]
    incident: dict[int, list[GraphEdge]] = {node.id: [] for node in endpoint_nodes}
    for edge in graph.edges:
        if edge.start in incident:
            incident[edge.start].append(edge)
        if edge.end in incident:
            incident[edge.end].append(edge)

    new_edges = list(graph.edges)
    bridged_nodes: set[int] = set()
    for left, right in combinations(endpoint_nodes, 2):
        if left.id in bridged_nodes or right.id in bridged_nodes:
            continue
        if not incident[left.id] or not incident[right.id]:
            continue
        distance = hypot(right.xy[0] - left.xy[0], right.xy[1] - left.xy[1])
        if distance > gap_max:
            continue
        bridge_direction = (
            (right.xy[0] - left.xy[0]) / max(distance, 1e-6),
            (right.xy[1] - left.xy[1]) / max(distance, 1e-6),
        )
        right_bridge_direction = (-bridge_direction[0], -bridge_direction[1])

        left_angles = [
            _angle_degrees(tangent, bridge_direction)
            for edge in incident[left.id]
            if (tangent := _endpoint_tangent(edge, left.id)) is not None
        ]
        right_angles = [
            _angle_degrees(tangent, right_bridge_direction)
            for edge in incident[right.id]
            if (tangent := _endpoint_tangent(edge, right.id)) is not None
        ]
        if not left_angles or not right_angles:
            continue
        if min(left_angles) > angle_max or min(right_angles) > angle_max:
            continue

        scores = _draw_line_scores(left.xy, right.xy, graph.heatmap)
        if len(scores) and float(np.percentile(scores, 20)) < min_bridge_score:
            continue
        points = [left.xy, right.xy]
        new_edges.append(
            GraphEdge(
                id=len(new_edges),
                start=left.id,
                end=right.id,
                points=points,
                length=float(distance),
                mean_score=float(scores.mean()) if len(scores) else 0.0,
                median_score=float(np.median(scores)) if len(scores) else 0.0,
                p20_score=float(np.percentile(scores, 20)) if len(scores) else 0.0,
                min_score=float(scores.min()) if len(scores) else 0.0,
                low_score_fraction=float(np.mean(scores < 0.2)) if len(scores) else 0.0,
            )
        )
        bridged_nodes.add(left.id)
        bridged_nodes.add(right.id)

    return _with_edges(graph, new_edges)


def _snap_point_to_ridge(
    xy: tuple[float, float],
    heatmap: np.ndarray,
    radius: int,
) -> tuple[float, float]:
    if radius <= 0:
        return xy
    x, y = xy
    cx = int(round(x))
    cy = int(round(y))
    y0 = max(0, cy - radius)
    y1 = min(heatmap.shape[0], cy + radius + 1)
    x0 = max(0, cx - radius)
    x1 = min(heatmap.shape[1], cx + radius + 1)
    patch = heatmap[y0:y1, x0:x1]
    if patch.size == 0:
        return xy
    yy, xx = np.unravel_index(int(np.argmax(patch)), patch.shape)
    return float(x0 + xx), float(y0 + yy)


def snap_graph_to_heatmap_ridge(graph: HeatmapGraph, radius: int = 1) -> HeatmapGraph:
    if radius <= 0:
        return graph
    snapped_edges = []
    for edge in graph.edges:
        if len(edge.points) < 2:
            snapped_edges.append(edge)
            continue
        points = [
            _snap_point_to_ridge(point, graph.heatmap, radius=radius)
            for point in edge.points
        ]
        points_xy = np.asarray(points, dtype=np.float32)
        snapped_edges.append(
            GraphEdge(
                id=edge.id,
                start=edge.start,
                end=edge.end,
                points=[(float(x), float(y)) for x, y in points_xy],
                length=_polyline_length(points_xy),
                mean_score=edge.mean_score,
                median_score=edge.median_score,
                p20_score=edge.p20_score,
                min_score=edge.min_score,
                low_score_fraction=edge.low_score_fraction,
            )
        )

    return _with_edges(graph, snapped_edges)


class _DisjointSet:
    def __init__(self, items: list[int]):
        self.parent = {item: item for item in items}

    def find(self, item: int) -> int:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def collapse_small_cycles(graph: HeatmapGraph, max_cycle_length: float = 15.0) -> HeatmapGraph:
    if max_cycle_length <= 0 or len(graph.nodes) < 3:
        return graph

    node_ids = [node.id for node in graph.nodes]
    dsu = _DisjointSet(node_ids)
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(node_ids)
    for edge in graph.edges:
        if edge.start is None or edge.end is None or edge.start == edge.end:
            continue
        nx_graph.add_edge(edge.start, edge.end, length=edge.length)

    for cycle in nx.cycle_basis(nx_graph):
        if len(cycle) < 3:
            continue
        perimeter = 0.0
        for left, right in zip(cycle, cycle[1:] + cycle[:1]):
            perimeter += float(nx_graph[left][right].get("length", 1.0))
        if perimeter > max_cycle_length:
            continue
        anchor = cycle[0]
        for node_id in cycle[1:]:
            dsu.union(anchor, node_id)

    groups: dict[int, list[GraphNode]] = {}
    for node in graph.nodes:
        groups.setdefault(dsu.find(node.id), []).append(node)

    old_to_new: dict[int, int] = {}
    new_nodes: list[GraphNode] = []
    collapsed_centroids: dict[int, tuple[float, float]] = {}
    for group in groups.values():
        if len(group) == 1:
            node = group[0]
            old_to_new[node.id] = len(new_nodes)
            new_nodes.append(
                GraphNode(
                    id=len(new_nodes),
                    xy=node.xy,
                    kind=node.kind,
                    confidence=node.confidence,
                    size=node.size,
                )
            )
            continue

        xs = [node.xy[0] for node in group]
        ys = [node.xy[1] for node in group]
        centroid = (float(np.mean(xs)), float(np.mean(ys)))
        new_id = len(new_nodes)
        for node in group:
            old_to_new[node.id] = new_id
            collapsed_centroids[node.id] = centroid
        new_nodes.append(
            GraphNode(
                id=new_id,
                xy=centroid,
                kind="junction",
                confidence=float(np.mean([node.confidence for node in group])),
                size=int(sum(node.size for node in group)),
            )
        )

    new_edges = []
    for edge in graph.edges:
        start = old_to_new.get(edge.start) if edge.start is not None else None
        end = old_to_new.get(edge.end) if edge.end is not None else None
        if start is not None and end is not None and start == end:
            continue
        points = list(edge.points)
        if edge.start in collapsed_centroids and points:
            points[0] = collapsed_centroids[edge.start]
        if edge.end in collapsed_centroids and points:
            points[-1] = collapsed_centroids[edge.end]
        points_xy = np.asarray(points, dtype=np.float32)
        new_edges.append(
            GraphEdge(
                id=edge.id,
                start=start,
                end=end,
                points=[(float(x), float(y)) for x, y in points_xy],
                length=_polyline_length(points_xy),
                mean_score=edge.mean_score,
                median_score=edge.median_score,
                p20_score=edge.p20_score,
                min_score=edge.min_score,
                low_score_fraction=edge.low_score_fraction,
            )
        )

    return HeatmapGraph(
        nodes=new_nodes,
        edges=_renumber_edges(new_edges),
        polylines=[edge.points for edge in new_edges],
        heatmap=graph.heatmap,
        mask=graph.mask,
        skeleton=graph.skeleton,
    )


def heatmap_to_graph(
    heatmap: np.ndarray,
    low: float = 0.35,
    high: float = 0.5,
    blur_sigma: float = 0.5,
    min_object_size: int = 4,
    closing_radius: int = 0,
    centerline_mode: str = "ridge_skeleton",
    ridge_nms_size: int = 3,
    min_len: float = 8.0,
    simplify_tol: float | None = 1.0,
    spur_min_length: float = 10.0,
    spur_min_p20_score: float = 0.25,
    edge_min_p20_score: float = 0.0,
    edge_max_low_score_fraction: float = 1.0,
    edge_keep_longer_than: float = 80.0,
    cycle_collapse_max_length: float = 15.0,
    bridge_gaps: bool = True,
    gap_max: float = 6.0,
    angle_max: float = 40.0,
    min_bridge_score: float = 0.2,
    snap_radius: int = 2,
) -> HeatmapGraph:
    heatmap = np.clip(heatmap.astype(np.float32), 0.0, 1.0)
    if blur_sigma > 0:
        heatmap = ndi.gaussian_filter(heatmap, sigma=blur_sigma)
    mask = apply_hysteresis_threshold(heatmap, low, high)
    if min_object_size > 0:
        mask = _remove_small_objects(mask, min_object_size)
    if closing_radius > 0:
        mask = binary_closing(mask, footprint=disk(closing_radius))
    if centerline_mode == "skeleton":
        skeleton = skeletonize(mask)
    elif centerline_mode == "ridge":
        skeleton = ridge_nms(heatmap, threshold=low, nms_size=ridge_nms_size) & mask
        if min_object_size > 0:
            skeleton = _remove_small_objects(skeleton, min_object_size)
    elif centerline_mode == "ridge_skeleton":
        ridge_support = ndi.binary_dilation(
            ridge_nms(heatmap, threshold=low, nms_size=ridge_nms_size),
            iterations=1,
        )
        skeleton = skeletonize(mask & ridge_support)
    else:
        raise ValueError(f"Unknown centerline_mode: {centerline_mode}")
    graph = skeleton_to_graph(skeleton, heatmap, min_len=min_len, simplify_tol=simplify_tol)
    graph = HeatmapGraph(
        nodes=graph.nodes,
        edges=graph.edges,
        polylines=graph.polylines,
        heatmap=heatmap,
        mask=mask,
        skeleton=skeleton,
    )
    graph = collapse_small_cycles(graph, max_cycle_length=cycle_collapse_max_length)
    graph = prune_short_low_score_spurs(
        graph,
        min_length=spur_min_length,
        min_p20_score=spur_min_p20_score,
    )
    graph = filter_weak_edges(
        graph,
        min_length=min_len,
        min_p20_score=edge_min_p20_score,
        max_low_score_fraction=edge_max_low_score_fraction,
        keep_longer_than=edge_keep_longer_than,
    )
    if bridge_gaps:
        graph = bridge_endpoint_gaps(
            graph,
            gap_max=gap_max,
            angle_max=angle_max,
            min_bridge_score=min_bridge_score,
        )
    graph = snap_graph_to_heatmap_ridge(graph, radius=snap_radius)
    return graph


def render_graph_overlay(
    image_rgb: np.ndarray,
    graph: HeatmapGraph,
    line_color: tuple[int, int, int] = (255, 64, 0),
    node_color: tuple[int, int, int] = (0, 255, 255),
) -> np.ndarray:
    import cv2

    overlay = image_rgb.copy()
    for edge in graph.edges:
        pts = np.rint(np.array(edge.points, dtype=np.float32)).astype(np.int32)
        if len(pts) >= 2:
            cv2.polylines(
                overlay,
                [pts],
                isClosed=False,
                color=line_color,
                thickness=2,
                lineType=cv2.LINE_AA,
            )
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
