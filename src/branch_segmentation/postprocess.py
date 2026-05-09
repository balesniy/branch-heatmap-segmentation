from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from math import hypot

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import apply_hysteresis_threshold
from skimage.measure import approximate_polygon
from skimage.morphology import remove_small_objects, skeletonize


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
    return HeatmapGraph(
        nodes=graph.nodes,
        edges=[
            GraphEdge(
                i,
                e.start,
                e.end,
                e.points,
                e.length,
                e.mean_score,
                e.median_score,
                e.p20_score,
                e.min_score,
                e.low_score_fraction,
            )
            for i, e in enumerate(kept)
        ],
        polylines=[edge.points for edge in kept],
        heatmap=graph.heatmap,
        mask=graph.mask,
        skeleton=graph.skeleton,
    )


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
    for left, right in combinations(endpoint_nodes, 2):
        if not incident[left.id] or not incident[right.id]:
            continue
        distance = hypot(right.xy[0] - left.xy[0], right.xy[1] - left.xy[1])
        if distance > gap_max:
            continue
        left_tangent = _endpoint_tangent(incident[left.id][0], left.id)
        right_tangent = _endpoint_tangent(incident[right.id][0], right.id)
        if left_tangent is None or right_tangent is None:
            continue
        bridge_direction = (
            (right.xy[0] - left.xy[0]) / max(distance, 1e-6),
            (right.xy[1] - left.xy[1]) / max(distance, 1e-6),
        )
        right_bridge_direction = (-bridge_direction[0], -bridge_direction[1])
        if _angle_degrees(left_tangent, bridge_direction) > angle_max:
            continue
        if _angle_degrees(right_tangent, right_bridge_direction) > angle_max:
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

    return HeatmapGraph(
        nodes=graph.nodes,
        edges=[
            GraphEdge(
                i,
                e.start,
                e.end,
                e.points,
                e.length,
                e.mean_score,
                e.median_score,
                e.p20_score,
                e.min_score,
                e.low_score_fraction,
            )
            for i, e in enumerate(new_edges)
        ],
        polylines=[edge.points for edge in new_edges],
        heatmap=graph.heatmap,
        mask=graph.mask,
        skeleton=graph.skeleton,
    )


def heatmap_to_graph(
    heatmap: np.ndarray,
    low: float = 0.2,
    high: float = 0.5,
    blur_sigma: float = 0.5,
    min_object_size: int = 8,
    min_len: float = 5.0,
    simplify_tol: float | None = 1.0,
    spur_min_length: float = 12.0,
    spur_min_p20_score: float = 0.25,
    bridge_gaps: bool = True,
    gap_max: float = 6.0,
    angle_max: float = 40.0,
    min_bridge_score: float = 0.2,
) -> HeatmapGraph:
    heatmap = np.clip(heatmap.astype(np.float32), 0.0, 1.0)
    if blur_sigma > 0:
        heatmap = ndi.gaussian_filter(heatmap, sigma=blur_sigma)
    mask = apply_hysteresis_threshold(heatmap, low, high)
    if min_object_size > 0:
        mask = _remove_small_objects(mask, min_object_size)
    skeleton = skeletonize(mask)
    graph = skeleton_to_graph(skeleton, heatmap, min_len=min_len, simplify_tol=simplify_tol)
    graph = HeatmapGraph(
        nodes=graph.nodes,
        edges=graph.edges,
        polylines=graph.polylines,
        heatmap=heatmap,
        mask=mask,
        skeleton=skeleton,
    )
    graph = prune_short_low_score_spurs(
        graph,
        min_length=spur_min_length,
        min_p20_score=spur_min_p20_score,
    )
    if bridge_gaps:
        graph = bridge_endpoint_gaps(
            graph,
            gap_max=gap_max,
            angle_max=angle_max,
            min_bridge_score=min_bridge_score,
        )
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
