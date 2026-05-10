# Branch Heatmap Segmentation

Pipeline for training a thin tree-branch segmentation network:

- SegFormer MiT encoder.
- FPN-style CLNet decoder with strip/local/dilated convolutions.
- HR stem at full resolution for thin branch detail.
- DRIU-style side outputs.
- Two targets: binary `center_mask` and soft `gaussian_target`.
- Loss: side MSE + main heatmap MSE + centerline BCE-with-logits + soft recall.

## Data Format

Expected layout:

```text
data/
  train/
    images/
      img_001.jpg
    annotations/
      img_001.json
  val/
    images/
    annotations/
```

Annotation JSON can use either `branches` or `polylines`:

```json
{
  "branches": [
    [[10, 40], [25, 55], [40, 80]],
    [[80, 10], [90, 35], [120, 60]]
  ]
}
```

The dataset applies geometric augmentation to image + polyline keypoints, then renders:

1. `center_mask`: binary centerline.
2. `gaussian_target`: distance-transform Gaussian around the rendered centerline.

This avoids corrupting Gaussian heatmaps with nearest-neighbor mask transforms.

## Install

```bash
pip install -r requirements.txt
```

If you run scripts directly from the repo:

```bash
export PYTHONPATH="$PWD/src"
```

## Train

Edit `configs/train.yaml`, then run:

```bash
PYTHONPATH=src python scripts/train.py --config configs/train.yaml
```

Best checkpoint is saved to:

```text
outputs/branch_heatmap/best.pt
```

## Inference

```bash
PYTHONPATH=src python scripts/infer.py \
  --checkpoint outputs/branch_heatmap/best.pt \
  --image path/to/image.jpg \
  --output outputs/prediction.png
```

This writes:

- `prediction.heatmap.png`
- `prediction.mask.png` from hysteresis thresholding
- `prediction.skeleton.png`
- `prediction.graph.png` overlay with traced polylines and graph nodes
- `prediction.graph.json` with nodes, edges, polyline points, length, mean score,
  median score, 20th-percentile score, minimum score, and low-score fraction

The default inference path is heatmap-to-graph:

```text
model + TTA
-> clipped and lightly blurred heatmap
-> hysteresis threshold
-> ridge-NMS constrained skeleton
-> junction-cluster graph construction
-> small-cycle junction cleanup
-> polyline tracing
-> short low-score spur pruning
-> small endpoint-gap bridging
-> RDP simplification
-> snap simplified points back to heatmap ridge
```

Useful knobs:

```bash
PYTHONPATH=src python scripts/infer.py \
  --checkpoint outputs/branch_heatmap/best.pt \
  --image path/to/image.jpg \
  --output outputs/prediction.png \
  --low-thr 0.35 \
  --high-thr 0.5 \
  --centerline-mode ridge_skeleton \
  --ridge-nms-size 3 \
  --snap-radius 2 \
  --simplify-tol 1.0 \
  --tta original hflip vflip rot90 rot180 rot270
```

## Vectorization Tuning

For postprocess-only tuning, reuse saved `.heatmap.png` files and run:

```bash
PYTHONPATH=src python scripts/tune_vectorization.py \
  --images-dir data/val/images \
  --heatmap-dir outputs/infer_val \
  --coco-json data/annotations_coco_polyline.json \
  --output outputs/vector_tuning_summary.json
```

`edge_f1` is length-weighted KD-tree buffer coverage over sampled polylines. It
does not penalize a visually correct branch just because graph tracing split it
into several segments. The older one-to-one edge matcher is still available as
`edge_instance_f1` for diagnosing fragmentation.

## Notes

- Start with `hr_channels: 24`. If full-resolution images cause OOM, try `16`.
- `pos_weight` and `recall_weight` are the first loss knobs to tune.
- Validation keeps pixel F1 as a sanity check, but selects `best.pt` by
  `clDice@2px`, which is more stable for thin centerlines.
- For final graph model selection, combine graph/vector metrics instead of relying
  on IoU/Dice alone:

```text
primary:   APLS-like graph score
secondary: clDice@2px, edge F1@3px, endpoint F1@5px, junction F1@5px
geometry:  matched-edge Chamfer, Hausdorff p95, relative length error
```

- If the downstream goal is graph reconstruction, add auxiliary junction/endpoint
  heads and an orientation field later.
