from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from branch_segmentation import BranchHeatmapLoss, BranchHeatmapNet
from branch_segmentation.dataset import BranchPolylineDataset
from branch_segmentation.metrics import centerline_metrics


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_optimizer(model: BranchHeatmapNet, cfg: dict) -> torch.optim.Optimizer:
    encoder_params = []
    decoder_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("encoder."):
            encoder_params.append(param)
        else:
            decoder_params.append(param)
    return torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": cfg["train"]["encoder_lr"]},
            {"params": decoder_params, "lr": cfg["train"]["decoder_lr"]},
        ],
        weight_decay=cfg["train"]["weight_decay"],
    )


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, grad_clip, use_amp):
    model.train()
    running = 0.0
    pbar = tqdm(loader, desc="train", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        gaussian = batch["gaussian"].to(device, non_blocking=True)
        center = batch["center"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            preds = model(images, return_side_outputs=True)
            loss = criterion(preds, gaussian, center)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        running += float(loss.detach().cpu())
        pbar.set_postfix(loss=running / max(1, pbar.n))
    return running / max(1, len(loader))


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp):
    model.eval()
    running = 0.0
    totals: dict[str, float] = {}
    for batch in tqdm(loader, desc="val", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        gaussian = batch["gaussian"].to(device, non_blocking=True)
        center = batch["center"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            preds = model(images, return_side_outputs=True)
            loss = criterion(preds, gaussian, center)
        pred_prob = torch.sigmoid(preds[0])
        metrics = centerline_metrics(pred_prob, center, tolerance_radius=2)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        running += float(loss.cpu())
    n = max(1, len(loader))
    return running / n, {key: value / n for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    seed_everything(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg["train"]["amp"] and device.type == "cuda")
    output_dir = Path(cfg["train"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    image_size = tuple(cfg["data"]["image_size"])
    train_ds = BranchPolylineDataset(
        cfg["data"]["train_images"],
        cfg["data"]["train_annotations"],
        image_size=image_size,
        train=True,
        gaussian_sigma=cfg["data"]["gaussian_sigma"],
        line_thickness=cfg["data"]["line_thickness"],
        render_scale=cfg["data"]["render_scale"],
    )
    val_ds = BranchPolylineDataset(
        cfg["data"]["val_images"],
        cfg["data"]["val_annotations"],
        image_size=image_size,
        train=False,
        gaussian_sigma=cfg["data"]["gaussian_sigma"],
        line_thickness=cfg["data"]["line_thickness"],
        render_scale=cfg["data"]["render_scale"],
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=device.type == "cuda",
    )

    model = BranchHeatmapNet(**cfg["model"]).to(device)
    criterion = BranchHeatmapLoss(**cfg["loss"]).to(device)
    optimizer = make_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["train"]["epochs"]
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_score = -1.0
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            cfg["train"]["grad_clip"],
            use_amp,
        )
        val_loss, val_metrics = validate(model, val_loader, criterion, device, use_amp)
        scheduler.step()

        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"precision@2px={val_metrics['precision@2px']:.4f} "
            f"recall@2px={val_metrics['recall@2px']:.4f} "
            f"cldice@2px={val_metrics['cldice@2px']:.4f} "
            f"f1={val_metrics['f1']:.4f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": cfg,
            "val_metrics": val_metrics,
        }
        if epoch % cfg["train"]["save_every"] == 0:
            torch.save(checkpoint, output_dir / f"epoch_{epoch:03d}.pt")
        if val_metrics["cldice@2px"] > best_score:
            best_score = val_metrics["cldice@2px"]
            torch.save(checkpoint, output_dir / "best.pt")


if __name__ == "__main__":
    main()
