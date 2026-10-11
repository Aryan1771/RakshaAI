"""Staged R3D-18 fine-tuning with validation-only checkpoint selection."""
from collections import Counter
from datetime import datetime, timezone
from importlib import metadata
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from .metrics import classification_metrics
from .models import FocalLoss, build_r3d18
from .video_data import VideoWindowDataset, build_window_manifest, load_jsonl


def seed_everything(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = bool(deterministic)


def _worker_seed(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def _counts(rows):
    return Counter(int(r["target"]) for r in rows)


def _loss_and_sampling(train_rows, variant, device, gamma=2.0):
    counts = _counts(train_rows)
    if set(counts) != {0, 1}:
        raise ValueError(f"Training requires both reviewed classes; found {dict(counts)}")
    if variant not in {"baseline", "class_weighted", "balanced_sampling", "focal_loss"}:
        raise ValueError(f"Unsupported one-factor R3D variant: {variant}")
    weights = torch.tensor([sum(counts.values()) / (2 * counts[i]) for i in (0, 1)], dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights if variant == "class_weighted" else None)
    if variant == "focal_loss":
        criterion = FocalLoss(gamma=gamma)
    sampler = None
    if variant == "balanced_sampling":
        per_row = [1.0 / counts[int(r["target"])] for r in train_rows]
        sampler = WeightedRandomSampler(per_row, num_samples=len(per_row), replacement=True)
    return criterion, sampler, dict(counts)


def _evaluate(model, loader, device, criterion, amp=True):
    model.eval()
    losses, targets, probabilities, metadata = [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            video = batch["video"].to(device, non_blocking=True)
            target = batch["label"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda" and amp):
                logits = model(video)
                loss = criterion(logits, target)
            probs = torch.softmax(logits.float(), dim=1)[:, 1]
            losses.append(float(loss.item()))
            targets.extend(target.cpu().tolist())
            probabilities.extend(probs.cpu().tolist())
            for i in range(len(target)):
                metadata.append({"record_id": batch["record_id"][i], "incident_id": batch["incident_id"][i],
                                 "window_start_seconds": float(batch["window_start_seconds"][i]),
                                 "window_end_seconds": float(batch["window_end_seconds"][i]),
                                 "temporal_known": bool(batch["temporal_known"][i]),
                                 "camera_id": batch["camera_id"][i]})
    if not targets:
        raise ValueError("Validation set has no decodable windows")
    metrics = classification_metrics(targets, probabilities, .5)
    metrics["loss"] = float(np.mean(losses))
    return metrics, targets, probabilities, metadata


def _save_checkpoint(path, *, model, optimizer, scaler, epoch, stage, best_metric, config, model_info, variant):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(), "epoch": epoch, "stage": stage,
                "best_validation_average_precision": best_metric,
                "config": config, "model_info": model_info, "variant": variant,
                "torch_version": torch.__version__, "created_at": datetime.now(timezone.utc).isoformat()}, path)


def train_r3d(config, video_manifest, *, output_dir, variant="baseline", resume=None):
    seed = int(config.get("seed", 42))
    seed_everything(seed, config.get("deterministic", True))
    vcfg, mcfg = config["video"], config["r3d"]
    source_rows = load_jsonl(video_manifest)
    eligible_rows = [r for r in source_rows if r.get("split") in {"train", "validation"}]
    if not eligible_rows:
        raise ValueError("No pre-split eligible human-reviewed Indian clips; run manifest-phase1 and review labels before training")
    train_source = [r for r in eligible_rows if r["split"] == "train"]
    val_source = [r for r in eligible_rows if r["split"] == "validation"]
    train_rows, train_rejected = build_window_manifest(train_source,
        window_seconds=float(vcfg["temporal_span_seconds"]), stride_seconds=float(vcfg["window_stride_seconds"]),
        boundary_guard=float(vcfg.get("boundary_guard_seconds", .5)),
        positive_overlap_fraction=float(vcfg.get("positive_overlap_fraction", .5)))
    val_rows, val_rejected = build_window_manifest(val_source,
        window_seconds=float(vcfg["temporal_span_seconds"]), stride_seconds=float(vcfg["window_stride_seconds"]),
        boundary_guard=float(vcfg.get("boundary_guard_seconds", .5)),
        positive_overlap_fraction=float(vcfg.get("positive_overlap_fraction", .5)))
    if not train_rows or not val_rows:
        raise ValueError(f"Insufficient usable windows: train={len(train_rows)} validation={len(val_rows)}; rejected train={train_rejected}, validation={val_rejected}")
    device = torch.device("cuda" if torch.cuda.is_available() and mcfg.get("device", "auto") != "cpu" else "cpu")
    criterion, sampler, class_counts = _loss_and_sampling(train_rows, variant, device, mcfg.get("focal_gamma", 2.0))
    if set(_counts(val_rows)) != {0, 1}:
        raise ValueError(f"Validation must contain both classes for AP/checkpoint selection; found {dict(_counts(val_rows))}")
    model, model_info = build_r3d18(2, mcfg.get("weights_enum", "R3D_18_Weights.KINETICS400_V1"))
    model = model.to(device)
    transform_mean = model_info["normalization_mean"]
    transform_std = model_info["normalization_std"]
    train_ds = VideoWindowDataset(train_rows, split="train", config=vcfg, mean=transform_mean, std=transform_std, augment=True)
    val_ds = VideoWindowDataset(val_rows, split="validation", config=vcfg, mean=transform_mean, std=transform_std)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=int(mcfg["batch_size"]), sampler=sampler,
                              shuffle=sampler is None, num_workers=int(mcfg["workers"]),
                              pin_memory=device.type == "cuda", worker_init_fn=_worker_seed, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=int(mcfg["batch_size"]), shuffle=False,
                            num_workers=int(mcfg["workers"]), pin_memory=device.type == "cuda")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    package_versions = {name: metadata.version(name) for name in
                        ("torch", "torchvision", "opencv-python", "ultralytics", "scikit-learn", "tensorboard", "numpy")}
    (output / "run_config.json").write_text(json.dumps({"config": config, "variant": variant,
        "model_info": model_info, "device": str(device), "class_counts": class_counts,
        "train_windows": len(train_rows), "validation_windows": len(val_rows),
        "train_unique_incidents": len({r["incident_id"] for r in train_rows}),
        "validation_unique_incidents": len({r["incident_id"] for r in val_rows}),
        "preprocessing": {k: v for k, v in vcfg.items()},
        "package_versions": package_versions,
        "created_at": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")
    writer = SummaryWriter(log_dir=str(output / "tensorboard"))
    history_path = output / "history.jsonl"
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and bool(mcfg.get("amp", True)))
    best_ap, best_epoch = -1.0, None
    start_epoch, stage_start = 0, "head"
    resume_state = None
    if resume:
        resume_state = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(resume_state["model"])
        start_epoch = int(resume_state["epoch"]) + 1
        best_ap = float(resume_state.get("best_validation_average_precision", -1))
        stage_start = resume_state.get("stage", "head")

    def train_stage(stage, epochs, unfreeze_names, learning_rate, epoch_offset):
        nonlocal best_ap, best_epoch
        for parameter in model.parameters():
            parameter.requires_grad = False
        for name in unfreeze_names:
            module = getattr(model, name, None)
            if module is None:
                raise ValueError(f"Unknown R3D layer for unfreezing: {name}")
            for parameter in module.parameters():
                parameter.requires_grad = True
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
            lr=float(learning_rate), weight_decay=float(mcfg["weight_decay"]))
        if resume_state and stage == stage_start:
            optimizer.load_state_dict(resume_state["optimizer"])
            scaler_state = resume_state.get("scaler", {})
            if scaler_state:
                scaler.load_state_dict(scaler_state)
        for local_epoch in range(epochs):
            epoch = epoch_offset + local_epoch
            model.train()
            running, seen = 0.0, 0
            for batch in train_loader:
                video = batch["video"].to(device, non_blocking=True)
                target = batch["label"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda" and bool(mcfg.get("amp", True))):
                    logits = model(video)
                    loss = criterion(logits, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running += float(loss.item()) * len(target)
                seen += len(target)
            val_metrics, targets, probs, metadata = _evaluate(model, val_loader, device, criterion, bool(mcfg.get("amp", True)))
            val_ap = val_metrics["average_precision_pr_auc"]
            if val_ap is None:
                raise ValueError("Validation AP is undefined because validation contains no crash examples")
            train_loss = running / max(1, seen)
            row = {"epoch": epoch, "stage": stage, "train_loss": train_loss, "validation": val_metrics}
            with history_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/validation", val_metrics["loss"], epoch)
            writer.add_scalar("average_precision/validation", val_ap, epoch)
            _save_checkpoint(output / "last.pt", model=model, optimizer=optimizer, scaler=scaler,
                             epoch=epoch, stage=stage, best_metric=best_ap, config=config,
                             model_info=model_info, variant=variant)
            if val_ap > best_ap:
                best_ap, best_epoch = float(val_ap), epoch
                _save_checkpoint(output / "best.pt", model=model, optimizer=optimizer, scaler=scaler,
                                 epoch=epoch, stage=stage, best_metric=best_ap, config=config,
                                 model_info=model_info, variant=variant)
                with (output / "validation_predictions.jsonl").open("w", encoding="utf-8") as stream:
                    for meta, target, prob in zip(metadata, targets, probs):
                        stream.write(json.dumps({**meta, "target": target, "crash_probability": prob}) + "\n")
        return epoch_offset + epochs

    try:
        epoch = start_epoch
        head_epochs = int(mcfg["head_epochs"])
        if stage_start == "head" and head_epochs:
            epoch = train_stage("head", max(0, head_epochs - start_epoch), ["fc"], mcfg["head_lr"], start_epoch)
        ft_epochs = int(mcfg["finetune_epochs"])
        layers = mcfg.get("unfreeze_layers", ["layer4", "fc"])
        if ft_epochs:
            done = max(0, start_epoch - head_epochs) if resume_state and stage_start == "fine_tune" else 0
            epoch = train_stage("fine_tune", max(0, ft_epochs - done), layers, mcfg["backbone_lr"], epoch)
    finally:
        writer.close()
    summary = {"best_checkpoint": str((output / "best.pt").resolve()), "best_validation_average_precision": best_ap,
            "best_epoch": best_epoch, "device": str(device), "variant": variant,
            "train_windows": len(train_ds), "validation_windows": len(val_ds),
            "train_unique_incidents": len({r["incident_id"] for r in train_rows}),
            "validation_unique_incidents": len({r["incident_id"] for r in val_rows}),
            "package_versions": package_versions}
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_r3d_checkpoint(path, device="auto"):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model, model_info = build_r3d18(2, None)
    model.load_state_dict(state["model"])
    resolved = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else device if device != "auto" else "cpu")
    model.to(resolved).eval()
    return model, state, model_info, resolved

