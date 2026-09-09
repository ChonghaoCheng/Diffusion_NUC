#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from diffusion_coverage.learning import (
    TeacherPathDataset,
    collate_teacher_paths,
    create_instance_split,
    filter_manifest_by_candidate_name,
    load_manifest,
)
from diffusion_coverage.models import ConditionalFlowMatcher, PathVectorField, PathVectorFieldConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train variable-token k=1 conditional surface Flow Matching")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "flow_matching_m2")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument(
        "--path-self-attention",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--surface-points", type=int, default=512)
    parser.add_argument("--path-waypoints", type=int, default=256)
    parser.add_argument("--path-representation", choices=("variable", "fixed"), default="variable")
    parser.add_argument(
        "--path-coordinate-system",
        choices=(
            "xyz", "analytic_uv", "analytic_uv_control", "analytic_uv_structured",
            "analytic_uv_structured_residual",
        ),
        default="xyz",
    )
    parser.add_argument("--tokens-per-footprint-area", type=float, default=1.0)
    parser.add_argument("--minimum-path-tokens", type=int, default=32)
    parser.add_argument("--maximum-path-tokens", type=int, default=2048)
    parser.add_argument("--candidate-policy", choices=("all", "best"), default="all")
    parser.add_argument(
        "--candidate-name",
        default=None,
        help="Use only one exact canonical proposal name, excluding sanitizer suffixes",
    )
    parser.add_argument(
        "--allow-repaired-candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--include-mode-conditioning", action="store_true")
    parser.add_argument("--balance-pattern-modes", action="store_true")
    parser.add_argument(
        "--preserve-source-waypoints",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--smoothness-weight", type=float, default=0.0)
    parser.add_argument("--coverage-weight", type=float, default=0.0)
    parser.add_argument("--excess-length-weight", type=float, default=0.0)
    parser.add_argument("--tangent-weight", type=float, default=0.0)
    parser.add_argument("--surface-consistency-weight", type=float, default=0.0)
    parser.add_argument("--coverage-max-path-points", type=int, default=256)
    parser.add_argument("--noise-smoothing-sigma", type=float, default=None)
    parser.add_argument("--noise-smoothing-fraction", type=float, default=0.03)
    parser.add_argument(
        "--coupling",
        choices=("independent", "minibatch_ot"),
        default="independent",
    )
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--validation-interval-epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--split-seed", type=int, default=None,
        help="Instance split seed; defaults to --seed",
    )
    parser.add_argument(
        "--data-seed", type=int, default=None,
        help="Surface subsampling seed; defaults to --seed",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--cuda-device-index",
        type=int,
        default=None,
        help="Bind this process to one visible CUDA device without shell environment variables",
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--matmul-precision",
        choices=("highest", "high", "medium"),
        default="high",
    )
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        default=None,
        help="Warm-start model weights while creating a fresh optimizer",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validation_interval_epochs < 1:
        raise ValueError("validation_interval_epochs must be positive")
    if args.prefetch_factor < 1:
        raise ValueError("prefetch_factor must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.cuda_device_index is not None and args.cuda_device_index < 0:
        raise ValueError("cuda_device_index must be nonnegative")
    if args.cuda_device_index is not None and args.device == "cpu":
        raise ValueError("cuda_device_index cannot be used with --device cpu")
    cuda_name = (
        "cuda"
        if args.cuda_device_index is None
        else f"cuda:{args.cuda_device_index}"
    )
    device = (
        torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(cuda_name if args.device == "cuda" else args.device)
    )
    torch.set_float32_matmul_precision(args.matmul_precision)
    if args.path_coordinate_system != "xyz" and (
        args.coverage_weight > 0.0 or args.surface_consistency_weight > 0.0
    ):
        raise ValueError("XYZ coverage and surface-consistency surrogates do not apply to analytic_uv paths")
    args.output.mkdir(parents=True, exist_ok=True)

    rows = filter_manifest_by_candidate_name(
        args.dataset,
        load_manifest(args.dataset),
        args.candidate_name,
        allow_repaired=args.allow_repaired_candidates,
    )
    split_seed = args.seed if args.split_seed is None else args.split_seed
    data_seed = args.seed if args.data_seed is None else args.data_seed
    train_ids, validation_ids = create_instance_split(
        rows, validation_fraction=args.validation_fraction, seed=split_seed
    )
    dataset_kwargs = {
        "num_surface_points": args.surface_points,
        "num_path_waypoints": args.path_waypoints if args.path_representation == "fixed" else None,
        "tokens_per_footprint_area": args.tokens_per_footprint_area,
        "minimum_path_tokens": args.minimum_path_tokens,
        "maximum_path_tokens": args.maximum_path_tokens,
        "candidate_policy": args.candidate_policy,
        "candidate_name": args.candidate_name,
        "allow_repaired_candidates": args.allow_repaired_candidates,
        "path_coordinate_system": args.path_coordinate_system,
        "include_mode_conditioning": args.include_mode_conditioning,
        "preserve_source_waypoints": args.preserve_source_waypoints,
        "seed": data_seed,
    }
    train_dataset = TeacherPathDataset(
        args.dataset,
        instance_ids=train_ids,
        **dataset_kwargs,
    )
    validation_dataset = None
    if validation_ids:
        validation_dataset = TeacherPathDataset(
            args.dataset,
            instance_ids=validation_ids,
            **dataset_kwargs,
        )
    generator = torch.Generator().manual_seed(args.seed)
    train_sampler = None
    if args.balance_pattern_modes:
        if not args.include_mode_conditioning:
            raise ValueError("mode balancing requires --include-mode-conditioning")
        mode_ids = [int(train_dataset[index]["mode_id"]) for index in range(len(train_dataset))]
        mode_counts = np.bincount(mode_ids)
        sample_weights = torch.tensor(
            [1.0 / mode_counts[mode_id] for mode_id in mode_ids], dtype=torch.double
        )
        train_sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        )
    worker_kwargs = {}
    if args.num_workers > 0:
        worker_kwargs = {
            "persistent_workers": args.persistent_workers,
            "prefetch_factor": args.prefetch_factor,
        }
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
        collate_fn=collate_teacher_paths,
        **worker_kwargs,
    )
    validation_loader = None if validation_dataset is None else DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_teacher_paths,
        **worker_kwargs,
    )

    model_config = PathVectorFieldConfig(
        path_dim=2 if args.path_coordinate_system != "xyz" else 3,
        condition_dim=7 if args.include_mode_conditioning else 2,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        path_self_attention=args.path_self_attention,
    )
    vector_field = PathVectorField(model_config).to(device)
    if args.initial_checkpoint is not None:
        initial = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=True)
        initial_config = PathVectorFieldConfig(**initial["model_config"])
        if initial_config != model_config:
            raise ValueError("initial checkpoint model configuration does not match CLI settings")
        vector_field.load_state_dict(initial["model_state"])
    noise_smoothing_sigma = 0.0 if args.noise_smoothing_sigma is None else args.noise_smoothing_sigma
    noise_smoothing_fraction = (
        args.noise_smoothing_fraction if args.noise_smoothing_sigma is None else 0.0
    )
    training_vector_field = (
        torch.compile(vector_field, dynamic=True) if args.compile else vector_field
    )
    matcher = ConditionalFlowMatcher(
        training_vector_field,
        smoothness_weight=args.smoothness_weight,
        coverage_weight=args.coverage_weight,
        excess_length_weight=args.excess_length_weight,
        tangent_weight=args.tangent_weight,
        surface_consistency_weight=args.surface_consistency_weight,
        coverage_max_path_points=args.coverage_max_path_points,
        noise_smoothing_sigma=noise_smoothing_sigma,
        noise_smoothing_fraction=noise_smoothing_fraction,
        coupling=args.coupling,
    )
    optimizer = torch.optim.AdamW(
        vector_field.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    amp_enabled = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    history: list[dict[str, float | int]] = []
    best_validation = float("inf")
    global_step = 0
    training_start = perf_counter()

    for epoch in range(1, args.epochs + 1):
        vector_field.train()
        train_values: list[float] = []
        train_velocity_values: list[float] = []
        train_coverage_values: list[float] = []
        train_excess_length_values: list[float] = []
        train_tangent_values: list[float] = []
        train_surface_consistency_values: list[float] = []
        train_token_counts: list[int] = []
        for batch in train_loader:
            path = batch["path"].to(device, non_blocking=True)
            surface = batch["surface"].to(device, non_blocking=True)
            condition = batch["condition"].to(device, non_blocking=True)
            path_mask = batch["path_mask"].to(device, non_blocking=True)
            path_arclength = batch["path_arclength"].to(device, non_blocking=True)
            train_token_counts.extend(int(value) for value in path_mask.sum(dim=1).cpu())
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                losses = matcher.loss(
                    path, surface, condition,
                    path_mask=path_mask, path_arclength=path_arclength,
                )
            scaler.scale(losses.total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(vector_field.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_values.append(float(losses.total.detach()))
            train_velocity_values.append(float(losses.velocity.detach()))
            train_coverage_values.append(float(losses.coverage.detach()))
            train_excess_length_values.append(float(losses.excess_length.detach()))
            train_tangent_values.append(float(losses.tangent.detach()))
            train_surface_consistency_values.append(
                float(losses.surface_consistency.detach())
            )
            global_step += 1
            if args.max_train_steps is not None and global_step >= args.max_train_steps:
                break

        reached_step_limit = (
            args.max_train_steps is not None and global_step >= args.max_train_steps
        )
        should_validate = (
            epoch % args.validation_interval_epochs == 0
            or reached_step_limit
            or epoch == args.epochs
        )
        if not should_validate:
            continue
        validation = evaluate_losses(matcher, validation_loader, device, amp_enabled)
        validation_loss = validation["total"]
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "batch_size": args.batch_size,
            "compile": args.compile,
            "matmul_precision": args.matmul_precision,
            "cuda_device_index": args.cuda_device_index,
            "initial_checkpoint": None if args.initial_checkpoint is None else str(args.initial_checkpoint),
            "train_loss": float(np.mean(train_values)),
            "validation_loss": validation_loss,
            "train_velocity_loss": float(np.mean(train_velocity_values)),
            "train_coverage_loss": float(np.mean(train_coverage_values)),
            "train_excess_length_loss": float(np.mean(train_excess_length_values)),
            "train_tangent_loss": float(np.mean(train_tangent_values)),
            "train_surface_consistency_loss": float(
                np.mean(train_surface_consistency_values)
            ),
            "validation_velocity_loss": validation["velocity"],
            "validation_coverage_loss": validation["coverage"],
            "validation_excess_length_loss": validation["excess_length"],
            "validation_tangent_loss": validation["tangent"],
            "validation_surface_consistency_loss": validation["surface_consistency"],
            "elapsed_seconds": perf_counter() - training_start,
            "minimum_path_tokens": min(train_token_counts),
            "maximum_path_tokens": max(train_token_counts),
            "mean_path_tokens": float(np.mean(train_token_counts)),
        }
        history.append(record)
        print(
            f"epoch={epoch:04d} step={global_step:06d} "
            f"train={record['train_loss']:.6f} val={validation_loss:.6f}"
        )
        checkpoint = {
            "model_state": vector_field.state_dict(),
            "model_config": model_config.to_dict(),
            "num_surface_points": args.surface_points,
            "path_representation": args.path_representation,
            "path_coordinate_system": args.path_coordinate_system,
            "include_mode_conditioning": args.include_mode_conditioning,
            "balance_pattern_modes": args.balance_pattern_modes,
            "num_path_waypoints": args.path_waypoints if args.path_representation == "fixed" else None,
            "tokens_per_footprint_area": args.tokens_per_footprint_area,
            "minimum_path_tokens": args.minimum_path_tokens,
            "maximum_path_tokens": args.maximum_path_tokens,
            "candidate_policy": args.candidate_policy,
            "candidate_name": args.candidate_name,
            "allow_repaired_candidates": args.allow_repaired_candidates,
            "smoothness_weight": args.smoothness_weight,
            "coverage_weight": args.coverage_weight,
            "excess_length_weight": args.excess_length_weight,
            "tangent_weight": args.tangent_weight,
            "surface_consistency_weight": args.surface_consistency_weight,
            "coverage_max_path_points": args.coverage_max_path_points,
            "preserve_source_waypoints": args.preserve_source_waypoints,
            "train_instance_ids": train_ids,
            "validation_instance_ids": validation_ids,
            "seed": args.seed,
            "split_seed": split_seed,
            "data_seed": data_seed,
            "cuda_device_index": args.cuda_device_index,
            "noise_smoothing_sigma": noise_smoothing_sigma,
            "noise_smoothing_fraction": noise_smoothing_fraction,
            "coupling": args.coupling,
            "epoch": epoch,
            "global_step": global_step,
            "history": history,
        }
        torch.save(checkpoint, args.output / "last.pt")
        score = validation_loss if validation_loader is not None else record["train_loss"]
        if score < best_validation:
            best_validation = score
            torch.save(checkpoint, args.output / "best.pt")
        (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        if reached_step_limit:
            break
    print(f"checkpoint: {args.output / 'best.pt'}")


@torch.no_grad()
def evaluate_losses(
    matcher: ConditionalFlowMatcher,
    loader: DataLoader | None,
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, float]:
    if loader is None:
        return {
            key: float("nan")
            for key in (
                "total", "velocity", "coverage", "excess_length", "tangent",
                "surface_consistency",
            )
        }
    matcher.vector_field.eval()
    values = {
        key: []
        for key in (
            "total", "velocity", "coverage", "excess_length", "tangent",
            "surface_consistency",
        )
    }
    for batch in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            losses = matcher.loss(
                batch["path"].to(device),
                batch["surface"].to(device),
                batch["condition"].to(device),
                path_mask=batch["path_mask"].to(device),
                path_arclength=batch["path_arclength"].to(device),
            )
        values["total"].append(float(losses.total))
        values["velocity"].append(float(losses.velocity))
        values["coverage"].append(float(losses.coverage))
        values["excess_length"].append(float(losses.excess_length))
        values["tangent"].append(float(losses.tangent))
        values["surface_consistency"].append(float(losses.surface_consistency))
    return {key: float(np.mean(component)) for key, component in values.items()}


if __name__ == "__main__":
    main()
