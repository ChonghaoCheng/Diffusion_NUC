#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from diffusion_coverage.models import ConditionalFlowMatcher, PathVectorField, PathVectorFieldConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark six-DoF FM training throughput")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--surface-points", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    config = PathVectorFieldConfig(
        path_dim=6,
        condition_dim=4,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        num_heads=4,
    )
    model = PathVectorField(config).to(device)
    matcher = ConditionalFlowMatcher(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    target = torch.randn(args.batch_size, args.tokens, 6, device=device)
    surface = torch.randn(args.batch_size, args.surface_points, 6, device=device)
    surface[..., 3:] = torch.nn.functional.normalize(surface[..., 3:], dim=-1)
    condition = torch.randn(args.batch_size, 4, device=device)
    mask = torch.ones(args.batch_size, args.tokens, dtype=torch.bool, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    durations = []
    for step in range(args.warmup + args.steps):
        optimizer.zero_grad(set_to_none=True)
        start = perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = matcher.loss(target, surface, condition, path_mask=mask).total
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        if step >= args.warmup:
            durations.append(perf_counter() - start)
    elapsed = sum(durations)
    result = {
        "device": str(device),
        "batch_size": args.batch_size,
        "tokens": args.tokens,
        "surface_points": args.surface_points,
        "hidden_dim": args.hidden_dim,
        "layers": args.layers,
        "mean_step_seconds": elapsed / len(durations),
        "plans_per_second": args.batch_size * len(durations) / elapsed,
        "path_tokens_per_second": args.batch_size * args.tokens * len(durations) / elapsed,
        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "final_loss": float(loss.detach()),
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
