#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import numpy as np
import torch


TRAIN_PATH = ROOT / "scripts" / "train_ur5e_component_gnn.py"
SPEC = importlib.util.spec_from_file_location("component_gnn_training", TRAIN_PATH)
TRAIN = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(TRAIN)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate learned and heuristic IK-component priors")
    parser.add_argument("--graphs", nargs="+", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def greedy_marginal(coverage: np.ndarray, reachable: np.ndarray, budget: int) -> np.ndarray:
    selected = []
    covered = np.zeros(len(reachable), dtype=bool)
    for _ in range(min(budget, coverage.shape[1])):
        gains = np.sum((coverage > 0.5) & ~covered[:, None], axis=0)
        if selected:
            gains[np.asarray(selected)] = -1
        choice = int(np.argmax(gains))
        selected.append(choice)
        covered |= coverage[:, choice] > 0.5
    return np.asarray(selected, dtype=np.int64)


def covered_nodes(coverage: np.ndarray, selected: np.ndarray) -> int:
    if not len(selected):
        return 0
    return int(np.any(coverage[:, selected] > 0.5, axis=1).sum())


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved = checkpoint["args"]
    paths = sorted(path for root in args.graphs for path in (root / "instances").glob("*.npz"))
    instance_ids = sorted({path.stem for path in paths})
    import random

    random.Random(int(saved["seed"])).shuffle(instance_ids)
    split = max(1, round(len(instance_ids) * (1.0 - float(saved["validation_fraction"]))))
    validation_ids = set(instance_ids[split:])
    validation_paths = [path for path in paths if path.stem in validation_ids]
    dataset = TRAIN.ComponentDataset(validation_paths, range(1, int(saved["max_segments"]) + 1))
    device = torch.device(args.device)
    model = TRAIN.IKComponentMPNN(
        hidden_dim=int(saved["hidden_dim"]), layers=int(saved["layers"])
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    rows = []
    with torch.no_grad():
        for raw, budget, target, optimal in dataset:
            batch = TRAIN.collate([(raw, budget, target, optimal)])
            logits, _ = TRAIN.forward(model, batch, device)
            coverage, reachable = TRAIN.FRONTIER.component_coverage(raw["component_labels"])
            count = coverage.shape[1]
            learned = torch.topk(logits, k=min(budget, count)).indices.cpu().numpy()
            size_order = np.argsort(-coverage.sum(axis=0), kind="stable")[:budget]
            marginal = greedy_marginal(coverage, reachable, budget)
            rows.append(
                {
                    "instance_id": Path(raw["path"]).stem,
                    "condition": Path(raw["path"]).parents[1].name,
                    "budget": budget,
                    "optimal_covered": optimal,
                    "gnn_ratio": covered_nodes(coverage, learned) / max(optimal, 1),
                    "size_ratio": covered_nodes(coverage, size_order) / max(optimal, 1),
                    "marginal_ratio": covered_nodes(coverage, marginal) / max(optimal, 1),
                }
            )
    summary = {}
    for method in ("gnn", "size", "marginal"):
        values = np.asarray([row[f"{method}_ratio"] for row in rows])
        summary[method] = {
            "mean_coverage_ratio": float(values.mean()),
            "median_coverage_ratio": float(np.median(values)),
            "optimal_hit_rate": float(np.mean(values >= 1.0 - 1e-9)),
            "by_budget": {
                str(budget): float(
                    np.mean([row[f"{method}_ratio"] for row in rows if row["budget"] == budget])
                )
                for budget in sorted({row["budget"] for row in rows})
            },
        }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
