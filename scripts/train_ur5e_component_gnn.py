#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from diffusion_coverage.models import IKComponentMPNN


FRONTIER_PATH = ROOT / "scripts" / "analyze_ur5e_component_frontier.py"
SPEC = importlib.util.spec_from_file_location("component_frontier", FRONTIER_PATH)
FRONTIER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(FRONTIER)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a deterministic UR5e IK-component GNN")
    parser.add_argument("--graphs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-segments", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--initial-checkpoint", type=Path)
    return parser.parse_args()


class ComponentDataset(Dataset):
    def __init__(self, paths: list[Path], budgets: range) -> None:
        self.examples = []
        for path in paths:
            raw = load_graph(path)
            coverage, reachable = FRONTIER.component_coverage(raw["component_labels"])
            for budget in budgets:
                selected, covered = FRONTIER.optimal_component_selection(coverage, reachable, budget)
                optimal = int(covered.sum())
                self.examples.append((raw, budget, selected.astype(np.float32), optimal))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        return self.examples[index]


def load_graph(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    mask = values["candidate_mask"]
    node, candidate = np.nonzero(mask)
    local_to_flat = np.full(mask.shape, -1, dtype=np.int64)
    local_to_flat[node, candidate] = np.arange(len(node))
    metadata = json.loads(str(values["metadata_json"]))
    tau = float(metadata["axis_tolerance_degrees"]) / 10.0
    features = np.column_stack(
        (
            values["uv"][node],
            values["positions"][node],
            values["axes"][node],
            values["q_candidates"][node, candidate],
            values["manipulability"][node, candidate, None],
            values["joint_limit_margin"][node, candidate, None],
            np.full((len(node), 1), tau),
        )
    ).astype(np.float32)
    edges = []
    for edge_id, (source, target) in enumerate(values["edge_index"].T):
        for source_candidate, target_candidate in np.argwhere(values["edge_compatibility"][edge_id]):
            source_flat = local_to_flat[int(source), source_candidate]
            target_flat = local_to_flat[int(target), target_candidate]
            if source_flat >= 0 and target_flat >= 0:
                edges.append((source_flat, target_flat))
    return {
        "path": str(path),
        "features": features,
        "edge_index": np.asarray(edges, dtype=np.int64).reshape(-1, 2).T,
        "component_index": values["component_labels"][node, candidate].astype(np.int64),
        "component_labels": values["component_labels"],
    }


def collate(examples):
    features, edges, component_indices = [], [], []
    candidate_graph, component_graph, budgets, targets = [], [], [], []
    candidate_offset = component_offset = 0
    for graph_id, (raw, budget, target, _) in enumerate(examples):
        num_candidates = len(raw["features"])
        num_components = len(target)
        features.append(raw["features"])
        edges.append(raw["edge_index"] + candidate_offset)
        component_indices.append(raw["component_index"] + component_offset)
        candidate_graph.append(np.full(num_candidates, graph_id))
        component_graph.append(np.full(num_components, graph_id))
        budgets.append(np.full(num_components, budget / 8.0, dtype=np.float32))
        targets.append(target)
        candidate_offset += num_candidates
        component_offset += num_components
    return {
        "features": torch.from_numpy(np.concatenate(features)),
        "edge_index": torch.from_numpy(np.concatenate(edges, axis=1)),
        "component_index": torch.from_numpy(np.concatenate(component_indices)),
        "candidate_graph_index": torch.from_numpy(np.concatenate(candidate_graph)),
        "component_graph_index": torch.from_numpy(np.concatenate(component_graph)),
        "component_budget": torch.from_numpy(np.concatenate(budgets)),
        "target": torch.from_numpy(np.concatenate(targets)),
        "num_graphs": len(examples),
        "num_components": component_offset,
        "component_counts": [len(target) for _, _, target, _ in examples],
        "examples": examples,
    }


def forward(model, batch, device):
    tensor_keys = (
        "features", "edge_index", "component_index", "candidate_graph_index",
        "component_graph_index", "component_budget", "target",
    )
    moved = {key: batch[key].to(device, non_blocking=True) for key in tensor_keys}
    logits = model(
        moved["features"], moved["edge_index"], moved["component_index"],
        batch["num_components"], moved["candidate_graph_index"],
        moved["component_graph_index"], batch["num_graphs"], moved["component_budget"],
    )
    return logits, moved["target"]


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ratios = []
    for batch in loader:
        logits, _ = forward(model, batch, device)
        offset = 0
        for count, (raw, budget, _, optimal) in zip(
            batch["component_counts"], batch["examples"]
        ):
            scores = logits[offset : offset + count]
            selected = torch.topk(scores, k=min(budget, count)).indices.cpu().numpy()
            coverage, reachable = FRONTIER.component_coverage(raw["component_labels"])
            predicted = np.any(coverage[:, selected] > 0.5, axis=1).sum()
            ratios.append(predicted / max(optimal, 1))
            offset += count
    return float(np.mean(ratios))


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    paths = sorted(path for root in args.graphs for path in (root / "instances").glob("*.npz"))
    instance_ids = sorted({path.stem for path in paths})
    random.Random(args.seed).shuffle(instance_ids)
    split = max(1, round(len(instance_ids) * (1.0 - args.validation_fraction)))
    train_ids = set(instance_ids[:split])
    train_paths = [path for path in paths if path.stem in train_ids]
    validation_paths = [path for path in paths if path.stem not in train_ids]
    budgets = range(1, args.max_segments + 1)
    train = ComponentDataset(train_paths, budgets)
    validation = ComponentDataset(validation_paths, budgets)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    validation_loader = DataLoader(validation, batch_size=args.batch_size, collate_fn=collate)
    device = torch.device(args.device)
    model = IKComponentMPNN(hidden_dim=args.hidden_dim, layers=args.layers).to(device)
    if args.initial_checkpoint is not None:
        checkpoint = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    history = []
    best = -1.0
    args.output.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                logits, target = forward(model, batch, device)
                positives = target.sum().clamp_min(1.0)
                pos_weight = (target.numel() - positives) / positives
                loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        validation_ratio = evaluate(model, validation_loader, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_coverage_ratio": validation_ratio}
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation_ratio > best:
            best = validation_ratio
            torch.save({"model_state": model.state_dict(), "args": vars(args), "epoch": epoch}, args.output / "best.pt")
    (args.output / "history.json").write_text(json.dumps(history, indent=2, default=str) + "\n")
    (args.output / "summary.json").write_text(json.dumps({"best_validation_coverage_ratio": best, "train_instances": len(train_ids), "validation_instances": len(set(instance_ids)-train_ids)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
