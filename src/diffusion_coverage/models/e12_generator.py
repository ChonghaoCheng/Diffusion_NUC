from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn


PAD, BOS, END, VIA = 0, 1, 2, 9


def scan_symbol(family_index: int, direction: int) -> int:
    if family_index not in (0, 1, 2) or direction not in (-1, 1):
        raise ValueError("invalid SCAN symbol")
    return 3 + 2 * family_index + (direction < 0)


def unpack_scan_symbol(symbol: int) -> tuple[int, int]:
    if symbol < 3 or symbol > 8:
        raise ValueError("not a SCAN symbol")
    value = symbol - 3
    return value // 2, -1 if value % 2 else 1


@dataclass(frozen=True)
class GeneratorConfig:
    condition_dim: int = 21
    width: int = 128
    layers: int = 4
    heads: int = 4
    ffn: int = 512
    dropout: float = 0.0
    maximum_tokens: int = 64
    vocabulary_size: int = 10
    control_dim: int = 2


class _TransformerBody(nn.Module):
    def __init__(self, config: GeneratorConfig, *, continuous: bool):
        super().__init__()
        self.config = config
        self.token = nn.Embedding(config.vocabulary_size, config.width)
        self.position = nn.Embedding(config.maximum_tokens + 1, config.width)
        self.condition = nn.Linear(config.condition_dim, config.width)
        self.continuous = continuous
        if continuous:
            self.controls = nn.Linear(config.control_dim, config.width)
            self.time = nn.Sequential(nn.Linear(1, config.width), nn.SiLU(), nn.Linear(config.width, config.width))
        layer = nn.TransformerEncoderLayer(
            config.width, config.heads, config.ffn, config.dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, config.layers, norm=nn.LayerNorm(config.width))

    def forward(self, tokens: Tensor, condition: Tensor, controls: Tensor | None = None, time: Tensor | None = None, *, causal: bool = False) -> Tensor:
        batch, length = tokens.shape
        positions = torch.arange(length, device=tokens.device).unsqueeze(0)
        hidden = self.token(tokens) + self.position(positions) + self.condition(condition).unsqueeze(1)
        if self.continuous:
            if controls is None or time is None:
                raise ValueError("continuous body requires controls and time")
            if time.ndim == 1:
                time = time[:, None]
            hidden = hidden + self.controls(controls) + self.time(time).unsqueeze(1)
        mask = torch.triu(torch.ones(length, length, dtype=torch.bool, device=tokens.device), diagonal=1) if causal else None
        return self.encoder(hidden, mask=mask, src_key_padding_mask=tokens.eq(PAD))


class ProgramDecoder(nn.Module):
    def __init__(self, config: GeneratorConfig):
        super().__init__()
        self.config = config
        self.body = _TransformerBody(config, continuous=False)
        self.output = nn.Linear(config.width, config.vocabulary_size)

    def forward(self, input_tokens: Tensor, condition: Tensor) -> Tensor:
        return self.output(self.body(input_tokens, condition, causal=True))

    @torch.no_grad()
    def generate(self, condition: Tensor, *, generator: torch.Generator, temperature: float | None) -> Tensor:
        tokens = torch.full((condition.shape[0], 1), BOS, dtype=torch.long, device=condition.device)
        finished = torch.zeros(condition.shape[0], dtype=torch.bool, device=condition.device)
        for _ in range(self.config.maximum_tokens):
            logits = self(tokens, condition)[:, -1]
            # PAD/BOS are never valid generated program tokens.
            logits[:, :2] = -torch.inf
            if temperature is None:
                nxt = logits.argmax(-1)
            else:
                nxt = torch.multinomial(torch.softmax(logits / temperature, -1), 1, generator=generator).squeeze(1)
            nxt = torch.where(finished, torch.full_like(nxt, END), nxt)
            tokens = torch.cat((tokens, nxt[:, None]), 1)
            finished |= nxt.eq(END)
            if bool(finished.all()):
                break
        return tokens[:, 1:]


class ContinuousProgramHead(nn.Module):
    def __init__(self, config: GeneratorConfig):
        super().__init__()
        self.config = config
        self.body = _TransformerBody(config, continuous=True)
        self.output = nn.Linear(config.width, config.control_dim)

    def forward(self, tokens: Tensor, condition: Tensor, controls: Tensor, time: Tensor) -> Tensor:
        return self.output(self.body(tokens, condition, controls, time, causal=False))


@torch.no_grad()
def heun_generate(model: ContinuousProgramHead, tokens: Tensor, condition: Tensor, noise: Tensor, *, steps: int = 32) -> tuple[Tensor, int]:
    value = noise
    dt = 1.0 / steps
    for index in range(steps):
        t = torch.full((tokens.shape[0],), index / steps, device=tokens.device, dtype=value.dtype)
        first = model(tokens, condition, value, t)
        predicted = value + dt * first
        second = model(tokens, condition, predicted, torch.clamp(t + dt, max=1.0))
        value = value + 0.5 * dt * (first + second)
    return value, 2 * steps


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def sequence_key(symbols: Iterable[int]) -> str:
    sequence = []
    for symbol in symbols:
        value = int(symbol)
        if value in (PAD, BOS):
            continue
        sequence.append(value)
        if value == END:
            break
    return ",".join(map(str, sequence))
