"""Causal test: does editing along the probe direction change behavior?

A probe finding a direction only shows the information is *present* and
linearly decodable. It does not show the model uses it. To claim that, you
have to intervene on the direction and watch behavior move.

Readout: on untrusted-condition prompts (instruction buried in data), the
log-probability the model assigns to the first token of the word the injected
instruction demanded. If the direction is causally involved in provenance
handling, steering toward the trusted pole should raise compliance.

Control: a random unit direction, matched in norm, steered at the same
coefficients. Adding any large vector to a residual stream perturbs the
output distribution, so a compliance shift only means something if it beats
a random perturbation of equal size.
"""
from __future__ import annotations

import numpy as np
import torch

from provenance.data import Example


def first_token_id(tokenizer, target_token: str) -> int:
    """Multi-token targets are read out on their first token."""
    return tokenizer.encode(target_token)[0]


def get_block(model, layer: int):
    return model.transformer.h[layer]


@torch.no_grad()
def target_logprob(
    model, tokenizer, example: Example, layer: int, direction: np.ndarray | None, alpha: float
) -> float:
    """Log-prob of the injected instruction's target token, optionally under
    a steered forward pass."""
    inputs = tokenizer(example.text, return_tensors="pt", truncation=True, max_length=1024)

    handle = None
    if direction is not None and alpha != 0.0:
        steer = torch.tensor(direction, dtype=torch.float32)

        def hook(_module, _args, output):
            # GPT-2 blocks return a tuple; the residual stream is element 0.
            hidden = output[0] if isinstance(output, tuple) else output
            hidden = hidden + alpha * steer
            return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

        handle = get_block(model, layer).register_forward_hook(hook)

    try:
        logits = model(**inputs).logits[0, -1, :]
    finally:
        if handle is not None:
            handle.remove()

    logprobs = torch.log_softmax(logits.float(), dim=-1)
    return float(logprobs[first_token_id(tokenizer, example.target_token)])


def sweep_alphas(
    model,
    tokenizer,
    examples: list[Example],
    layer: int,
    direction: np.ndarray,
    alphas: list[float],
    seed: int = 0,
) -> dict:
    """Steer along the probe direction and a matched random direction.

    Returns mean target log-prob per alpha for both, so the probe direction's
    effect can be read against what an arbitrary perturbation of the same size
    already does.
    """
    rng = np.random.default_rng(seed)
    random_direction = rng.normal(size=direction.shape)
    random_direction /= np.linalg.norm(random_direction)

    results = {"alphas": alphas, "probe_direction": [], "random_direction": []}
    for alpha in alphas:
        probe_scores = [
            target_logprob(model, tokenizer, e, layer, direction, alpha) for e in examples
        ]
        random_scores = [
            target_logprob(model, tokenizer, e, layer, random_direction, alpha) for e in examples
        ]
        results["probe_direction"].append(float(np.mean(probe_scores)))
        results["random_direction"].append(float(np.mean(random_scores)))

    baseline_index = alphas.index(0.0) if 0.0 in alphas else None
    results["baseline"] = (
        results["probe_direction"][baseline_index] if baseline_index is not None else None
    )
    return results
