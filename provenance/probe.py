"""Capture residual-stream activations and fit a linear probe per layer.

Reads the residual stream after every transformer block, at the final token
position, and asks whether a linear classifier can recover instruction
provenance from it. Reports accuracy per layer against two baselines, because
a probe accuracy with nothing to compare it to is not evidence:

  - majority-class: what you get by always guessing the larger class
  - shuffled-label: the same probe fit to randomized labels, which measures
    how much accuracy this many features and this little data buy you for
    free. With d_model=768 and a few hundred examples, that number is not
    guaranteed to be 50%, and pretending otherwise is how people report
    "the model represents X" when it does not.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from provenance.data import Example


@torch.no_grad()
def capture_activations(model, tokenizer, examples: list[Example]) -> np.ndarray:
    """Return activations of shape (n_examples, n_layers, d_model).

    Uses output_hidden_states rather than manual hooks for the per-layer sweep;
    intervene.py registers real forward hooks where it needs to *edit* the
    stream rather than read it.
    """
    per_example = []
    for example in examples:
        inputs = tokenizer(example.text, return_tensors="pt", truncation=True, max_length=1024)
        outputs = model(**inputs, output_hidden_states=True)
        # hidden_states[0] is the embedding output; blocks are 1..n_layer.
        # Last token position: the residual stream the model would decode from.
        layers = torch.stack([h[0, -1, :] for h in outputs.hidden_states[1:]])
        per_example.append(layers.float().numpy())
    return np.stack(per_example)


def fit_probe(
    train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray
) -> tuple[float, np.ndarray]:
    """Fit a logistic probe on one layer. Returns (test accuracy, direction).

    The direction is returned in the original activation space (scaler undone)
    so intervene.py can add it straight onto the residual stream.
    """
    scaler = StandardScaler().fit(train_x)
    probe = LogisticRegression(max_iter=2000, C=1.0)
    probe.fit(scaler.transform(train_x), train_y)
    accuracy = float(probe.score(scaler.transform(test_x), test_y))

    direction = probe.coef_[0] / scaler.scale_
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction = direction / norm
    return accuracy, direction


def length_baseline(
    tokenizer, train: list[Example], test: list[Example], train_y: np.ndarray, test_y: np.ndarray
) -> float:
    """Fit a probe on token count alone.

    This is the control that caught the v1 design: if prompt length by itself
    predicts the label, then any residual-stream probe may be reading position
    rather than the concept, and a high layer-0 accuracy means nothing. Near
    chance here is what licenses interpreting the layer sweep at all.
    """
    train_lengths = np.array([[len(tokenizer.encode(e.text))] for e in train], dtype=float)
    test_lengths = np.array([[len(tokenizer.encode(e.text))] for e in test], dtype=float)
    accuracy, _ = fit_probe(train_lengths, train_y, test_lengths, test_y)
    return accuracy


def length_stats(tokenizer, examples: list[Example]) -> dict[str, float]:
    trusted = [len(tokenizer.encode(e.text)) for e in examples if e.trusted == 1]
    untrusted = [len(tokenizer.encode(e.text)) for e in examples if e.trusted == 0]
    return {
        "mean_tokens_trusted": float(np.mean(trusted)),
        "mean_tokens_untrusted": float(np.mean(untrusted)),
        "mean_abs_gap": float(abs(np.mean(trusted) - np.mean(untrusted))),
    }


def majority_baseline(train_y: np.ndarray, test_y: np.ndarray) -> float:
    majority = 1 if train_y.mean() >= 0.5 else 0
    return float((test_y == majority).mean())


def sweep_layers(
    train_acts: np.ndarray,
    train_y: np.ndarray,
    test_acts: np.ndarray,
    test_y: np.ndarray,
    seed: int = 0,
) -> dict:
    """Fit a probe at every layer, plus a shuffled-label control at each."""
    rng = np.random.default_rng(seed)
    n_layers = train_acts.shape[1]

    accuracies, shuffled, directions = [], [], []
    for layer in range(n_layers):
        accuracy, direction = fit_probe(
            train_acts[:, layer, :], train_y, test_acts[:, layer, :], test_y
        )
        accuracies.append(accuracy)
        directions.append(direction)

        shuffled_y = rng.permutation(train_y)
        control_accuracy, _ = fit_probe(
            train_acts[:, layer, :], shuffled_y, test_acts[:, layer, :], test_y
        )
        shuffled.append(control_accuracy)

    best_layer = int(np.argmax(accuracies))
    return {
        "layer_accuracy": accuracies,
        "shuffled_label_accuracy": shuffled,
        "majority_baseline": majority_baseline(train_y, test_y),
        "best_layer": best_layer,
        "best_accuracy": accuracies[best_layer],
        "directions": directions,
    }
