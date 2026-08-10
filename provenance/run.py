"""Run the experiment and write results.

    python -m provenance.run

Runs two designs and reports both, because the difference between them is the
finding:

  v1 "naive"   - trusted and untrusted prompts have different structure, so
                 they also have different lengths. Confounded.
  v2 "swap"    - both conditions contain the same two instruction strings with
                 only their slots exchanged. Length and token content matched.

Each design reports: a length-only probe (is the confound live?), a per-layer
residual-stream probe, shuffled-label and majority-class baselines, and - for
the controlled design - a causal steering intervention against a norm-matched
random direction.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from provenance.data import (
    build_dataset,
    build_swapped_dataset,
    held_out_split,
)
from provenance.intervene import sweep_alphas
from provenance.probe import (
    capture_activations,
    length_baseline,
    length_stats,
    sweep_layers,
)

MODEL_NAME = "gpt2"
RESULTS_DIR = Path(__file__).parent.parent / "results"
# Steering coefficients as a fraction of the layer's mean residual norm;
# absolute coefficients aren't comparable across layers.
ALPHA_FRACTIONS = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]


def run_design(model, tokenizer, name: str, examples: list) -> tuple[dict, list, np.ndarray]:
    train, test = held_out_split(examples)
    train_y = np.array([e.trusted for e in train])
    test_y = np.array([e.trusted for e in test])

    lengths = length_stats(tokenizer, examples)
    length_acc = length_baseline(tokenizer, train, test, train_y, test_y)

    print(f"\n{'=' * 60}\ndesign: {name}   ({len(train)} train / {len(test)} test)")
    print(
        f"mean tokens  trusted {lengths['mean_tokens_trusted']:.0f}  "
        f"untrusted {lengths['mean_tokens_untrusted']:.0f}  "
        f"gap {lengths['mean_abs_gap']:.0f}"
    )
    print(f"length-only probe: {length_acc:.1%}   <- confound check (want ~chance)")

    train_acts = capture_activations(model, tokenizer, train)
    test_acts = capture_activations(model, tokenizer, test)
    sweep = sweep_layers(train_acts, train_y, test_acts, test_y)

    print(f"\n{'layer':>5}  {'probe':>8}  {'shuffled':>9}  {'majority':>9}")
    print("-" * 36)
    for layer, (accuracy, shuffled) in enumerate(
        zip(sweep["layer_accuracy"], sweep["shuffled_label_accuracy"])
    ):
        marker = "  <- best" if layer == sweep["best_layer"] else ""
        print(
            f"{layer:>5}  {accuracy:>8.1%}  {shuffled:>9.1%}  "
            f"{sweep['majority_baseline']:>9.1%}{marker}"
        )

    record = {k: v for k, v in sweep.items() if k != "directions"}
    record["length_only_probe"] = length_acc
    record["length_stats"] = lengths
    record["layer0_accuracy"] = sweep["layer_accuracy"][0]
    return record, test, train_acts


def main() -> int:
    torch.manual_seed(0)
    np.random.seed(0)

    print(f"loading {MODEL_NAME} ...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model.eval()

    naive_record, _, _ = run_design(model, tokenizer, "v1 naive (confounded)", build_dataset())

    swap_examples = build_swapped_dataset()
    swap_record, swap_test, swap_train_acts = run_design(
        model, tokenizer, "v2 swap (length-matched)", swap_examples
    )

    # Re-fit to recover the direction for the controlled design's best layer.
    swap_train, _ = held_out_split(swap_examples)
    swap_train_y = np.array([e.trusted for e in swap_train])
    swap_test_y = np.array([e.trusted for e in swap_test])
    swap_test_acts = capture_activations(model, tokenizer, swap_test)
    full_sweep = sweep_layers(swap_train_acts, swap_train_y, swap_test_acts, swap_test_y)
    best_layer = full_sweep["best_layer"]
    direction = full_sweep["directions"][best_layer]

    mean_norm = float(np.linalg.norm(swap_train_acts[:, best_layer, :], axis=1).mean())
    alphas = [round(f * mean_norm, 4) for f in ALPHA_FRACTIONS]
    untrusted_test = [e for e in swap_test if e.trusted == 0]

    print(f"\n{'=' * 60}")
    print(f"intervention at layer {best_layer} (mean residual norm {mean_norm:.1f})")
    intervention = sweep_alphas(
        model, tokenizer, untrusted_test, best_layer, direction, alphas
    )
    intervention["alpha_fractions"] = ALPHA_FRACTIONS
    intervention["mean_residual_norm"] = mean_norm
    intervention["n_prompts"] = len(untrusted_test)
    intervention["layer"] = best_layer

    print(f"\nmean log-prob of injected target token, {len(untrusted_test)} untrusted prompts")
    print(f"{'alpha':>10}  {'probe dir':>10}  {'random dir':>11}  {'delta':>8}")
    print("-" * 46)
    for fraction, probe_score, random_score in zip(
        ALPHA_FRACTIONS, intervention["probe_direction"], intervention["random_direction"]
    ):
        label = f"{fraction:+.1f}n" if fraction else "0 (base)"
        print(
            f"{label:>10}  {probe_score:>10.3f}  {random_score:>11.3f}  "
            f"{probe_score - random_score:>+8.3f}"
        )

    RESULTS_DIR.mkdir(exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "designs": {"v1_naive": naive_record, "v2_swap": swap_record},
        "intervention": intervention,
    }
    out = RESULTS_DIR / "results.json"
    out.write_text(json.dumps(payload, indent=2))
    np.save(RESULTS_DIR / "best_direction.npy", direction)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
