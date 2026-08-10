# instruction-provenance-probe

Does a language model internally distinguish an instruction its operator gave
from a byte-identical instruction that arrived inside attacker-controlled data?

Prompt-injection benchmarks answer that behaviorally — they measure whether the
model *obeys* the injected instruction. This is the mechanistic version of the
same question: is instruction provenance linearly decodable from the residual
stream, and if so, is that representation causally involved in the model's
behavior?

**Result on GPT-2 small: no, on both counts.** Details and caveats below. The
headline finding of this repo is a null result plus a methodological one.

## The methodological finding

The first design produced 100% probe accuracy at every layer, including layer 0:

```
design: v1 naive (confounded)
mean tokens  trusted 25  untrusted 50  gap 25
length-only probe: 100.0%

layer     probe   shuffled   majority
    0    100.0%      64.2%      50.0%
    ...
   11    100.0%      47.1%      50.0%
```

Layer 0 has done essentially no computation. Anything perfectly decodable there
is a surface property of the input, not a learned concept. The cause: in that
design the untrusted prompts were structurally different from the trusted ones
and therefore twice as long, and GPT-2 adds positional embeddings — so the
last-token residual encoded sequence position, which correlated perfectly with
the label. **A probe fit on token count alone also scored 100%.** The probe was
reading prompt length.

The fix is a swap design. Each pair uses two instruction strings, I and D:

- `trusted=1` — operator slot holds I, document slot holds D
- `trusted=0` — operator slot holds D, document slot holds I

Both prompts contain I and D exactly once, so token content is identical across
conditions and length is matched within every pair. The readout token always
belongs to I, so the only difference is which slot I occupies.

```
design: v2 swap (length-matched)
mean tokens  trusted 58  untrusted 58  gap 0
length-only probe: 50.0%
```

The length-only probe runs on every execution. It is not a one-time check —
it is the control that licenses interpreting the layer sweep at all.

## The null result

With the confound removed, no layer carries a linearly decodable, generalizing
provenance signal:

```
layer     probe   shuffled   majority
    0     26.2%      47.5%      50.0%
    1     32.5%      52.5%      50.0%
    2     33.8%      50.0%      50.0%
    3     31.2%      46.2%      50.0%
    4     35.0%      50.0%      50.0%
    5     36.2%      57.5%      50.0%
    6     38.8%      42.5%      50.0%
    7     43.8%      55.0%      50.0%
    8     55.0%      53.8%      50.0%
    9     56.2%      55.0%      50.0%  <- best
   10     42.5%      55.0%      50.0%
   11     40.0%      53.8%      50.0%
```

The best layer reaches 56.2%, against a shuffled-label control of 55.0% at the
same layer. That gap is noise. Several layers land *below* chance, which on a
split held out by instruction means the probe fit instruction-specific structure
in training that inverts on unseen instructions — the signature of no
generalizable direction rather than a weak one.

The train/test split is by whole held-out instruction, not at random. A random
split would place near-duplicate prompts (same instruction, different carrier) on
both sides, letting a probe score well by memorizing wording.

## The intervention

Steering the layer-9 residual stream along the probe direction, on untrusted
prompts, measuring the log-probability of the token the injected instruction
demanded, against a norm-matched random direction:

```
     alpha   probe dir   random dir     delta
     -2.0n      -8.991       -9.095    +0.105
     -1.0n      -7.684       -7.726    +0.042
     -0.5n      -6.658       -6.787    +0.129
  0 (base)      -6.072       -6.072    +0.000
     +0.5n      -6.981       -7.653    +0.673
     +1.0n      -7.836       -9.206    +1.371
     +2.0n      -8.928      -10.450    +1.522
```

This is **not** evidence of a causal provenance direction, and it would be easy
to misread as one. Steering in *either* direction lowers the target token's
probability relative to the unsteered baseline (-6.072). The probe direction is
merely less destructive than a random vector of the same norm — which is the
expected result for any direction fit to the data, since it lies in a subspace
the model actually uses. A causal provenance direction would push compliance
*above* baseline in one direction. Nothing here does.

## What this does and does not license

Does not show that models lack a provenance representation. It shows that
**this** probe, at **this** readout position, in **this** model, does not find one:

- GPT-2 small (124M) is not instruction-tuned and follows instructions poorly.
  A model that barely acts on the trusted/untrusted distinction has weak reason
  to represent it.
- Only the final-token residual is read. Provenance may be represented at the
  instruction tokens themselves and never moved to the last position.
- Linear probes only. A nonlinear or multi-directional encoding would be missed.
- 320 training examples from templated prompts; narrow distribution.

The natural next step is an instruction-tuned model where the behavioral effect
demonstrably exists, and per-token rather than last-token readout.

## Layout

```
provenance/data.py       both designs; the swap construction and the split-by-instruction logic
provenance/probe.py      activation capture, per-layer linear probes, length/shuffled/majority baselines
provenance/intervene.py  forward-hook steering with a norm-matched random control
provenance/run.py        runs both designs, prints the tables above, writes results/results.json
```

## Running it

```
pip install torch transformers scikit-learn numpy
python -m provenance.run
```

CPU-only is fine; the full run is a few minutes on GPT-2 small. Steering
coefficients are expressed as fractions of each layer's mean residual norm,
since absolute coefficients are not comparable across layers (GPT-2's residual
norm grows roughly an order of magnitude from layer 0 to layer 11).
