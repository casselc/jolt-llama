# Exactness: what jolt-llama guarantees, and what it cannot

This document exists because a jolt-hegel property test failed, the failure was
real, and the first explanation for it was wrong. Both halves are recorded.

## Summary

| Claim | Status |
| --- | --- |
| Evaluating the same token vector the same way is deterministic | **Guaranteed** (measured 0.0, always) |
| Save/restore is transparent: restoring is indistinguishable from never clearing | **Guaranteed** (measured bit-exact) |
| Restore + append of >= 64 tokens reproduces a one-pass prefill | **Guaranteed for this model**, calibrated not assumed |
| Restore + append of < 64 tokens reproduces a one-pass prefill | **False**, and it is not save/restore's fault |
| Top-1 and top-5 ranking are stable across all evaluation paths | **Observed**, head agrees to 5e-4 nats |
| Scores from different batch-size regimes are interchangeable | **No.** See "Scoring convention" |

## How this was found

`check-state-roundtrip!` in `test/jolt/hegel_properties.clj` asserted that

    eval base -> save -> clear -> restore -> append 1 token

matches a full recompute to 1e-6. It failed and shrank to a minimal case.

The first hypothesis was that save/restore was lossy. It is not. The control in
`test/jolt/probe_suffix_len.clj` settles it:

    recompute vs recompute                      0.0000000000
    restore + 1-token append vs one-pass        0.1964101791
    split prefill vs one-pass, NO restore       0.1964101791   <-- identical

The same number with and without a restore. Save/restore contributes nothing.
What diverges is evaluating a prompt as two `llama_decode` calls instead of one.

## The actual rule

`test/jolt/probe_split_threshold.clj` and `probe_threshold_exact.clj` sweep it:

* split POINT is irrelevant (64 through 639, all exact)
* number of decode calls is irrelevant (a three-way split is exact)
* only the SIZE of each call matters, with a sharp boundary at 64:

      suffix_len= 63   0.2437810898  diverges
      suffix_len= 64   0.0000000000  EXACT

64 is the fused Gated Delta Net chunk size for this hybrid model.
`probe_which_path.clj` shows there are in fact **three** self-consistent paths,
matching the three graph shapes llama.cpp reserves at load time
(`n_tokens = 1`, `16`, `1024`):

| decode call size | path |
| --- | --- |
| 1 | fused Gated Delta Net, autoregressive |
| 2 .. 63 | intermediate; every size in this range gives identical results |
| >= 64 | fused Gated Delta Net, chunked |

Each path is deterministic and internally consistent. They are three
implementations of one recurrence, not nondeterminism.

## Is this a llama.cpp defect? No.

The first write-up of this finding quoted "~0.1 to 0.5 nats", which sounds
alarming. That number was an artifact of the metric. Max |delta logprob| over a
top-50 is dominated by the tail, where a float32 logit of magnitude ~20 carries
absolute error in exactly that range from having ~7 significant digits.

Broken out by rank (`probe_magnitude.clj`, sequential vs chunked):

| rank | p | \|delta\| |
| --- | --- | --- |
| 0 | 0.992992 | **0.00046** |
| 1 | 0.003278 | 0.09180 |
| 30 | ~1e-8 | 0.28632 |

Top-1 agrees across all three paths; the top-5 ordering is identical. The head
of the distribution is stable to ~5e-4 nats. This is ordinary float32
non-associativity between kernel implementations.

llama.cpp does not promise bit-identical logits across batch configurations, and
no upstream issue describes this as a bug; the analogous CUDA repeatability
report (ggml-org/llama.cpp#7228) was closed. **We are not patching llama.cpp and
we are not filing an issue.** The measurement stands on its own as a constraint
on how this library may be used.

## What WAS our bug

The shim chunked `jl_eval` greedily at `n_batch`. A 2056-token prompt with
`n_batch = 2048` therefore emitted a 2048-token decode followed by an 8-token
decode, putting the tail of an ordinary prefill on the short-kernel path with
the caller doing nothing unusual:

    len=2056 (n_batch remainder=   8)   0.4747180939  DIVERGES
    len=2100 (n_batch remainder=  52)   0.3135204315  DIVERGES

`jl_eval` now chunks in BALANCED pieces -- `n_chunks = ceil(n/n_batch)`, sizes
differing by at most one -- so 2056 becomes 1028 + 1028. After the fix:

    len=2056   0.0000000000  EXACT
    len=2100   0.0000000000  EXACT
    len=2112   0.0000000000  EXACT
    len=2560   0.0000000000  EXACT

This does not make a short caller-supplied append exact, and cannot. It only
stops the shim from manufacturing a short chunk on its own.

## Calibration, not a magic number

64 is a property of this model's kernel selection, not of this library, so it is
not hard-coded. `jolt.llama/calibrate-append-exactness` binary-searches for the
shortest append that still reproduces a one-pass evaluation and reports the
probes it took, so a model with a different threshold -- or a non-monotone one --
is visible rather than silently mis-assumed. A pure-attention model is expected
to calibrate at 1.

`jolt.llama/append-divergence` returns a map, not a number, precisely because a
single number here misled the first analysis. It reports `:top1-abs` and
`:order-same?` alongside `:max-abs`.

## Scoring convention

`score-candidates` reads a candidate's FIRST token from the base logits -- which
came from a long prefill -- and reaches its remaining tokens by single-token
decodes. Those are two different paths. Therefore:

* single-token candidates are exactly comparable with each other
* equal-length candidates are exactly comparable with each other
* unequal-length candidates are comparable, but their sums mix the paths in
  different proportions, so a near-tie between a 1-token and a 4-token candidate
  is not meaningful at the 1e-3 level
* none of these are comparable against a logprob obtained by prefilling the
  candidate text inside one long prompt

The result map carries `:convention` and `:homogeneous?` so a caller can assert
the case it relies on. For a controller choosing among a fixed set of
single-token or equal-length actions -- the intended use -- the comparison is
exact.

## What M1 actually proved

M1 reports max |delta logprob| = 0.00000000 for a 2792-token spine and a
691-token delta. That is true and it is bit-exact. The precondition, now
explicit: the appended suffix was 691 tokens, comfortably above 64, so both arms
ran the chunked kernel. Had the delta been 30 tokens, the same protocol would
have shown ~0.2 in the tail while leaving the top-1 unchanged.

The exact-spine result is real. Its scope is "appends at or above the calibrated
threshold", and the property suite now asserts that scope on both sides:
`check-restore-matches-onepass-above-threshold!` asserts exactness above it, and
`check-short-append-diverges!` asserts the divergence below it, so a future
change that makes short appends exact fails a test instead of quietly
invalidating this document.
