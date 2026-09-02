# jolt-llama results

Machine: AMD Ryzen AI MAX+ 395 (Strix Halo), 16C/32T, Ubuntu 26.04, CPU-only.
Model: Qwen3.5-0.8B Q4_0, a hybrid (6 attention + 18 DeltaNet) architecture.
Exact coordinates: [config/coordinates.edn](../config/coordinates.edn).

Everything below was produced by the checked-in tests. `scripts/run-tests.sh`
reproduces it; `scripts/run-tests.sh probes` reproduces the exactness
investigation.

## Stock Jolt

The hard requirement, checked rather than asserted. `deps.edn` declares
`:deps {}` at the top level; jolt-hegel lives only in the `:test` alias.

    $ jolt run examples/rank_actions.clj      # no alias, no jolt-hegel
      ok   core library runs on stock Jolt with :deps {}

`scripts/run-tests.sh stock` runs this, so a future runtime dependency fails the
suite instead of going unnoticed.

## Verdict

**Embedded exact-state reuse is worth building on, with one documented
constraint that is not about state reuse at all.**

State save/restore is exactly transparent -- restoring is bit-for-bit
indistinguishable from never having cleared -- and delivers a 3.9x latency
reduction on a realistic controller prompt. The constraint discovered along the
way is that llama.cpp selects kernels by batch size on this architecture, so
logits from different batch-size regimes agree at the head of the distribution
but not in the tail. That is a rule about how scores may be compared. It is not
a defect, it is now calibrated rather than assumed, and it is asserted in both
directions by the property suite.

## M0 -- stock Jolt to real logits

Path: stock Jolt -> `jolt.ffi` -> `jl_*` shim -> libllama -> logits.
Compared against `native/smoke`, an independent C reference exercising the same
shim with no Jolt involved.

| | Jolt | C reference |
| --- | --- | --- |
| tokens for `"The capital of France is"` | `[760 6511 314 9338 369]` | identical |
| top-1 | ` Paris` @ `-1.555883` | ` Paris` @ `-1.555883` |
| logits length | 248320 | 248320 |
| state bytes | 20263632 | 20263632 |

All 27 M0 checks pass. `token-logprob` agrees with the top-k table, every
logprob is <= 0 (log-softmax over the full vocabulary), and logits are correctly
refused both after `clear!` and after a bare `load-state!`.

## M1 -- the token seam

    spine_text_tokens=2793  delta_tokens=691  full_tokens=3483  exact_boundary=2792
    boundary_lost_to_bpe_merge=1
    naive_concat_tokens=3484  canonical_tokens=3483  naive_concat_equals_canonical=false

A 120-service spine tokenizes to 2793 tokens, but only **2792** of them are a
true prefix of `spine ++ delta`. A BPE merge consumes the last one. Naive
concatenation of separately-tokenized parts produces a **different sequence**
(3484 tokens) from the canonical projection (3483).

This reproduces the frozen Halo finding through the new library, and it is the
reason `load-state!` requires `:for-tokens` and refuses a mismatch with
`:state/prefix-mismatch` rather than performing a rollback.

## M1 -- exact spine restore

    full_recompute_ms=6520   n_tokens=3483
    spine_eval_ms=5074       state_bytes=54566028  state_tokens=2792
    restore_ms=200           delta_eval_ms=1396    suffix_tokens=691
    warm_total_ms=1596 vs full_ms=6520  speedup=4.09x
    n_common_topk=100  max_abs_dlogprob=0.00000000  mean_abs_dlogprob=0.00000000
    top1_A=17021  top1_B=17021

**Bit-exact** across all 100 common top-k entries, identical top-1, at 4.09x.

Precondition, now explicit rather than incidental: the appended suffix was 691
tokens, above the calibrated 64-token threshold, so both arms ran the same
kernel. See [EXACTNESS.md](EXACTNESS.md).

## M1 -- alternating domains

Three tagged domains cycled through **one** session, 100 cycles, with a fresh
recompute oracle every 10th cycle. Each domain carries an unguessable 64-bit tag
so a leak is a real leak rather than a lucky prediction.

    domains=3  state_bytes=34651684  n_tokens=[1174 1176 1176]
    cycles=100  wall_ms=69063
    restore_ms p50=116 p95=125   delta_eval_ms p50=328 p95=339
    verified_cycles=10  mismatches=0  contaminated=0

**Zero** numerical mismatches against recompute and **zero** cross-domain
contamination.

## M2 -- candidate scoring

A finite candidate set, scored teacher-forced and ranked. No sampler, no
grammar, no free-form generation.

    id=continue rank=0 n=1 sum=-1.555883
    id=verify   rank=1 n=1 sum=-4.554820
    id=split    rank=2 n=1 sum=-5.692328

` Paris` scores `-1.555883`, exactly matching its top-k logprob -- the two code
paths cross-validate. The result map reports `:convention` and `:homogeneous?`
so callers can assert the comparability case they rely on rather than assume it.

## Properties (jolt-hegel v0.33.3)

    calibrated exact-append threshold: 64  monotone? true

    PASS computed token boundary is a true prefix of every extension (150 cases)
    PASS token-prefix-ok? agrees with token-for-token comparison (300 cases)
    PASS illegal lifecycle sequences fail safely, never crash (60 cases)
    PASS close! is idempotent for any repetition count (5 cases)
    PASS save/restore is transparent (10 cases)
    PASS restore + append >= 64 tokens equals a one-pass recompute (8 cases)
    PASS a short append (< 64) does NOT equal a one-pass recompute (8 cases)
    PASS load-state! refuses any non-extending token vector (17 cases)

    FAIL DEMONSTRATION: a text prefix is not a token prefix        <- the point
    FAIL DEMONSTRATION: tokenize(a++b) != tokenize(a)++tokenize(b) <- the point

The two demonstrations are expected to fail. Hegel finds and shrinks a minimal
suffix that breaks the naive text-prefix assumption; a pass would mean the
generated domain never exercised a BPE merge, which would make the run weaker
evidence, not stronger. They are counted separately from the contract
properties and do not gate the suite.

The first demonstration initially **passed**, which is how a weak test was
caught. Its stable text ended on a newline, and a newline rarely merges
rightwards, so 400 generated suffixes found nothing. The boundary character is
now drawn from realistic template endings (`ACTION:`, `owner=team`, `budget=`,
`p95=`, a bare space, and a newline) rather than fixed. It now finds and shrinks
a counterexample in 3 runs out of 3.

Note that hegel's `TooSlow` health check fired twice during development, both
times correctly: the properties were allocating a ~1 GiB context per generated
case. Sharing one session and shrinking the prompts fixed the cause rather than
silencing the check.

## Performance

Median and p95 over 7 trials. Workload: 2792-token spine, 691-token delta.

| Operation | p50 | p95 | Note |
| --- | ---: | ---: | --- |
| `open-model` | 287 ms | 311 ms | page cache not controlled |
| `new-session` (8192 ctx) | 25 ms | 32 ms | allocates the compute buffer |
| cold prefill, full prompt | 4357 ms | 4442 ms | 1.25 ms/token |
| spine only | 3425 ms | 3455 ms | 1.23 ms/token |
| `save-state` | 374 ms | 456 ms | 139 MiB/s |
| `load-state!` | 203 ms | 324 ms | 256 MiB/s |
| delta append | 956 ms | 994 ms | 691 tokens, 1.38 ms/token |
| **restore + delta** | **1119 ms** | 1153 ms | **3.90x** vs 4364 ms cold |
| score 5 single-token candidates | **4 ms** | 4 ms | no evaluation at all |
| score 5 multi-token candidates | 762 ms | 834 ms | a restore per candidate |

State costs **19544 bytes/token** (52.0 MiB for 2792 tokens), which is what
decides whether keeping N warm domains resident is affordable. It is large
because this is a hybrid model with recurrent state, not only a KV cache.

Two observations worth acting on:

* **Single-token candidates are ~200x cheaper than multi-token ones.** The base
  distribution already contains `P(first token | base)` for every candidate, so
  a single-token domain needs no evaluation. Multi-token candidates pay a full
  `load-state!` each. Controllers should prefer single-token action vocabularies,
  which is also the case where scores are exactly comparable.
* Prefill dominates everything. The warm path is worth it whenever the spine is
  large relative to the delta.

## Negative and corrective results, kept deliberately

**`save-state` was 19x slower than it should have been, and the perf harness is
what caught it.** It measured 7679 ms against `load-state!`'s 203 ms -- a 41x
asymmetry for what is the same memcpy in both directions. The cause was
`(take 4096 blob)` in the provenance hash: seqing a 54 MB byte array walks the
whole array before `take` sees its first element. Indexing instead brought it to
374 ms. The two-call sizing convention and `ffi/read-into!` were both innocent;
measuring the parts separately is what showed that.

**`score-candidates` could not be called twice.** Scoring multi-token candidates
ends with a restore, and a restore deliberately leaves no logits, so the second
call had no base distribution to read. Re-evaluating a token to regenerate them
would have moved the base onto a different kernel path and silently changed the
scoring convention, so instead the base log-probabilities are returned and can
be passed back via `:base-logprobs`.

**The shim manufactured its own kernel-path straddle.** Greedy chunking at
`n_batch` turned a 2056-token prompt into 2048 + 8, putting the tail of an
ordinary prefill on the short-kernel path with the caller doing nothing unusual.
Balanced chunking fixed it. Full detail in [EXACTNESS.md](EXACTNESS.md).

**The first explanation of the exactness finding was wrong.** A failing property
was initially read as save/restore being lossy. The control -- the same split
with no restore at all -- produced an identical number, which ruled that out.
Recorded because the wrong hypothesis is the reason the control exists.

**A 0.8B base model is a weak controller.** In `examples/rank_actions.clj` the
ranking barely moves between a healthy topology and a badly degraded one
(`hold` at `-0.827` vs `-0.833`), and `hold` wins both. The example demonstrates
the *shape* -- closed domain, exact comparability, selection by trusted code --
and is not evidence that this model makes good operational decisions.
