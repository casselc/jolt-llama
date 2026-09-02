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

---

# v0 promotion hardening

A short pass after review, to close native-lifetime and state-identity gaps and
freeze a coordinate Samizdat can depend on. The earlier sections are the
original evidence and are not rewritten.

## Safety issues found

**A real use-after-free.** `jl_model_close` freed the `llama_model` and its
wrapper unconditionally, while a live `jl_session` still held both that
`jl_model*` and a `llama_context` built from it:

    model = jl_model_open(...)
    sess  = jl_session_new(model, ...)
    jl_model_close(model)          <- freed
    jl_eval(sess, ...)             <- used

**A comment that was wrong in the dangerous direction.** `jl_model_close`
claimed the cleared fields made a second raw close "a clean
JL_ERR_INVALID_ARG ... rather than a use-after-free". After `free(model)` the
struct is gone, so a second call reads freed memory *before* it can check
anything. Idempotent close is a property of the Jolt wrapper and cannot be
provided at the C layer without an out-of-band handle table. Deliberately not
demonstrated by calling a freed pointer, which would prove it by committing the
fault.

**Sequence identity was misrepresented.** One token ledger per session, but
`seq-id` and `n-seq-max` were exposed, so a second native sequence would have
had no distinct token identity to check a restore against.

**Evaluation could silently desynchronise.** `eval!` took an arbitrary `:pos`
and truncated its token vector there; the native recurrent/KV state was not
truncated to the same position.

**State identity was not identity.** Saved state was bound to model *path* and
*description*. A path can point at different bytes between runs, and
descriptions collide across quantizations.

**The core invariant was bypassable, and the bypass was the shortest call.**
`(load-state! session state)` skipped the token-prefix check entirely.

## Changes

| Area | Change |
| --- | --- |
| Ownership | `jl_model` refcounts sessions; `jl_model_close` returns `JL_ERR_SESSIONS_ACTIVE` while any live, non-destructively. Jolt throws `:model/sessions-active`. Child sessions are never closed behind the caller's back. |
| Sequence | `n_seq_max != 1` and `seq_id != 0` refused (`:seq/unsupported`), not clamped. v0 is deliberately single-sequence. |
| Append | `:pos` removed; there is no public unsafe variant. The shim tracks its own resident count and answers `JL_ERR_NOT_APPEND`, so the rule holds even if the Jolt ledger were wrong. |
| State identity | Bound to the GGUF's sha256 (computed once at open), ABI, seq id, token vector + hash, blob size + hash. `state-compatible?` names *which* rule failed, and runs before the native call — `llama_state_seq_set_data` given a foreign blob does not reliably fail, it can succeed into nonsense. |
| Token prefix | `tokens` is now a required positional argument to `load-state!`. The unchecked path is private. |
| ABI | 1 → 2, checked exactly rather than as a floor. |
| FFI | `:blocking` on the calls that measured in the hundreds of ms (model open, session new, eval, state save/load); deliberately not on scalar queries. |

Model identity is keyed on **content, not handle**. Two handles over the same
GGUF are the same model, and refusing that restore would be the kind of false
negative that teaches callers to reach for an unchecked path. There is a
property asserting that restore is *accepted*.

## Clean llama.cpp coordinate

The prior coordinate was a fork SHA plus a captured dirty diff. Two findings:

* The diff is XDNA2/NPU offload for BitNet I2_S GEMMs, gated at build *and*
  runtime. Qwen3.5 Q4_0 never takes the I2_S path and the build never set
  `BITNET_XDNA_RUNTIME_DIR` — `nm` shows zero `bitnet_xdna` symbols. It was
  never needed.
* `390c3077` is a **fork** commit, not an upstream ancestor. The GitHub API
  resolves it because forks share a commit network; `compare` reports it as
  diverged. Its `ggml-cpu/CMakeLists.txt` references
  `../../../../src/ggml-bitnet-lut.cpp`, a file in BitNet's parent tree, so it
  **cannot build standalone**. The first clone-and-build attempt failed exactly
  there.

Promoted instead:

    ggml-org/llama.cpp  b81c99b479d4c24e5eeca10de99032ebd343ef8f   (clean tree)

One API difference, absorbed by the shim: llama.cpp replaced
`use_mmap`/`use_mlock` with a `load_mode` enum. `native/Makefile` probes
`llama.h` and defines `JL_HAS_LOAD_MODE`, so one source builds against both and
`jl_model_params` keeps its two booleans. Probed rather than version-gated
because llama.cpp exposes no API version macro.

## Regression gate on the clean build

| Gate | Result |
| --- | --- |
| C oracle | `SMOKE OK` at ABI 2, including ownership and append assertions |
| M0 | logits **bit-identical** to the experimental build — token 11751 at `-1.555883` |
| M1 | `max_abs_dlogprob = 0.00000000` at **4.01x**, same 1-token BPE seam |
| Alternating | 100 cycles, **0** mismatches, **0** contamination |
| Hegel | **14** contract properties pass, 0 failures; both seam demonstrations still find and shrink counterexamples |
| Stock Jolt | example runs with no alias; `:deps {}` still true |

Two different llama.cpp builds producing the same logit to six decimals is the
strongest single piece of evidence here.

The state blob is 20263652 bytes on the clean build against 20263632 on the
fork — a 20-byte serialization difference. Harmless, and exactly why state now
carries an ABI and a model content id instead of being assumed portable.

## Performance: one regression, expected and bounded

| Operation | before | after |
| --- | ---: | ---: |
| **open-model** | 287 ms | **1437 ms** |
| new-session | 25 ms | 26 ms |
| cold prefill | 4357 ms | 4240 ms |
| save-state | 374 ms | 375 ms |
| load-state! | 203 ms | 201 ms |
| restore + delta | 1119 ms (3.90x) | 1100 ms (3.90x) |
| 5 single-token candidates | 4 ms | 4 ms |

The single regression is sha256 over a 0.52 GiB GGUF, computed **once** when a
model is opened. The file is never hashed again — not per state save, not per
load, not per decision. Buying a real artifact identity for about a second at
startup is the right trade, since a descriptor bound to a path and a
description was not an identity at all.

## Exact coordinates

    jolt         v0.8.0  ccd6a73fd20b8e69cba654e008024958d5b4bd8a
    llama.cpp    b81c99b479d4c24e5eeca10de99032ebd343ef8f  (clean)
    jolt-hegel   2186afd9fef8b8ba766af5ea06b517e6af36cd4e  (libhegel v0.33.3)
    model        qwen3.5-0.8b-q4_0.gguf
                 sha256 57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf

## Samizdat canary regression

The embedded-controller canary was validated against jolt-llama
`6ec88472bc63371ac9b70729ffbd58e35f35ce8d` on the experimental llama.cpp build.
That evidence is not rewritten. Re-run against the hardened coordinate and the
clean llama.cpp build, the results are **byte-identical**:

    situation "healthy":      hold -1.08148  margin 1.68001  decision act
    situation "api degraded": hold -1.03207  margin 1.58262  decision act

    ranking CHANGED with state: false
    journal round trip: read back from SQLite, decision "act", no machine state

Same legal-domain behaviour, same selected action, same scores to five
decimals, same journal/provenance, no lifecycle problem. Identical across BOTH
a jolt-llama hardening and a llama.cpp build change, which is a stronger
statement than either alone.

Run through a temporary local-path override; the canary's own pin was left at
the reviewed SHA and the override was reverted.

## Deferred

* **Aspect join points (§12).** Not attempted in this pass. They are P1 behind
  the safety items, and the safety items plus the clean-build migration used
  the budget. Nothing in the library depends on them and no runtime dependency
  on an aspect compiler was added.
* **Multi-sequence support.** Explicitly out of scope: v0 is single-sequence by
  construction, and candidate forking via `llama_memory_seq_cp` needs a
  per-sequence identity design with its own validation.
* **The 20-byte state format difference between builds** is recorded, not
  investigated. State is not claimed to be portable across llama.cpp builds,
  and the ABI/content checks now enforce that.

---

# v0 final correctness pass

The last pass before freezing v0. Three P0 defects, two P1 contracts, and a
documentation claim that was not true.

## P0-1 — saved state was not bound to the llama.cpp build

The promotion evidence already showed the same model serializing to 20263632
bytes on one build and 20263652 on another, and then concluded that ABI plus
model content id kept them apart. That was wrong. The shim can be byte-identical
across two different llama.cpp trees, so:

    same GGUF + same shim ABI + different llama.cpp  ->  state-compatible? said yes

Saved state now carries `:runtime-id`, embedded in the shim at **build** time by
`native/Makefile` from the `LLAMA_SRC` checkout — never discovered by shelling
out to git during inference:

    llama.cpp:b81c99b479d4c24e5eeca10de99032ebd343ef8f:clean

`state-compatible?` checks it **separately from the ABI**, before any native
call, and answers `:state/runtime-mismatch`. A tree that is not a git checkout
yields `unknown:…`, which never matches; `make promote` refuses to build a
library whose coordinate is unknown or dirty, so a promoted build cannot ship
state it could not attribute.

## P0-2 — single-sequence did not mean single-sequence

Audited every entry point that accepts a sequence identity. Holes found and
closed:

| entry point | was | is |
| --- | --- | --- |
| `jl_session_new` | accepted `n_seq_max == 0` and clamped it to 1 | requires **exactly** 1 |
| `jl_session_clear` | `seq_id < 0` meant "clear everything" | rejects any nonzero |
| `jl_state_size` | no check | rejects any nonzero |
| `jl_state_save` | no check | rejects any nonzero |
| Jolt `clear!` | exposed `:seq-id` | no such option |
| Jolt `save-state` | exposed `:seq-id` | no such option |

Clamping 0 to 1 is the same class of lie as accepting 4: the caller's stated
intent and the library's behaviour differ and nothing says so. And `-1` meaning
"all sequences" is a multi-sequence concept smuggled into a single-sequence
API — a caller passing it by accident got a *different operation* than the one
they named.

The public Jolt API no longer advertises a capability that does not exist.

## P0-3 — candidate rewind accepted the wrong base

`score-candidates` reads every candidate's **first** token from the logits the
session is holding *now* (base A), then reaches later tokens by restoring the
supplied state. Nothing checked that the state *was* A. Given a different but
perfectly valid state B, the score mixed

    P(t1 | A)   with   P(t2 | B, t1)

a conditional that describes no sequence — and `state-compatible?` accepted B
happily, because B is a good state for this model, runtime and ABI.

Now required: `(= (:tokens state) @(:tokens session))`. **Equality, not prefix**
— a strict prefix of the base is individually reusable as a checkpoint and still
wrong here, because the tokens between it and the base would silently vanish
from the conditional. Refused with `:score/base-state-mismatch` *before* any
candidate touches native state.

The ledger bug beside it is also fixed. The old code restored native state and
then separately forced the token atom back to `base-tokens`, so if the two ever
disagreed the ledger would confidently describe a sequence the session was not
holding — and `load-state!`'s prefix check, the thing that would have caught it,
reads that same ledger. Both sides now move through one operation.

The single-token fast path is unchanged and asserted to stay that way: no saved
state, no restore, no candidate evaluation, and each score equal to the direct
base log-probability.

## P1-4 — the hashes were named better than they are

`:token-hash` and `:state-hash` were a 32-bit polynomial rolling value, and the
state one only read the first 4096 bytes. Nothing about that is a content
identity, and the earlier name (`sha256-hex`) implied one.

At the time this was written the conclusion was Option B — keep the cheap
checks, rename them honestly to `:token-check` / `:state-check`.

**SUPERSEDED. They are real SHA-256 digests now.** The reasoning above was
over-constrained: the requirement was that the core run on *stock Jolt*, and
`:deps {}` was an implementation of that which I then treated as the
requirement. jolt-crypto is a first-party library, not a runtime patch.

Measuring changed the answer again. jolt-crypto works, but on the sizes that
matter it is ~1 MB/s — FFI marshalling, not OpenSSL:

| input | via jolt-crypto |
| --- | ---: |
| token vector, 11 KB | 2 ms |
| state blob, 52 MiB | **~47 000 ms** |

So the dependency was never the obstacle; moving 52 MiB through an FFI to reach
it is. The shim already holds the blob in native memory, where SHA-256 costs
~350 ms and needs no dependency at all. `jl_sha256_hex` is FIPS 180-4 in ~130
lines, checked against the published vectors in `smoke.c` before anything trusts
it — which caught a two-block padding bug that passed the empty and `"abc"`
vectors and failed the 56-byte one.

The fields are `:token-sha256` (over the canonical little-endian int32 encoding)
and `:state-sha256` (over the whole blob), with
`:state/token-digest-mismatch` and `:state/blob-digest-mismatch`.

**One trade, stated because it is a real weakening.** The token digest is
verified on every restore (~2 ms). The blob digest is verified only when the
state did not originate in the restoring session: candidate rewind restores once
per multi-token candidate, and verifying each time cost ~345 ms and took the
exact-spine speedup from 3.92x to 3.00x, to detect only memory corruption of an
immutable in-process array this library created and never hands out.
`verify-state-digest` forces the full check.

  save-state 376 → 729 ms; load-state! and restore+delta unchanged.

## P1-5 — the v0 concurrency contract

Handles are **thread-confined**: one session must not be evaluated from two
threads at once, and this library does not claim concurrent inference safety.

**Close is the deliberate exception**, because cleanup is where accidental
concurrent calls actually happen — two `finally` blocks, a shutdown hook racing
a worker. The handle carries a state atom moved only by `compare-and-set!`:

    :open --CAS--> :closing --> :closed
                            or  :close-failed   (native close threw)

Only the thread that wins `:open -> :closing` calls native free. A boolean could
not express this: read-then-write left a window where two threads both saw
`false`, both wrote `true`, and both called free on one pointer.

`:close-failed` is terminal, so a native close that throws cannot strand the
handle at `:closing` where every later caller would be told `:already-closed`
about a resource that was never freed. It is **defensive and untested**: no
valid close has a practical failure path, and reaching it would need a second
shipped fault-injection hook. Recorded as a known gap rather than claimed as
covered.

`init-runtime!` is CAS'd the same way — it was read-then-write, so N threads
opening their first model could all call `jl_runtime_init`, whose native
refcount is a plain int.

The native runtime refcount is **not** thread-safe and is not part of ordinary
handle lifecycle — Jolt initialises the runtime once, on first model open, and
never frees it. Stated rather than defended with a lock nothing needs yet.

## The state compatibility coordinate, in full

    { shim ABI version
      native runtime build id      llama.cpp:<sha>:clean, and an
                                   unattributable "unknown:" id never matches,
                                   not even another "unknown:"
      model content SHA-256        of the GGUF, computed once at open
      sequence id                  0 in v0
      exact token vector           plus its SHA-256
      native state byte count      plus the blob's SHA-256

      and, for a candidate-rewind checkpoint only:
      originating session id
      unique per-save state id     the session must be sitting ON it }

Token equality is not NUMERICAL equality — the same token vector reached by a
different call structure produces different logits below the calibrated append
threshold — so a rewind checkpoint needs an evaluation identity, not just
matching tokens. A monotonic counter was tried first and was wrong: restoring
rewound it, so two divergent evaluations could share a number.

Every one is checked **before** the native state load, because
`llama_state_seq_set_data` given a foreign blob does not reliably fail — it can
succeed into nonsense.
