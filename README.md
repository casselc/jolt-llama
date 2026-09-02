# jolt-llama

Direct embedded access to llama.cpp from [Jolt](https://github.com/casselc/jolt),
for programs that need a model to **rank a finite set of choices they already
constructed** -- not to generate text.

**The core library runs on stock Jolt.** No aspect compiler, no jolt-hegel, no
patched runtime. jolt-hegel is a test-only dependency in the `:test` alias;
aspect join points, when they land, are an optional enhancement and will not
become a runtime requirement.

## Architecture

    application
       |
    jolt-llama          Clojure-side semantics: lifecycle, token identity,
       |                candidate domains, error representation
    jolt.ffi
       |
    jolt_llama.{h,c}    small, stable C compatibility surface (ABI v1)
       |
    libllama / ggml     model execution, tensor kernels, KV/state mechanics

The shim exists so that llama.cpp's C API can change without breaking callers.
Users bind `jl_*`, never `llama_*`. The surface is deliberately narrow: opaque
handles, int32 tokens, size_t lengths, byte buffers, float scores, a `jl_status`
enum, a two-call sizing convention, a thread-local error string, and an ABI
version query.

**Jolt owns**: lifecycle semantics, token and state identity, candidate-domain
semantics, error representation, optional aspect join points.
**Native owns**: model execution, tensor kernels, llama.cpp state and context
mechanics.

No transformer math is implemented in Jolt.

## The two things this library is careful about

### 1. A stable text prefix is not a stable token prefix

This is the whole reason `load-state!` takes `:for-tokens`. BPE merges across the
boundary between stable and dynamic text, so the token vector for a spine is
**not** necessarily a prefix of the token vector for spine ++ delta.

Measured on the M1 workload: a 120-service spine tokenizes to 2793 tokens, but
only **2792** of them are a true prefix of `spine ++ delta`. Naive
concatenation of separately-tokenized parts gives 3484 tokens where the
canonical projection gives 3483.

    (llama/load-state! s st full-token-vector)
    ;; tokens are REQUIRED, not an option. Throws :state/prefix-mismatch with
    ;; :jolt.llama/diverges-at unless the saved tokens are a genuine prefix.
    ;; When the check was optional, the shortest call was also the unsafe one.

State is additionally bound to the model's content sha256, the shim ABI, the
sequence id, and hashes of both the token vector and the blob, so a state saved
against a different model or a different ABI is refused before any native call.

There is no text-prefix heuristic anywhere in this library, and no silent
rollback. A mismatch is refused with the divergence index attached; recovering
is the caller's decision, because only the caller knows whether a cold rebuild
or a rebase is the right answer.

The property suite asserts this from both directions: the contract holds for
generated suffixes, and two *demonstration* properties are expected to FAIL,
shrinking a minimal suffix that breaks the naive text-prefix assumption. A
passing demonstration would mean the generated domain never exercised a merge.

### 2. Not all logits are comparable

Hybrid models select a kernel by batch size. On the Qwen3.5-0.8B model used
here there are three self-consistent numerical paths (decode calls of 1 token,
2-63 tokens, and >= 64 tokens). Scores from different regimes agree at the head
of the distribution to ~5e-4 nats but not in the tail.

This is not a llama.cpp defect and we did not patch it. It is a constraint on
how scores may be compared, it is calibrated rather than assumed, and it is
asserted in both directions by the property suite.
See **[docs/EXACTNESS.md](docs/EXACTNESS.md)** -- including the part where the
first explanation for it was wrong.

## Minimal example

```clojure
(require '[jolt.llama :as llama])

(llama/with-model [m {:path (System/getenv "JOLT_LLAMA_MODEL")}]
  (llama/with-session [s m {:context-size 4096 :threads 4}]

    ;; ONE tokenization of the whole prompt. Never tokenize parts and concat.
    (let [tokens (llama/tokenize m "The capital of France is")]
      (llama/eval! s tokens)

      ;; the finite legal domain, constructed by trusted code
      (let [candidates [{:id :paris  :tokens (llama/tokenize m " Paris"  {:add-special? false})}
                        {:id :london :tokens (llama/tokenize m " London" {:add-special? false})}
                        {:id :berlin :tokens (llama/tokenize m " Berlin" {:add-special? false})}]
            result (llama/score-candidates s candidates {:state (llama/save-state s)})]

        (:convention result)    ;; => :teacher-forced/first-from-base-rest-single-token
        (:homogeneous? result)  ;; => true, so these scores are exactly comparable
        (:id (:best result))    ;; => :paris
        (map (juxt :id :rank :logprob-sum) (:candidates result))))))
```

See `examples/` for a runnable version.

## Results

| Milestone | Result |
| --- | --- |
| **M0** stock Jolt -> FFI -> shim -> libllama -> real logits | Jolt and the independent C reference agree exactly: tokens `[760 6511 314 9338 369]`, top-1 ` Paris` at `-1.555883` |
| **M1** exact state save/restore | `max_abs_dlogprob = 0.00000000` over all 100 common top-k entries, identical top-1, **4.09x** faster (1596 ms vs 6520 ms) |
| **M1** token seam | 2793-token spine, 2792-token true prefix; naive concat 3484 != canonical 3483 |
| **M1** alternating domains | 100 cycles, 3 domains, one session: 0 mismatches, 0 cross-domain contamination |
| **M2** candidate scoring | finite domain scored and ranked, convention documented and exposed |
| Properties | 8 contract properties pass; 2 seam demonstrations find and shrink counterexamples |

Full numbers: **[docs/RESULTS.md](docs/RESULTS.md)**.

## Building and running

```bash
# 1. build the shim against a llama.cpp build
cd native
make LLAMA_SRC=/path/to/llama.cpp LLAMA_LIB=/path/to/build/bin

# 2. point the library at it
export JOLT_LLAMA_LIB=$PWD/libjolt_llama.so
export JOLT_LLAMA_MODEL=/path/to/model.gguf

# 3. run the suite (stock Jolt for everything but the property tests)
./scripts/run-tests.sh            # all
./scripts/run-tests.sh m0         # one milestone
./scripts/run-tests.sh probes     # the exactness investigation
```

`scripts/run-tests.sh` sources a gitignored env file if present -- `.env.clean`
by preference, then `.env.local` -- so no path in this repository is specific to
any machine.

**Keep exactly one of them authoritative.** They began as copies and drifted:
`.env.local` went on pointing `LD_LIBRARY_PATH` at an older llama.cpp than the
shim was linked against, which is precisely the cross-build mismatch
`:runtime-id` exists to detect. The runner preferring `.env.clean` meant the
suite was fine and only a hand-run `source .env.local` was wrong, which is the
worst shape for a trap. If you keep both, make one a single `.` of the other.

Property tests additionally need the `:test` alias and a one-time install:

```bash
JOLT_CACHE_DIR=$PWD/.jolt-cache/jolt-hegel-<sha> jolt -A:test -m hegel.install
```

## Exact coordinates

Every source, toolchain, model and build coordinate needed to reproduce these
numbers is in **[config/coordinates.edn](config/coordinates.edn)**, including a
deviation from the bootstrap lock file and the fact that the llama.cpp working
tree is dirty (the diff is captured at `docs/llamacpp-worktree.diff`).

## Layout

    native/          the C compatibility shim, its Makefile, and smoke.c --
                     an independent C reference with no Jolt in the picture
    src/jolt/        the library
    test/jolt/       milestone tests, hegel properties, and probe_*.clj --
                     the exactness investigation, kept as executable evidence
    docs/            EXACTNESS.md, RESULTS.md, llamacpp-worktree.diff
    config/          coordinates.edn
    examples/        runnable minimal example
