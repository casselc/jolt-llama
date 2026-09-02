/*
 * jolt_llama — a narrow, stable C ABI over libllama.
 *
 * WHY THIS EXISTS
 *
 * llama.cpp's C API is large and moves quickly: structs gain fields, enums gain
 * members, and functions are deprecated in place. Binding it directly from Jolt
 * would couple every Jolt release to a llama.cpp release. This shim is the
 * compatibility surface: it exposes opaque handles and boring scalars, so the
 * churn stops here.
 *
 * DESIGN RULES
 *
 *   - opaque pointers only; no llama.cpp struct crosses this boundary
 *   - int32 tokens, size_t lengths, float scores, byte buffers
 *   - every fallible call returns jl_status; nothing aborts the process for
 *     ordinary bad input, because a segfault in Jolt's FFI is not catchable
 *   - no process-global mutable state beyond the backend init refcount and a
 *     thread-local error buffer
 *   - two-call sizing convention: pass NULL/0 to learn a required length
 *
 * The token-identity contract (see docs/TOKEN-IDENTITY.md in the bootstrap
 * repo) is deliberately NOT enforced here. This layer reports exact token
 * vectors and exact state; Jolt owns the semantics of when reuse is legal.
 */
#ifndef JOLT_LLAMA_H
#define JOLT_LLAMA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------- ABI */

/*
 * Bumped whenever the meaning or layout of anything below changes. Jolt checks
 * this at load time so a stale .so fails loudly instead of misreading memory.
 */
/*
 * 2: jl_model_close now REFUSES while sessions are open (was: freed the model
 *    underneath them); jl_eval is append-only and rejects seq_id != 0;
 *    jl_state_load takes the token count the state represents. A v1 caller
 *    linked against v2 would silently pass its n_read pointer as n_tokens, so
 *    this is a hard version bump and Jolt checks it.
 */
#define JL_ABI_VERSION 2

int32_t     jl_abi_version(void);
/*
 * A stable identity for the NATIVE RUNTIME that serialized a state blob.
 *
 * Not decoration. The promotion evidence measured the same model producing a
 * 20263632-byte state on one llama.cpp build and 20263652 bytes on another, so
 * native state is NOT portable across builds -- and the shim ABI does not imply
 * native state-format compatibility, because the shim can be byte-identical
 * across two different llama.cpp trees.
 *
 * Embedded at BUILD time from JL_LLAMA_BUILD_ID (see native/Makefile), never
 * discovered by shelling out to git at inference time. A build that cannot
 * determine its coordinate reports "unknown:..." rather than claiming a false
 * one; for a promoted build, "unknown" is not acceptable and the Makefile says
 * so loudly.
 *
 * Shape: "llama.cpp:<40-char sha>:clean" or ":dirty".
 */
const char *jl_runtime_build_id(void);

const char *jl_llama_build(void);   /* deprecated alias for the above */

/* ------------------------------------------------------------- status */

typedef enum {
    JL_OK                 =  0,
    JL_ERR_GENERIC        = -1,
    JL_ERR_INVALID_ARG    = -2,
    JL_ERR_ALLOC          = -3,
    JL_ERR_MODEL_LOAD     = -4,
    JL_ERR_CONTEXT        = -5,
    JL_ERR_TOKENIZE       = -6,
    JL_ERR_DECODE         = -7,
    JL_ERR_STATE          = -8,
    JL_ERR_BUFFER_TOO_SMALL = -9,
    JL_ERR_NO_LOGITS      = -10,
    /*
     * jl_model_close was called while sessions created from that model are
     * still open. Refused rather than honoured: a jl_session holds both its
     * jl_model* and a llama_context built from that model's llama_model, so
     * freeing the model underneath a live session is a use-after-free the
     * caller cannot detect. The caller closes its sessions first; this library
     * will NOT close them on the caller's behalf, because a handle it did not
     * create is not its to invalidate.
     */
    JL_ERR_SESSIONS_ACTIVE = -11,
    /*
     * v0 is deliberately single-sequence: n_seq_max must be 1 and every seq_id
     * must be 0. The Jolt layer keeps ONE token ledger per session, so a second
     * native sequence would have no distinct token identity to check a restore
     * against -- and the token-identity contract is the point of this library.
     * Multi-sequence support needs a per-sequence identity design and its own
     * validation, not a relaxed bound here.
     */
    JL_ERR_SEQ_UNSUPPORTED = -12,
    /*
     * jl_eval was asked to write at a position other than the end of what the
     * session has already evaluated. v0 evaluation is append-only: the native
     * recurrent/KV state is not truncated to an arbitrary position, so writing
     * into the middle would leave the session's state and its token ledger
     * describing different things.
     */
    JL_ERR_NOT_APPEND     = -13,
    /*
     * A previous jl_eval failed partway through its chunks, so earlier chunks
     * had already mutated the context while n_resident still described the
     * state before the call. The ledger and the context disagree and nothing
     * at this layer can reconcile them, so the session refuses everything but
     * jl_session_clear (which makes them agree again, both empty) and
     * jl_session_close.
     */
    JL_ERR_POISONED       = -14
} jl_status;

/*
 * Message for the most recent failure ON THE CALLING THREAD. Returns the number
 * of bytes that would be written (excluding NUL); pass NULL/0 to size it.
 * Always NUL-terminates when buf is non-NULL and cap > 0.
 */
size_t jl_last_error(char *buf, size_t cap);

/* ------------------------------------------------------------ handles */

typedef struct jl_model   jl_model;
typedef struct jl_session jl_session;

/* ------------------------------------------------------------ runtime */

/*
 * Refcounted so several independent Jolt components can init/free without
 * fighting. llama_backend_init() is only called on the 0->1 transition.
 */
jl_status jl_runtime_init(void);
jl_status jl_runtime_free(void);

/* -------------------------------------------------------------- model */

typedef struct {
    int32_t n_gpu_layers;   /* 0 = CPU only */
    int32_t use_mmap;       /* bool */
    int32_t use_mlock;      /* bool */
    int32_t _reserved[5];   /* keeps the struct size stable across additions */
} jl_model_params;

void      jl_model_params_default(jl_model_params *out);
jl_status jl_model_open(const char *path, const jl_model_params *params, jl_model **out);
jl_status jl_model_close(jl_model *model);

int32_t   jl_model_n_vocab(const jl_model *model);
int32_t   jl_model_n_ctx_train(const jl_model *model);

/*
 * Stable identity for provenance: a hash over the model file's bytes is too
 * slow to do casually, so this returns llama.cpp's own description plus the
 * path the model was opened from. Jolt hashes the file separately when it needs
 * a content address.
 */
size_t    jl_model_desc(const jl_model *model, char *buf, size_t cap);

/* ------------------------------------------------------------ session */

typedef struct {
    uint32_t n_ctx;
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    int32_t  n_threads;
    int32_t  n_threads_batch;
    int32_t  _reserved[4];
} jl_session_params;

void      jl_session_params_default(jl_session_params *out);
jl_status jl_session_new(jl_model *model, const jl_session_params *params, jl_session **out);
jl_status jl_session_close(jl_session *session);

uint32_t  jl_session_n_ctx(const jl_session *session);

/*
 * Drop the cached state for sequence 0. seq_id must be 0: v0 is
 * single-sequence, and the old contract -- where a negative id cleared EVERY
 * sequence -- gave a caller who passed -1 by accident a different operation
 * than the one they named.
 */
jl_status jl_session_clear(jl_session *session, int32_t seq_id);

/* ----------------------------------------------------------- tokenize */

/*
 * Two-call sizing: tokens==NULL returns the required count in *n_out without
 * writing. Otherwise writes at most cap tokens. This is the ONLY supported way
 * to obtain a token vector — Jolt must never re-tokenize concatenated text and
 * assume the seam is stable.
 */
jl_status jl_tokenize(const jl_model *model,
                      const char *text, size_t text_len,
                      int32_t add_special, int32_t parse_special,
                      int32_t *tokens, size_t cap, size_t *n_out);

/* Render one token. Two-call sizing as above. */
jl_status jl_token_to_piece(const jl_model *model, int32_t token,
                            char *buf, size_t cap, size_t *n_out);

/* ---------------------------------------------------------------- eval */

/*
 * Evaluate n_tokens at absolute position pos0 in sequence seq_id. Logits are
 * produced for the final token only, which is what candidate scoring needs and
 * keeps the output buffer small.
 */
/*
 * APPEND-ONLY: pos0 must equal the session's resident token count, which
 * jl_session_n_resident reports. Evaluating into an earlier position, or past
 * the end, is JL_ERR_NOT_APPEND. The native recurrent/KV state is not truncated
 * to an arbitrary position, so allowing either would leave the state and the
 * caller's token ledger describing different sequences.
 */
jl_status jl_eval(jl_session *session, int32_t seq_id,
                  const int32_t *tokens, size_t n_tokens, int32_t pos0);

/* Tokens currently resident in the session, i.e. the only legal pos0. */
int32_t jl_session_n_resident(const jl_session *session);

/* --------------------------------------------------------------- logits */

/*
 * Raw logits for the last evaluated position: n_vocab floats. Two-call sizing.
 * These are logits, not probabilities; jl_logits_topk normalises.
 */
jl_status jl_logits(jl_session *session, float *out, size_t cap, size_t *n_out);

/*
 * Top-k token ids with LOG-SOFTMAX-normalised scores, descending. Computed in
 * one pass over the vocabulary with a max-subtracted softmax so the result is
 * comparable across calls and numerically stable.
 */
jl_status jl_logits_topk(jl_session *session, int32_t k,
                         int32_t *out_tokens, float *out_logprobs, size_t *n_out);

/* Log-probability of one specific token at the last evaluated position. */
jl_status jl_token_logprob(jl_session *session, int32_t token, float *out);

/* ---------------------------------------------------------------- state */

/*
 * Exact sequence state. jl_state_save/load move the native blob; Jolt attaches
 * the token vector and hashes. Size can change between calls, so always ask.
 */
jl_status jl_state_size(jl_session *session, int32_t seq_id, size_t *out);
jl_status jl_state_save(jl_session *session, int32_t seq_id,
                        uint8_t *buf, size_t cap, size_t *n_out);
/*
 * `n_tokens` is how many tokens the blob represents. The shim cannot derive it
 * from the blob and needs it to restore its resident count, without which the
 * next jl_eval could not tell an append from an overwrite.
 */
jl_status jl_state_load(jl_session *session, int32_t seq_id,
                        const uint8_t *buf, size_t len,
                        int32_t n_tokens, size_t *n_read);

#ifdef __cplusplus
}
#endif

#endif /* JOLT_LLAMA_H */
