/*
 * jolt_llama — implementation of the narrow C ABI over libllama.
 *
 * Every exported function validates its arguments before touching libllama.
 * That is not defensive decoration: Jolt's FFI cannot catch SIGSEGV, so a null
 * handle arriving from a buggy caller must become JL_ERR_INVALID_ARG rather
 * than a dead process. The Hegel lifecycle properties deliberately generate
 * illegal call sequences against this boundary.
 */
#include "jolt_llama.h"

#include "llama.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* --------------------------------------------------------------- errors */

/*
 * Thread-local so concurrent Jolt threads do not scribble on each other's
 * message. Bounded; a truncated message is fine, a heap allocation on the
 * error path is not.
 */
#define JL_ERRBUF 512
static _Thread_local char g_err[JL_ERRBUF];

static void jl_set_error(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_err, sizeof(g_err), fmt, ap);
    va_end(ap);
}

static void jl_clear_error(void) { g_err[0] = '\0'; }

size_t jl_last_error(char *buf, size_t cap) {
    size_t n = strlen(g_err);
    if (buf && cap > 0) {
        size_t c = n < cap - 1 ? n : cap - 1;
        memcpy(buf, g_err, c);
        buf[c] = '\0';
    }
    return n;
}

/* --------------------------------------------------------------- handles */

struct jl_model {
    struct llama_model       *model;
    const struct llama_vocab *vocab;
    char                     *path;
    /*
     * Sessions created from this model and not yet closed. jl_model_close
     * refuses while this is nonzero, because a jl_session holds this pointer
     * AND a llama_context built from model->model: freeing either underneath a
     * live session is a use-after-free the caller has no way to detect.
     */
    int                       active_sessions;
};

struct jl_session {
    struct llama_context *ctx;
    jl_model             *model;
    /*
     * Tokens resident in sequence 0, maintained here rather than trusted from
     * the caller. It is the only legal pos0 for jl_eval, which makes evaluation
     * append-only; see JL_ERR_NOT_APPEND.
     */
    int32_t               n_resident;
    /*
     * Whether the most recent jl_eval produced logits. Asking for logits before
     * any eval is a caller error we can report precisely instead of handing
     * back a stale or uninitialised buffer.
     */
    int                   have_logits;
};

/* --------------------------------------------------------------- runtime */

/*
 * llama_backend_init/free are process-global. Refcount so independent
 * components can bracket their own use without tearing the backend out from
 * under each other. Not thread-safe by design: Jolt initialises the runtime
 * once, early. Documented rather than hidden behind a lock that would imply a
 * stronger guarantee than we test.
 */
static int g_runtime_refs = 0;

int32_t jl_abi_version(void) { return JL_ABI_VERSION; }

const char *jl_llama_build(void) {
    static char buf[128];
    snprintf(buf, sizeof(buf), "llama.cpp");
    return buf;
}

jl_status jl_runtime_init(void) {
    jl_clear_error();
    if (g_runtime_refs == 0) {
        llama_backend_init();
    }
    g_runtime_refs++;
    return JL_OK;
}

jl_status jl_runtime_free(void) {
    jl_clear_error();
    if (g_runtime_refs <= 0) {
        jl_set_error("jl_runtime_free with no matching init");
        return JL_ERR_INVALID_ARG;
    }
    g_runtime_refs--;
    if (g_runtime_refs == 0) {
        llama_backend_free();
    }
    return JL_OK;
}

/* ----------------------------------------------------------------- model */

void jl_model_params_default(jl_model_params *out) {
    if (!out) return;
    memset(out, 0, sizeof(*out));
    out->n_gpu_layers = 0;      /* CPU by default: this pass is CPU-oriented */
    out->use_mmap     = 1;
    out->use_mlock    = 0;
}

jl_status jl_model_open(const char *path, const jl_model_params *params, jl_model **out) {
    jl_clear_error();
    if (!path || !out) { jl_set_error("jl_model_open: null path or out"); return JL_ERR_INVALID_ARG; }
    *out = NULL;

    jl_model_params p;
    if (params) p = *params; else jl_model_params_default(&p);

    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = p.n_gpu_layers;
    mp.use_mmap     = p.use_mmap  ? true : false;
    mp.use_mlock    = p.use_mlock ? true : false;

    struct llama_model *m = llama_model_load_from_file(path, mp);
    if (!m) { jl_set_error("llama_model_load_from_file failed for '%s'", path); return JL_ERR_MODEL_LOAD; }

    jl_model *h = (jl_model *) calloc(1, sizeof(jl_model));
    if (!h) { llama_model_free(m); jl_set_error("out of memory"); return JL_ERR_ALLOC; }
    h->model = m;
    h->vocab = llama_model_get_vocab(m);
    h->path  = strdup(path);

    *out = h;
    return JL_OK;
}

jl_status jl_model_close(jl_model *model) {
    jl_clear_error();
    if (!model) { jl_set_error("jl_model_close: null model"); return JL_ERR_INVALID_ARG; }
    if (model->active_sessions > 0) {
        jl_set_error("jl_model_close: %d session(s) still open; close them first",
                     model->active_sessions);
        return JL_ERR_SESSIONS_ACTIVE;
    }
    if (model->model) llama_model_free(model->model);
    free(model->path);
    model->model = NULL;
    model->vocab = NULL;
    model->path  = NULL;
    free(model);
    /*
     * NOTE, corrected: the fields are cleared before the free purely to make a
     * debugger's view unambiguous. They do NOT make a second raw jl_model_close
     * safe. After free(model) the struct is gone, so a second call READS FREED
     * MEMORY before it can check anything -- the earlier comment here claimed
     * the opposite and was wrong. Idempotent close is provided by the Jolt
     * wrapper, which drops its pointer and never calls in twice; there is no
     * way to make it safe at this layer without an out-of-band handle table.
     */
    return JL_OK;
}

int32_t jl_model_n_vocab(const jl_model *model) {
    if (!model || !model->vocab) return -1;
    return llama_vocab_n_tokens(model->vocab);
}

int32_t jl_model_n_ctx_train(const jl_model *model) {
    if (!model || !model->model) return -1;
    return llama_model_n_ctx_train(model->model);
}

size_t jl_model_desc(const jl_model *model, char *buf, size_t cap) {
    if (!model || !model->model) return 0;
    char tmp[512];
    int n = llama_model_desc(model->model, tmp, sizeof(tmp));
    if (n < 0) n = 0;
    size_t len = (size_t) n;
    if (buf && cap > 0) {
        size_t c = len < cap - 1 ? len : cap - 1;
        memcpy(buf, tmp, c);
        buf[c] = '\0';
    }
    return len;
}

/* --------------------------------------------------------------- session */

void jl_session_params_default(jl_session_params *out) {
    if (!out) return;
    memset(out, 0, sizeof(*out));
    out->n_ctx           = 4096;
    out->n_batch         = 2048;
    out->n_ubatch        = 2048;
    out->n_seq_max       = 1;
    out->n_threads       = 4;
    out->n_threads_batch = 4;
}

jl_status jl_session_new(jl_model *model, const jl_session_params *params, jl_session **out) {
    jl_clear_error();
    if (!model || !model->model || !out) { jl_set_error("jl_session_new: null model or out"); return JL_ERR_INVALID_ARG; }
    *out = NULL;

    jl_session_params p;
    if (params) p = *params; else jl_session_params_default(&p);

    /*
     * v0 is single-sequence. Rejected here rather than clamped: a caller that
     * asked for 4 sequences and silently got 1 would build on an assumption the
     * token-identity contract cannot support.
     */
    if (p.n_seq_max > 1) {
        jl_set_error("jl_session_new: n_seq_max %u unsupported; v0 is single-sequence",
                     p.n_seq_max);
        return JL_ERR_SEQ_UNSUPPORTED;
    }

    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = p.n_ctx;
    cp.n_batch         = p.n_batch;
    cp.n_ubatch        = p.n_ubatch;
    cp.n_seq_max       = 1;
    cp.n_threads       = p.n_threads;
    cp.n_threads_batch = p.n_threads_batch;

    struct llama_context *ctx = llama_init_from_model(model->model, cp);
    if (!ctx) { jl_set_error("llama_init_from_model failed"); return JL_ERR_CONTEXT; }

    jl_session *s = (jl_session *) calloc(1, sizeof(jl_session));
    if (!s) { llama_free(ctx); jl_set_error("out of memory"); return JL_ERR_ALLOC; }
    s->ctx = ctx;
    s->model = model;
    s->have_logits = 0;
    s->n_resident = 0;
    /* claimed only once the session is fully built, so a failed construction
     * never leaves a count the caller can neither see nor release */
    model->active_sessions++;

    *out = s;
    return JL_OK;
}

int32_t jl_session_n_resident(const jl_session *session) {
    if (!session || !session->ctx) return -1;
    return session->n_resident;
}

jl_status jl_session_close(jl_session *session) {
    jl_clear_error();
    if (!session) { jl_set_error("jl_session_close: null session"); return JL_ERR_INVALID_ARG; }
    /* released exactly once: the count is dropped here, before the struct goes,
     * and the model pointer is read only while the session still owns it */
    if (session->model && session->model->active_sessions > 0) {
        session->model->active_sessions--;
    }
    if (session->ctx) llama_free(session->ctx);
    session->ctx = NULL;
    session->model = NULL;
    session->n_resident = 0;
    free(session);
    /*
     * Same correction as jl_model_close: after free(session) a second raw call
     * reads freed memory. Idempotence lives in the Jolt wrapper, not here.
     */
    return JL_OK;
}

uint32_t jl_session_n_ctx(const jl_session *session) {
    if (!session || !session->ctx) return 0;
    return llama_n_ctx(session->ctx);
}

jl_status jl_session_clear(jl_session *session, int32_t seq_id) {
    jl_clear_error();
    if (!session || !session->ctx) { jl_set_error("jl_session_clear: null session"); return JL_ERR_INVALID_ARG; }
    llama_memory_t mem = llama_get_memory(session->ctx);
    if (seq_id < 0) {
        llama_memory_clear(mem, true);
    } else {
        llama_memory_seq_rm(mem, seq_id, -1, -1);
    }
    session->have_logits = 0;
    /* nothing resident means the next eval must start at 0 */
    session->n_resident = 0;
    return JL_OK;
}

/* -------------------------------------------------------------- tokenize */

jl_status jl_tokenize(const jl_model *model,
                      const char *text, size_t text_len,
                      int32_t add_special, int32_t parse_special,
                      int32_t *tokens, size_t cap, size_t *n_out) {
    jl_clear_error();
    if (!model || !model->vocab || !text || !n_out) {
        jl_set_error("jl_tokenize: null model, text or n_out");
        return JL_ERR_INVALID_ARG;
    }
    *n_out = 0;

    /* Negative return is -(required count): the sizing call. */
    int32_t n = llama_tokenize(model->vocab, text, (int32_t) text_len,
                               NULL, 0, add_special ? true : false,
                               parse_special ? true : false);
    size_t need = (size_t) (n < 0 ? -n : n);
    *n_out = need;

    if (!tokens) return JL_OK;                 /* sizing call */
    if (cap < need) {
        jl_set_error("jl_tokenize: buffer holds %zu, need %zu", cap, need);
        return JL_ERR_BUFFER_TOO_SMALL;
    }

    int32_t w = llama_tokenize(model->vocab, text, (int32_t) text_len,
                               (llama_token *) tokens, (int32_t) cap,
                               add_special ? true : false,
                               parse_special ? true : false);
    if (w < 0) { jl_set_error("llama_tokenize failed (%d)", w); return JL_ERR_TOKENIZE; }
    *n_out = (size_t) w;
    return JL_OK;
}

jl_status jl_token_to_piece(const jl_model *model, int32_t token,
                            char *buf, size_t cap, size_t *n_out) {
    jl_clear_error();
    if (!model || !model->vocab || !n_out) { jl_set_error("jl_token_to_piece: null arg"); return JL_ERR_INVALID_ARG; }
    *n_out = 0;

    char tmp[256];
    int32_t n = llama_token_to_piece(model->vocab, token, tmp, sizeof(tmp), 0, true);
    if (n < 0) { jl_set_error("llama_token_to_piece failed (%d)", n); return JL_ERR_GENERIC; }
    *n_out = (size_t) n;

    if (!buf) return JL_OK;
    if (cap < (size_t) n) { jl_set_error("jl_token_to_piece: buffer too small"); return JL_ERR_BUFFER_TOO_SMALL; }
    memcpy(buf, tmp, (size_t) n);
    return JL_OK;
}

/* ------------------------------------------------------------------ eval */

jl_status jl_eval(jl_session *session, int32_t seq_id,
                  const int32_t *tokens, size_t n_tokens, int32_t pos0) {
    jl_clear_error();
    if (!session || !session->ctx || !tokens) { jl_set_error("jl_eval: null session or tokens"); return JL_ERR_INVALID_ARG; }
    if (n_tokens == 0) { jl_set_error("jl_eval: n_tokens == 0"); return JL_ERR_INVALID_ARG; }
    if (seq_id != 0) {
        jl_set_error("jl_eval: seq_id %d unsupported; v0 is single-sequence (seq 0)", seq_id);
        return JL_ERR_SEQ_UNSUPPORTED;
    }
    /*
     * Append-only. pos0 must be exactly what is already resident: earlier would
     * overwrite state this shim does not truncate, later would leave a hole.
     * Either way the session's native state and the caller's token vector would
     * stop describing the same sequence, which is precisely the confusion the
     * token-identity contract exists to prevent.
     */
    if (pos0 != session->n_resident) {
        jl_set_error("jl_eval: pos0 %d but %d tokens are resident; evaluation is append-only",
                     pos0, session->n_resident);
        return JL_ERR_NOT_APPEND;
    }

    const uint32_t n_ctx = llama_n_ctx(session->ctx);
    if ((size_t) pos0 + n_tokens > (size_t) n_ctx) {
        jl_set_error("jl_eval: pos0 %d + %zu tokens exceeds n_ctx %u", pos0, n_tokens, n_ctx);
        return JL_ERR_INVALID_ARG;
    }

    /*
     * Built by hand rather than via llama_batch_get_one so we control seq_id
     * and request logits for the final token only. Chunked at n_batch because
     * a caller may hand us a whole 1600-token spine at once.
     *
     * The chunks are BALANCED rather than greedy, and that is a correctness
     * property, not tidiness. Hybrid models select a kernel by batch size: this
     * repo measured a Gated Delta Net model with three self-consistent
     * paths, switching at 2 and at 64 tokens, so a decode call carrying fewer
     * than 64 tokens runs different arithmetic than the same tokens inside a
     * larger call. The head of the distribution barely moves (top-1 agrees to
     * 5e-4 nats) but the tail does, and mixing regimes makes scores
     * incomparable at that scale. Greedy chunking of 2056 tokens at n_batch 2048
     * emits a 2048 chunk and then an 8-token chunk, silently putting the tail
     * of an ordinary prefill on the short-kernel path. Balanced chunking emits
     * 1028 + 1028 instead. See docs/EXACTNESS.md.
     *
     * This does not and cannot make a short caller-supplied append exact; it
     * only stops the shim from manufacturing a short chunk on its own.
     */
    const uint32_t n_batch = llama_n_batch(session->ctx);
    const size_t n_chunks = (n_tokens + n_batch - 1) / n_batch;
    const size_t base_len = n_tokens / n_chunks;
    const size_t remainder = n_tokens % n_chunks;

    size_t off = 0;
    for (size_t ci = 0; ci < n_chunks; ci++) {
        /* spread the remainder one token at a time over the leading chunks, so
         * chunk sizes differ by at most 1 and none is pathologically small */
        const size_t chunk = base_len + (ci < remainder ? 1 : 0);
        const int last_chunk = (ci + 1) == n_chunks;

        struct llama_batch batch = llama_batch_init((int32_t) chunk, 0, 1);
        if (!batch.token) { jl_set_error("llama_batch_init failed"); return JL_ERR_ALLOC; }

        for (size_t i = 0; i < chunk; i++) {
            batch.token[i]     = (llama_token) tokens[off + i];
            batch.pos[i]       = (llama_pos) (pos0 + (int32_t) (off + i));
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = (llama_seq_id) seq_id;
            batch.logits[i]    = (last_chunk && i + 1 == chunk) ? 1 : 0;
        }
        batch.n_tokens = (int32_t) chunk;

        int32_t rc = llama_decode(session->ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            session->have_logits = 0;
            jl_set_error("llama_decode failed (%d) at offset %zu", rc, off);
            return JL_ERR_DECODE;
        }
        off += chunk;
    }

    session->have_logits = 1;
    session->n_resident += (int32_t) n_tokens;
    return JL_OK;
}

/* ---------------------------------------------------------------- logits */

static jl_status jl_logits_ptr(jl_session *session, float **out, int32_t *n_vocab) {
    if (!session || !session->ctx) { jl_set_error("null session"); return JL_ERR_INVALID_ARG; }
    if (!session->have_logits) { jl_set_error("no logits: call jl_eval first"); return JL_ERR_NO_LOGITS; }
    float *l = llama_get_logits_ith(session->ctx, -1);
    if (!l) { jl_set_error("llama_get_logits_ith returned NULL"); return JL_ERR_NO_LOGITS; }
    *out = l;
    *n_vocab = llama_vocab_n_tokens(session->model->vocab);
    return JL_OK;
}

jl_status jl_logits(jl_session *session, float *out, size_t cap, size_t *n_out) {
    jl_clear_error();
    if (!n_out) { jl_set_error("jl_logits: null n_out"); return JL_ERR_INVALID_ARG; }
    float *l = NULL; int32_t nv = 0;
    jl_status st = jl_logits_ptr(session, &l, &nv);
    if (st != JL_OK) return st;
    *n_out = (size_t) nv;
    if (!out) return JL_OK;
    if (cap < (size_t) nv) { jl_set_error("jl_logits: buffer holds %zu, need %d", cap, nv); return JL_ERR_BUFFER_TOO_SMALL; }
    memcpy(out, l, (size_t) nv * sizeof(float));
    return JL_OK;
}

/*
 * log-softmax denominator, max-subtracted. Computed over the whole vocabulary
 * so a top-k score means the same thing as a full-distribution score — mixing
 * the two conventions is exactly the kind of silent incomparability the work
 * order warns about.
 */
static float jl_log_sum_exp(const float *l, int32_t n, float *out_max) {
    float mx = l[0];
    for (int32_t i = 1; i < n; i++) if (l[i] > mx) mx = l[i];
    double sum = 0.0;
    for (int32_t i = 0; i < n; i++) sum += exp((double) (l[i] - mx));
    if (out_max) *out_max = mx;
    return (float) log(sum);
}

jl_status jl_logits_topk(jl_session *session, int32_t k,
                         int32_t *out_tokens, float *out_logprobs, size_t *n_out) {
    jl_clear_error();
    if (!n_out) { jl_set_error("jl_logits_topk: null n_out"); return JL_ERR_INVALID_ARG; }
    if (k <= 0) { jl_set_error("jl_logits_topk: k must be > 0"); return JL_ERR_INVALID_ARG; }
    float *l = NULL; int32_t nv = 0;
    jl_status st = jl_logits_ptr(session, &l, &nv);
    if (st != JL_OK) return st;

    if (k > nv) k = nv;
    *n_out = (size_t) k;
    if (!out_tokens || !out_logprobs) return JL_OK;   /* sizing call */

    float mx = 0.0f;
    const float lse = jl_log_sum_exp(l, nv, &mx);

    /*
     * Partial selection sort over k. k is small (top-100 at most in our tests)
     * and nv can be 250k, so this beats sorting the vocabulary; it also avoids
     * allocating an index array on the scoring hot path.
     */
    unsigned char *taken = (unsigned char *) calloc((size_t) nv, 1);
    if (!taken) { jl_set_error("out of memory"); return JL_ERR_ALLOC; }
    for (int32_t r = 0; r < k; r++) {
        int32_t best = -1;
        for (int32_t i = 0; i < nv; i++) {
            if (taken[i]) continue;
            if (best < 0 || l[i] > l[best]) best = i;
        }
        taken[best] = 1;
        out_tokens[r]   = best;
        out_logprobs[r] = (l[best] - mx) - lse;
    }
    free(taken);
    return JL_OK;
}

jl_status jl_token_logprob(jl_session *session, int32_t token, float *out) {
    jl_clear_error();
    if (!out) { jl_set_error("jl_token_logprob: null out"); return JL_ERR_INVALID_ARG; }
    float *l = NULL; int32_t nv = 0;
    jl_status st = jl_logits_ptr(session, &l, &nv);
    if (st != JL_OK) return st;
    if (token < 0 || token >= nv) { jl_set_error("jl_token_logprob: token %d out of range [0,%d)", token, nv); return JL_ERR_INVALID_ARG; }
    float mx = 0.0f;
    const float lse = jl_log_sum_exp(l, nv, &mx);
    *out = (l[token] - mx) - lse;
    return JL_OK;
}

/* ----------------------------------------------------------------- state */

jl_status jl_state_size(jl_session *session, int32_t seq_id, size_t *out) {
    jl_clear_error();
    if (!session || !session->ctx || !out) { jl_set_error("jl_state_size: null arg"); return JL_ERR_INVALID_ARG; }
    *out = llama_state_seq_get_size(session->ctx, (llama_seq_id) seq_id);
    return JL_OK;
}

jl_status jl_state_save(jl_session *session, int32_t seq_id,
                        uint8_t *buf, size_t cap, size_t *n_out) {
    jl_clear_error();
    if (!session || !session->ctx || !n_out) { jl_set_error("jl_state_save: null arg"); return JL_ERR_INVALID_ARG; }
    size_t need = llama_state_seq_get_size(session->ctx, (llama_seq_id) seq_id);
    *n_out = need;
    if (!buf) return JL_OK;                     /* sizing call */
    if (cap < need) { jl_set_error("jl_state_save: buffer holds %zu, need %zu", cap, need); return JL_ERR_BUFFER_TOO_SMALL; }
    size_t got = llama_state_seq_get_data(session->ctx, buf, cap, (llama_seq_id) seq_id);
    if (got == 0) { jl_set_error("llama_state_seq_get_data wrote 0 bytes"); return JL_ERR_STATE; }
    *n_out = got;
    return JL_OK;
}

jl_status jl_state_load(jl_session *session, int32_t seq_id,
                        const uint8_t *buf, size_t len,
                        int32_t n_tokens, size_t *n_read) {
    jl_clear_error();
    if (!session || !session->ctx || !buf || !n_read) { jl_set_error("jl_state_load: null arg"); return JL_ERR_INVALID_ARG; }
    if (seq_id != 0) {
        jl_set_error("jl_state_load: seq_id %d unsupported; v0 is single-sequence", seq_id);
        return JL_ERR_SEQ_UNSUPPORTED;
    }
    if (n_tokens < 0) {
        jl_set_error("jl_state_load: n_tokens %d is negative", n_tokens);
        return JL_ERR_INVALID_ARG;
    }
    size_t got = llama_state_seq_set_data(session->ctx, buf, len, (llama_seq_id) seq_id);
    if (got == 0) { jl_set_error("llama_state_seq_set_data rejected %zu bytes", len); return JL_ERR_STATE; }
    *n_read = got;
    /*
     * A restored sequence has no logits until something is evaluated on top of
     * it. Saying so explicitly stops a caller reading the previous occupant's
     * logits and believing they belong to the restored state.
     */
    session->have_logits = 0;
    /*
     * The ledger now describes the restored sequence, so the next jl_eval must
     * append at n_tokens. Taken from the caller because the blob does not carry
     * it; the Jolt layer has already checked the token vector this state was
     * saved against, so the number is not a guess.
     */
    session->n_resident = n_tokens;
    return JL_OK;
}
