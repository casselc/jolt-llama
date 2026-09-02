/*
 * C smoke test for the jolt_llama shim.
 *
 * Runs the whole vertical slice WITHOUT Jolt, so a failure here is
 * unambiguously the shim or libllama and never the FFI layer. When the Jolt
 * side later disagrees with this program on the same model, the difference is
 * the binding.
 *
 * It also prints a reference top-k that the Jolt test compares against, which
 * is what makes the M0 acceptance ("compared to an independent reference from
 * the same build") mean something.
 *
 *   ./smoke <model.gguf> [prompt]
 */
#include "jolt_llama.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail(const char *what, jl_status st) {
    char err[512];
    jl_last_error(err, sizeof(err));
    fprintf(stderr, "FAIL %s: status=%d err=%s\n", what, (int) st, err);
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model.gguf> [prompt]\n", argv[0]); return 2; }
    const char *path   = argv[1];
    const char *prompt = argc > 2 ? argv[2] : "The capital of France is";

    printf("abi_version=%d\n", jl_abi_version());

    jl_status st;
    if ((st = jl_runtime_init()) != JL_OK) return fail("runtime_init", st);

    jl_model_params mp;
    jl_model_params_default(&mp);
    jl_model *model = NULL;
    if ((st = jl_model_open(path, &mp, &model)) != JL_OK) return fail("model_open", st);

    char desc[256];
    jl_model_desc(model, desc, sizeof(desc));
    printf("model=%s\nn_vocab=%d n_ctx_train=%d\n",
           desc, jl_model_n_vocab(model), jl_model_n_ctx_train(model));

    jl_session_params sp;
    jl_session_params_default(&sp);
    sp.n_ctx = 4096;
    sp.n_threads = 4;
    sp.n_threads_batch = 4;
    jl_session *sess = NULL;
    if ((st = jl_session_new(model, &sp, &sess)) != JL_OK) return fail("session_new", st);

    /* two-call sizing, then the real tokenize */
    size_t n_tok = 0;
    if ((st = jl_tokenize(model, prompt, strlen(prompt), 1, 0, NULL, 0, &n_tok)) != JL_OK)
        return fail("tokenize(size)", st);
    int32_t *toks = (int32_t *) malloc(n_tok * sizeof(int32_t));
    if ((st = jl_tokenize(model, prompt, strlen(prompt), 1, 0, toks, n_tok, &n_tok)) != JL_OK)
        return fail("tokenize", st);

    printf("n_tokens=%zu tokens=", n_tok);
    for (size_t i = 0; i < n_tok && i < 16; i++) printf("%d ", toks[i]);
    printf("\n");

    if ((st = jl_eval(sess, 0, toks, n_tok, 0)) != JL_OK) return fail("eval", st);

    size_t n_logits = 0;
    if ((st = jl_logits(sess, NULL, 0, &n_logits)) != JL_OK) return fail("logits(size)", st);
    printf("n_logits=%zu\n", n_logits);

    /* the reference the Jolt test must reproduce */
    const int K = 10;
    int32_t tk[16]; float lp[16];
    size_t n_top = 0;
    if ((st = jl_logits_topk(sess, K, tk, lp, &n_top)) != JL_OK) return fail("logits_topk", st);

    /*
     * MACHINE ORACLE. Six-decimal output cannot prove bit identity -- two
     * different floats print the same -- so the comparable values are emitted
     * as raw IEEE-754 bits on ORACLE: lines that scripts/run-tests.sh diffs
     * against Jolt's. Anything a human reads stays below.
     */
    /* SHA-256 against the published FIPS test vectors, before anything trusts it */
    {
        char hx[65];
        if (jl_sha256_hex((const uint8_t *) "", 0, hx, sizeof(hx)) != JL_OK ||
            strcmp(hx, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")) {
            fprintf(stderr, "FAIL: sha256(\"\") = %s\n", hx); return 1; }
        if (jl_sha256_hex((const uint8_t *) "abc", 3, hx, sizeof(hx)) != JL_OK ||
            strcmp(hx, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")) {
            fprintf(stderr, "FAIL: sha256(\"abc\") = %s\n", hx); return 1; }
        if (jl_sha256_hex((const uint8_t *)
              "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq", 56, hx, sizeof(hx)) != JL_OK ||
            strcmp(hx, "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1")) {
            fprintf(stderr, "FAIL: sha256(56-byte vector) = %s\n", hx); return 1; }
        printf("sha256_vectors=ok\n");
    }

    printf("ORACLE abi=%d\n", (int) jl_abi_version());
    printf("ORACLE runtime=%s\n", jl_runtime_build_id());
    printf("ORACLE n_tokens=%zu\n", n_tok);
    printf("ORACLE tokens=");
    for (size_t i = 0; i < n_tok; i++) printf("%s%d", i ? "," : "", toks[i]);
    printf("\n");
    printf("ORACLE n_vocab=%d\n", (int) jl_model_n_vocab(model));
    for (size_t i = 0; i < n_top; i++) {
        uint32_t bits;
        memcpy(&bits, &lp[i], sizeof(bits));
        printf("ORACLE topk[%zu]=%d:%08x\n", i, tk[i], bits);
    }
    printf("topk:\n");
    for (size_t i = 0; i < n_top; i++) {
        char piece[64]; size_t np = 0;
        jl_token_to_piece(model, tk[i], piece, sizeof(piece), &np);
        piece[np < sizeof(piece) ? np : sizeof(piece) - 1] = '\0';
        printf("  %2zu  token=%-8d logprob=%.6f  piece=%s\n", i, tk[i], lp[i], piece);
    }

    /* state round trip, exercised here so a Jolt-side failure is isolable */
    size_t n_state = 0;
    if ((st = jl_state_size(sess, 0, &n_state)) != JL_OK) return fail("state_size", st);
    printf("ORACLE state_bytes=%zu\n", n_state);
    printf("state_bytes=%zu\n", n_state);

    uint8_t *blob = (uint8_t *) malloc(n_state);
    size_t written = 0;
    if ((st = jl_state_save(sess, 0, blob, n_state, &written)) != JL_OK) return fail("state_save", st);
    printf("state_saved=%zu\n", written);

    if ((st = jl_session_clear(sess, 0)) != JL_OK) return fail("session_clear", st);
    size_t nread = 0;
    if ((st = jl_state_load(sess, 0, blob, written, (int32_t) n_tok, &nread)) != JL_OK) return fail("state_load", st);
    printf("state_loaded=%zu\n", nread);

    /*
     * Negative path: logits must be refused immediately after a restore,
     * because nothing has been evaluated on top of the restored state yet.
     * Getting a stale buffer here is precisely the bug this guard prevents.
     */
    size_t dummy = 0;
    st = jl_logits(sess, NULL, 0, &dummy);
    printf("logits_after_restore_status=%d (expect %d JL_ERR_NO_LOGITS)\n", (int) st, (int) JL_ERR_NO_LOGITS);
    if (st != JL_ERR_NO_LOGITS) { fprintf(stderr, "FAIL: expected JL_ERR_NO_LOGITS\n"); return 1; }

    /* invalid-argument paths must be status codes, never crashes */
    if (jl_model_open(NULL, NULL, &model) != JL_ERR_INVALID_ARG) { fprintf(stderr, "FAIL: null path accepted\n"); return 1; }
    if (jl_eval(sess, 0, NULL, 0, 0)   != JL_ERR_INVALID_ARG) { fprintf(stderr, "FAIL: null tokens accepted\n"); return 1; }
    if (jl_eval(NULL, 0, toks, 1, 0)   != JL_ERR_INVALID_ARG) { fprintf(stderr, "FAIL: null session accepted\n"); return 1; }
    /*
     * OWNERSHIP. A jl_session holds its jl_model* and a llama_context built
     * from that model's llama_model, so closing the model underneath a live
     * session is a use-after-free the caller cannot detect. It must be refused,
     * and the model must still work afterwards.
     */
    if (jl_model_close(model) != JL_ERR_SESSIONS_ACTIVE) {
        fprintf(stderr, "FAIL: model closed with a live session\n"); return 1;
    }
    printf("model_close_with_live_session=refused\n");

    /* the refusal must be non-destructive: the model is still usable */
    if (jl_model_n_vocab(model) <= 0) {
        fprintf(stderr, "FAIL: refused close damaged the model\n"); return 1;
    }

    /* v0 is single-sequence */
    if (jl_eval(sess, 1, toks, 1, jl_session_n_resident(sess)) != JL_ERR_SEQ_UNSUPPORTED) {
        fprintf(stderr, "FAIL: nonzero seq_id accepted\n"); return 1;
    }
    if (jl_state_load(sess, 1, blob, written, (int32_t) n_tok, &nread) != JL_ERR_SEQ_UNSUPPORTED) {
        fprintf(stderr, "FAIL: state load into nonzero seq accepted\n"); return 1;
    }
    printf("runtime_build_id=%s\n", jl_runtime_build_id());
    /* every seq-carrying entry point refuses a nonzero sequence */
    {
        size_t sz = 0;
        if (jl_session_clear(sess, 1) != JL_ERR_SEQ_UNSUPPORTED) {
            fprintf(stderr, "FAIL: clear accepted seq 1\n"); return 1; }
        if (jl_session_clear(sess, -1) != JL_ERR_SEQ_UNSUPPORTED) {
            fprintf(stderr, "FAIL: clear accepted seq -1 (clear-all is not a v0 concept)\n"); return 1; }
        if (jl_state_size(sess, 1, &sz) != JL_ERR_SEQ_UNSUPPORTED) {
            fprintf(stderr, "FAIL: state_size accepted seq 1\n"); return 1; }
        if (jl_state_save(sess, 1, blob, n_state, &written) != JL_ERR_SEQ_UNSUPPORTED) {
            fprintf(stderr, "FAIL: state_save accepted seq 1\n"); return 1; }
    }
    printf("nonzero_seq=refused\n");

    /* n_seq_max must be EXACTLY 1; zero was silently clamped before */
    {
        jl_session_params sp;
        jl_session_params_default(&sp);
        jl_session *tmp = NULL;
        sp.n_seq_max = 0;
        if (jl_session_new(model, &sp, &tmp) != JL_ERR_SEQ_UNSUPPORTED) {
            fprintf(stderr, "FAIL: n_seq_max 0 accepted\n"); return 1; }
        sp.n_seq_max = 2;
        if (jl_session_new(model, &sp, &tmp) != JL_ERR_SEQ_UNSUPPORTED) {
            fprintf(stderr, "FAIL: n_seq_max 2 accepted\n"); return 1; }
        printf("seq_max_exactness=ok\n");
    }

    /* APPEND-ONLY: only the current end is a legal pos0 */
    {
        int32_t resident = jl_session_n_resident(sess);
        printf("n_resident=%d\n", resident);
        if (jl_eval(sess, 0, toks, 1, resident - 1) != JL_ERR_NOT_APPEND) {
            fprintf(stderr, "FAIL: eval into an earlier position accepted\n"); return 1;
        }
        if (jl_eval(sess, 0, toks, 1, resident + 5) != JL_ERR_NOT_APPEND) {
            fprintf(stderr, "FAIL: eval past the end accepted\n"); return 1;
        }
        if (jl_eval(sess, 0, toks, 1, resident) != JL_OK) {
            fprintf(stderr, "FAIL: append at the current end rejected\n"); return 1;
        }
        if (jl_session_n_resident(sess) != resident + 1) {
            fprintf(stderr, "FAIL: resident count did not advance\n"); return 1;
        }
        printf("append_only=ok\n");
    }

    printf("negative_paths=ok\n");

    free(blob); free(toks);
    if ((st = jl_session_close(sess))  != JL_OK) return fail("session_close", st);
    if ((st = jl_model_close(model))   != JL_OK) return fail("model_close", st);
    if ((st = jl_runtime_free())       != JL_OK) return fail("runtime_free", st);
    if (jl_runtime_free() != JL_ERR_INVALID_ARG) { fprintf(stderr, "FAIL: unbalanced runtime_free accepted\n"); return 1; }

    printf("SMOKE OK\n");
    return 0;
}
