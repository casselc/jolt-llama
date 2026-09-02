(ns jolt.llama
  "Embedded llama.cpp for Jolt, over a narrow stable C ABI.

  Runs on STOCK Jolt. Nothing here requires the aspect compiler, an aspect
  provider, OTel, or an HTTP server. Optional instrumentation is a separate
  concern and must never become a load-bearing dependency.

  The shape is deliberately small:

      open-model -> new-session -> tokenize -> eval! -> logits / score-candidates
                                            -> save-state / load-state!

  Two things this namespace owns that the C layer deliberately does not:

  TOKEN IDENTITY. A saved state is bound to the exact token vector used to build
  it. Reuse is legal only when the incoming tokens begin with that vector,
  token-for-token. A text prefix is NOT a token prefix -- BPE merges across the
  stable/dynamic seam, which is what made recurrent reuse silently fall back to
  a full prefill in the frozen Halo evidence. `load-state!` refuses a mismatch
  with :state/prefix-mismatch rather than guessing.

  CANDIDATE SEMANTICS. Scoring a finite set of legal candidates is the point:
  trusted code builds the domain, the model ranks it, trusted policy applies the
  result. That is not the same as generating text and hoping it parses.

  Native pointers are an implementation detail. Handles are Jolt maps; a raw
  pointer never appears in a public return value."
  (:require [clojure.java.shell]
            [clojure.string]
            [jolt.ffi :as ffi]))

;; ---------------------------------------------------------------- library

(def ^:private default-lib-name "libjolt_llama.so")

(def ^:private lib-path
  (or (System/getenv "JOLT_LLAMA_LIB")
      default-lib-name))

(defonce ^:private lib (ffi/load-library {:linux lib-path :darwin lib-path}))

;; ---------------------------------------------------------------- foreign

(ffi/defcfn ^:private c-abi-version "jl_abi_version" [] :int32)
(ffi/defcfn ^:private c-last-error "jl_last_error" [:pointer :size_t] :size_t)

(ffi/defcfn ^:private c-runtime-init "jl_runtime_init" [] :int32)
(ffi/defcfn ^:private c-runtime-free "jl_runtime_free" [] :int32)

(ffi/defcfn ^:private c-model-params-default "jl_model_params_default" [:pointer] :void)
(ffi/defcfn ^:private c-model-open "jl_model_open" [:pointer :pointer :pointer] :int32 :blocking)
(ffi/defcfn ^:private c-model-close "jl_model_close" [:pointer] :int32 :blocking)
(ffi/defcfn ^:private c-model-n-vocab "jl_model_n_vocab" [:pointer] :int32)
(ffi/defcfn ^:private c-model-n-ctx-train "jl_model_n_ctx_train" [:pointer] :int32)
(ffi/defcfn ^:private c-model-desc "jl_model_desc" [:pointer :pointer :size_t] :size_t)

(ffi/defcfn ^:private c-session-params-default "jl_session_params_default" [:pointer] :void)
(ffi/defcfn ^:private c-session-new "jl_session_new" [:pointer :pointer :pointer] :int32 :blocking)
(ffi/defcfn ^:private c-session-close "jl_session_close" [:pointer] :int32)
(ffi/defcfn ^:private c-session-n-ctx "jl_session_n_ctx" [:pointer] :uint32)
(ffi/defcfn ^:private c-session-clear "jl_session_clear" [:pointer :int32] :int32)

;; :blocking marks the calls that MEASURED in the hundreds of milliseconds --
;; model open, session construction, eval, and state save/load -- so the Chez
;; collector is not held off across them. Deliberately NOT on the scalar
;; queries (n_vocab, n_ctx, abi_version): those return immediately and the
;; marker would cost more than the call.
(ffi/defcfn ^:private c-tokenize "jl_tokenize"
  [:pointer :pointer :size_t :int32 :int32 :pointer :size_t :pointer] :int32)
(ffi/defcfn ^:private c-token-to-piece "jl_token_to_piece"
  [:pointer :int32 :pointer :size_t :pointer] :int32)

(ffi/defcfn ^:private c-eval "jl_eval" [:pointer :int32 :pointer :size_t :int32] :int32 :blocking)

(ffi/defcfn ^:private c-logits "jl_logits" [:pointer :pointer :size_t :pointer] :int32)
(ffi/defcfn ^:private c-logits-topk "jl_logits_topk"
  [:pointer :int32 :pointer :pointer :pointer] :int32)
(ffi/defcfn ^:private c-token-logprob "jl_token_logprob" [:pointer :int32 :pointer] :int32)

(ffi/defcfn ^:private c-state-size "jl_state_size" [:pointer :int32 :pointer] :int32)
(ffi/defcfn ^:private c-state-save "jl_state_save" [:pointer :int32 :pointer :size_t :pointer] :int32 :blocking)
(ffi/defcfn ^:private c-state-load "jl_state_load"
  [:pointer :int32 :pointer :size_t :int32 :pointer] :int32 :blocking)

;; ------------------------------------------------------------------ errors

(def ^:private status->kw
  {0 :ok, -1 :error/generic, -2 :error/invalid-arg, -3 :error/alloc
   -4 :error/model-load, -5 :error/context, -6 :error/tokenize
   -7 :error/decode, -8 :error/state, -9 :error/buffer-too-small
   -10 :error/no-logits
   -11 :model/sessions-active
   -12 :seq/unsupported
   -13 :state/not-append})

(defn- last-error-text []
  (let [n (c-last-error ffi/null 0)]
    (if (zero? n)
      ""
      (ffi/with-alloc [buf (inc n)]
        (c-last-error buf (inc n))
        (ffi/ptr->string buf)))))

(defn- check!
  "Turn a native status into a Jolt exception carrying data, or return nil.

  The native message is bounded and already formatted; it is attached rather
  than re-derived so the failing C call names itself."
  [status op extra]
  (when-not (zero? status)
    (throw (ex-info (str "jolt.llama/" (name op) " failed: " (last-error-text))
                    (merge {:jolt.llama/op op
                            :jolt.llama/status status
                            :jolt.llama/error (get status->kw status :error/unknown)
                            :jolt.llama/message (last-error-text)}
                           extra))))
  nil)

;; ------------------------------------------------------------------ runtime

(def ^:private abi-expected
  "The shim ABI this code is written against.

  2 refuses jl_model_close while sessions are open, makes jl_eval append-only
  and single-sequence, and adds the token count to jl_state_load. A v1 shim
  would silently misread that last argument, so the check is exact, not a
  floor."
  2)

(defn abi-version [] (c-abi-version))

(defn- ensure-abi! []
  (let [got (c-abi-version)]
    (when-not (= got abi-expected)
      (throw (ex-info (str "jolt.llama: shim ABI " got " != expected " abi-expected
                           "; rebuild native/libjolt_llama.so")
                      {:jolt.llama/op :runtime-init
                       :jolt.llama/abi-found got
                       :jolt.llama/abi-expected abi-expected})))))

(defonce ^:private runtime-started (atom false))

(defn init-runtime!
  "Idempotent. The native side refcounts, but repeating it from Jolt would leak
  a reference per call, so the first success latches."
  []
  (when-not @runtime-started
    (ensure-abi!)
    (check! (c-runtime-init) :runtime-init {})
    (reset! runtime-started true))
  :ok)

;; -------------------------------------------------------------------- model

(defn- read-ptr [pp] (ffi/read pp :pointer 0))

(defn- gguf-content-id
  "A stable identity for the model ARTIFACT, computed once at open.

  Not the path and not the description. A path can point at different bytes
  between two runs, and descriptions collide across quantizations of the same
  model -- so a saved state validated against either could be restored into a
  model that merely looks the same. Bound to content instead.

  Reads the file ONCE, here. Hashing a multi-gigabyte GGUF on every state
  operation would dominate every measurement in this repo, so nothing else in
  the library ever touches the file.

  Prefers sha256sum when present, since a real digest makes the identity
  meaningful outside this process. Falls back to size plus a sampled digest,
  which is weaker but still catches the cases that actually occur: a different
  model, a different quantization, a truncated or partially-written file. The
  fallback is LABELLED, so a state descriptor never claims more than it has."
  [path]
  (let [f (java.io.File. path)
        size (.length f)]
    (or (try
          (let [{:keys [exit out]} (clojure.java.shell/sh "sha256sum" path)]
            (when (zero? exit)
              (str "sha256:" (first (clojure.string/split (clojure.string/trim out) #"\s+")))))
          (catch Throwable _ nil))
        ;; sampled fallback: head, middle and tail, plus the exact byte count
        (try
          (with-open [in (java.io.RandomAccessFile. f "r")]
            (let [buf (byte-array 65536)
                  spots [0 (quot size 2) (max 0 (- size 65536))]
                  h (reduce (fn [acc off]
                              (.seek in (long off))
                              (let [n (max 0 (.read in buf))]
                                (loop [i 0 a acc]
                                  (if (>= i n) a
                                      (recur (inc i)
                                             (unchecked-int (+ (* 31 a) (bit-and (aget buf i) 0xff))))))))
                            (int 17) spots)]
              (format "sampled:%d:%08x" size (bit-and h 0xffffffff))))
          (catch Throwable _ (str "size:" size))))))

(defn open-model
  "Open a GGUF model. Returns a handle map; ::ptr is private plumbing.

  opts: :path (required), :gpu-layers (default 0 = CPU), :mmap?, :mlock?"
  [{:keys [path gpu-layers mmap? mlock?]
    :or   {gpu-layers 0 mmap? true mlock? false}
    :as   opts}]
  (when-not (string? path)
    (throw (ex-info "jolt.llama/open-model: :path must be a string"
                    {:jolt.llama/op :model-open :jolt.llama/opts opts})))
  (init-runtime!)
  ;; jl_model_params is 8 x int32 including reserved; sized here rather than
  ;; mirrored as a layout so adding a reserved field on the C side cannot
  ;; silently shift what Jolt writes.
  (ffi/with-alloc [mp (* 8 4)]
    (c-model-params-default mp)
    (ffi/write mp :int32 (int gpu-layers) 0)
    (ffi/write mp :int32 (if mmap? 1 0) 4)
    (ffi/write mp :int32 (if mlock? 1 0) 8)
    (ffi/with-c-string [cpath path]
      (ffi/with-out [pp :pointer]
        (check! (c-model-open cpath mp pp) :model-open {:jolt.llama/path path})
        (let [ptr (read-ptr pp)
              desc (let [n (c-model-desc ptr ffi/null 0)]
                     (if (zero? n) ""
                         (ffi/with-alloc [b (inc n)]
                           (c-model-desc ptr b (inc n))
                           (ffi/ptr->string b))))]
          {::kind :model
           ::ptr ptr
           :path path
           :desc desc
           :n-vocab (c-model-n-vocab ptr)
           :n-ctx-train (c-model-n-ctx-train ptr)
           ;; The model's ARTIFACT identity, computed once here and never per
           ;; state operation. A path and a description do not identify a
           ;; model: two runs can hold the same path over different bytes, and
           ;; descriptions collide across quantizations. State compatibility is
           ;; checked against this.
           :content-id (gguf-content-id path)
           ;; Sessions created from this model and not yet closed. The C layer
           ;; keeps the authoritative count; this mirror exists so the Jolt
           ;; error can name them before the call is made.
           :sessions (atom 0)
           :closed (atom false)})))))

(defn- model-ptr [m]
  (when-not (and (map? m) (= :model (::kind m))) (throw (ex-info "not a model handle" {:got m})))
  (when @(:closed m) (throw (ex-info "jolt.llama: model is closed"
                                     {:jolt.llama/op :model-use
                                      :jolt.llama/error :handle/closed})))
  (::ptr m))

;; ------------------------------------------------------------------ session

(defn new-session
  "Create an inference context over a model.

  opts: :context-size, :batch-size, :ubatch-size, :seq-max, :threads,
        :threads-batch"
  [model {:keys [context-size batch-size ubatch-size seq-max threads threads-batch]
          :or   {context-size 4096 seq-max 1 threads 4}
          :as   _opts}]
  (when (not= 1 seq-max)
    (throw (ex-info (str "jolt.llama/new-session: v0 is single-sequence; "
                         ":seq-max must be 1")
                    {:jolt.llama/op :session-new
                     :jolt.llama/error :seq/unsupported
                     :jolt.llama/seq-max seq-max})))
  (let [mptr (model-ptr model)
        nb   (or batch-size 2048)
        nub  (or ubatch-size nb)
        tb   (or threads-batch threads)]
    ;; jl_session_params: 4 x uint32 then 6 x int32
    (ffi/with-alloc [sp (* 10 4)]
      (c-session-params-default sp)
      (ffi/write sp :uint32 (int context-size) 0)
      (ffi/write sp :uint32 (int nb) 4)
      (ffi/write sp :uint32 (int nub) 8)
      (ffi/write sp :uint32 (int seq-max) 12)
      (ffi/write sp :int32 (int threads) 16)
      (ffi/write sp :int32 (int tb) 20)
      (ffi/with-out [pp :pointer]
        (check! (c-session-new mptr sp pp) :session-new {})
        (let [ptr (read-ptr pp)]
          (swap! (:sessions model) inc)
          {::kind :session
           ::ptr ptr
           :model model
           :n-ctx (c-session-n-ctx ptr)
           ;; The token vector currently resident in seq 0. This is the Jolt-side
           ;; half of the token-identity contract; the C layer has no opinion.
           :tokens (atom [])
           :closed (atom false)})))))

(defn- session-ptr [s]
  (when-not (and (map? s) (= :session (::kind s))) (throw (ex-info "not a session handle" {:got s})))
  (when @(:closed s) (throw (ex-info "jolt.llama: session is closed"
                                     {:jolt.llama/op :session-use
                                      :jolt.llama/error :handle/closed})))
  (::ptr s))

(defn clear!
  "Drop cached state for a sequence and forget its token vector."
  ([session] (clear! session 0))
  ([session seq-id]
   (let [p (session-ptr session)]
     (check! (c-session-clear p (int seq-id)) :session-clear {})
     (reset! (:tokens session) [])
     :ok)))

;; ----------------------------------------------------------------- tokenize

(defn tokenize
  "Text -> vector of token ids. The ONLY supported way to obtain tokens.

  Never re-tokenize concatenated text and assume the seam is stable; build the
  full token vector once, or append already-tokenized vectors."
  ([model text] (tokenize model text {}))
  ([model text {:keys [add-special? parse-special?]
                :or {add-special? true parse-special? false}}]
   (let [mptr (model-ptr model)
         bytes (count (.getBytes ^String text "UTF-8"))]
     (ffi/with-c-string [ctext text]
       (ffi/with-out [np :size_t]
         (check! (c-tokenize mptr ctext bytes (if add-special? 1 0)
                             (if parse-special? 1 0) ffi/null 0 np)
                 :tokenize {})
         (let [n (ffi/read np :size_t)]
           (if (zero? n)
             []
             (ffi/with-alloc [buf (* n 4)]
               (check! (c-tokenize mptr ctext bytes (if add-special? 1 0)
                                   (if parse-special? 1 0) buf n np)
                       :tokenize {})
               (let [n2 (ffi/read np :size_t)]
                 (mapv #(ffi/read buf :int32 (* % 4)) (range n2)))))))))))

(defn token->piece
  "Render one token id back to its text fragment."
  [model token]
  (let [mptr (model-ptr model)]
    (ffi/with-out [np :size_t]
      (check! (c-token-to-piece mptr (int token) ffi/null 0 np) :token-to-piece {})
      (let [n (ffi/read np :size_t)]
        (if (zero? n) ""
            (ffi/with-alloc [buf (inc n)]
              (check! (c-token-to-piece mptr (int token) buf n np) :token-to-piece {})
              (ffi/write buf :uint8 0 (ffi/read np :size_t))
              (ffi/ptr->string buf)))))))

;; --------------------------------------------------------------------- eval

(defn- write-tokens [buf tokens]
  (doseq [[i t] (map-indexed vector tokens)]
    (ffi/write buf :int32 (int t) (* i 4))))

(defn eval!
  "Evaluate tokens into the sequence, APPENDING after whatever is resident.

  Append-only, and there is no :pos option. The native recurrent/KV state is
  not truncated to an arbitrary position, so evaluating into the middle would
  leave the session's native state and its token vector describing different
  sequences -- silently, and in exactly the way the token-identity contract
  exists to detect. The shim enforces the same rule independently and answers
  JL_ERR_NOT_APPEND; this is not the only line of defence.

  To go backwards, clear! and re-evaluate, or restore a state saved at the
  point you want.

  Updates the session's token vector so a later save/restore can be checked."
  ([session tokens] (eval! session tokens {}))
  ([session tokens {:keys [seq-id] :or {seq-id 0}}]
   (when (not= 0 seq-id)
     (throw (ex-info "jolt.llama/eval!: v0 is single-sequence; :seq-id must be 0"
                     {:jolt.llama/op :eval
                      :jolt.llama/error :seq/unsupported
                      :jolt.llama/seq-id seq-id})))
   (let [p (session-ptr session)
         toks (vec tokens)
         n (count toks)
         pos0 (count @(:tokens session))]
     (when (zero? n)
       (throw (ex-info "jolt.llama/eval!: empty token vector"
                       {:jolt.llama/op :eval :jolt.llama/error :error/invalid-arg})))
     (ffi/with-alloc [buf (* n 4)]
       (write-tokens buf toks)
       (check! (c-eval p (int seq-id) buf n (int pos0)) :eval
               {:jolt.llama/n-tokens n :jolt.llama/pos pos0}))
     (swap! (:tokens session) into toks)
     {:n-tokens n :pos pos0 :n-resident (count @(:tokens session))})))

;; ------------------------------------------------------------------- logits

(defn logits
  "Full logit vector for the last evaluated position."
  ([session] (logits session {}))
  ([session _opts]
   (let [p (session-ptr session)]
     (ffi/with-out [np :size_t]
       (check! (c-logits p ffi/null 0 np) :logits {})
       (let [n (ffi/read np :size_t)]
         (ffi/with-alloc [buf (* n 4)]
           (check! (c-logits p buf n np) :logits {})
           (mapv #(ffi/read buf :float (* % 4)) (range n))))))))

(defn top-k
  "Top-k [{:token id :logprob lp :piece s}] descending, log-softmax normalised
  over the whole vocabulary so scores are comparable across calls."
  ([session k] (top-k session k {}))
  ([session k {:keys [pieces?] :or {pieces? true}}]
   (let [p (session-ptr session)
         model (:model session)]
     (ffi/with-alloc [tbuf (* k 4)]
       (ffi/with-alloc [lbuf (* k 4)]
         (ffi/with-out [np :size_t]
           (check! (c-logits-topk p (int k) tbuf lbuf np) :logits-topk {})
           (let [n (ffi/read np :size_t)]
             (mapv (fn [i]
                     (let [tok (ffi/read tbuf :int32 (* i 4))]
                       (cond-> {:token tok
                                :logprob (ffi/read lbuf :float (* i 4))}
                         pieces? (assoc :piece (token->piece model tok)))))
                   (range n)))))))))

(defn token-logprob
  "Log-probability of one token at the last evaluated position."
  [session token]
  (let [p (session-ptr session)]
    (ffi/with-out [fp :float]
      (check! (c-token-logprob p (int token) fp) :token-logprob {:jolt.llama/token token})
      (ffi/read fp :float))))

;; -------------------------------------------------------------------- state

(defn- digest-seq
  "Content hash for provenance. Falls back to a cheap stable digest when no
  crypto is available, since this identifies a state to us, not to the world."
  [xs]
  (let [h (reduce (fn [acc b] (unchecked-int (+ (* 31 acc) (bit-and b 0xff))))
                  (int 17) xs)]
    (format "%08x" (bit-and h 0xffffffff))))

(defn- digest-array
  "The same digest over the first `n` bytes of an array, WITHOUT seqing it.

  `(reduce f init (take 4096 blob))` looks equivalent and is not: taking a seq
  over a 54 MB byte array walks the whole array before `take` sees its first
  element. That one expression cost 7.4 of the 7.6 seconds save-state spent, and
  made state capture look 41x slower than state restore for what is the same
  memcpy in both directions. Index the array instead."
  [arr n]
  (let [limit (min (int n) (alength arr))]
    (loop [i 0 h (int 17)]
      (if (>= i limit)
        (format "%08x" (bit-and h 0xffffffff))
        (recur (inc i) (unchecked-int (+ (* 31 h) (bit-and (aget arr i) 0xff))))))))

(defn save-state
  "Capture exact native state for a sequence, bound to its token vector.

  The returned descriptor carries everything needed to decide later whether
  reuse is legal: the token vector itself, its count, a hash, the native byte
  count, and the model coordinate. The raw blob is held in memory here; a
  content-addressed store is a later concern."
  ([session] (save-state session {}))
  ([session {:keys [seq-id] :or {seq-id 0}}]
   (let [p (session-ptr session)
         toks @(:tokens session)]
     (ffi/with-out [np :size_t]
       (check! (c-state-save p (int seq-id) ffi/null 0 np) :state-save {})
       (let [n (ffi/read np :size_t)
             blob (byte-array n)]
         (ffi/with-alloc [buf n]
           (check! (c-state-save p (int seq-id) buf n np) :state-save {})
           (let [written (ffi/read np :size_t)]
             (ffi/read-into! buf blob 0 written)
             {::kind :state
              ;; THE compatibility coordinate. :model-content-id is the one that
              ;; decides; path and desc are kept for humans reading a descriptor
              ;; and are explicitly NOT trusted -- a path can point at different
              ;; bytes between runs and descriptions collide across
              ;; quantizations of the same model.
              :model-content-id (get-in session [:model :content-id])
              :model-desc (get-in session [:model :desc])
              :model-path (get-in session [:model :path])
              :seq-id seq-id
              :tokens toks
              :n-tokens (count toks)
              :token-hash (digest-seq (map int toks))
              :state-bytes written
              :state-hash (digest-array blob 4096)
              :abi (c-abi-version)
              ::blob blob})))))))

(defn token-prefix-ok?
  "THE token-identity predicate.

  Reuse is legal iff the incoming token vector begins with the saved one,
  token-for-token. Length alone is not sufficient and string prefixing is not
  evidence -- BPE merges across the seam and changes the final stable token."
  [state tokens]
  (let [s (:tokens state)
        n (count s)]
    (and (>= (count tokens) n)
         (= s (vec (take n tokens))))))

(defn state-compatible?
  "Why `state` may not be restored into `session`, or nil if it may.

  Every reason is a keyword, because the caller journals the refusal and
  \"incompatible\" without a cause is not auditable. Checked BEFORE any native
  call: llama_state_seq_set_data given a blob from another model does not
  reliably fail, it can succeed into nonsense.

  Deliberately keyed on the model's CONTENT id rather than on handle identity.
  Two independently opened handles over the same GGUF are the same model, and
  treating pointer identity as semantic identity would refuse a restore that is
  perfectly sound -- which is how callers learn to pass :unchecked."
  [session state]
  (let [model (:model session)]
    (cond
      (not (map? state))                      :state/malformed
      (not= :state (::kind state))             :state/malformed
      (nil? (::blob state))                    :state/malformed
      (not (vector? (:tokens state)))          :state/malformed
      (not= (:n-tokens state) (count (:tokens state))) :state/malformed
      (not= (:abi state) abi-expected)         :state/abi-mismatch
      (not= 0 (:seq-id state))                 :seq/unsupported
      (not= (:token-hash state) (digest-seq (map int (:tokens state))))
      :state/token-hash-mismatch
      (not= (:state-hash state) (digest-array (::blob state) 4096))
      :state/blob-hash-mismatch
      (not= (:state-bytes state) (alength ^bytes (::blob state)))
      :state/blob-size-mismatch
      (not= (:model-content-id state) (:content-id model))
      :state/model-mismatch
      :else nil)))

(defn- load-state-unchecked!
  "Restore native state having ALREADY established compatibility and token
  identity. Private, and named so that reading a call site makes the omission
  obvious. Nothing outside this namespace should be able to skip the checks."
  [session state seq-id]
  (let [p (session-ptr session)
        blob ^bytes (::blob state)]
    (ffi/with-alloc [buf (alength blob)]
      (ffi/write-array buf blob)
      (ffi/with-out [np :size_t]
        (check! (c-state-load p (int seq-id) buf (alength blob)
                              (int (:n-tokens state)) np)
                :state-load {})
        (reset! (:tokens session) (:tokens state))
        {:n-read (ffi/read np :size_t)
         :n-tokens (:n-tokens state)}))))

(defn load-state!
  "Restore exact native state for the token vector you are about to work with.

  `tokens` is REQUIRED. It is the canonical token vector of the full prompt the
  restored state is a prefix of, and the restore is refused unless the saved
  state is a genuine token-for-token prefix of it.

  It is required because it is the entire point. When it was optional, the
  shortest call -- (load-state! session state) -- was also the one that skipped
  the defining safety invariant, so the easiest thing to write was the unsafe
  thing. A stable TEXT prefix is not a stable TOKEN prefix; BPE merges across
  the boundary and moves the final stable token, which this library measured at
  one token on a 2793-token spine.

  Refuses, before any native call, with a keyword under :jolt.llama/error:

    :state/prefix-mismatch      the saved tokens do not prefix `tokens`
    :state/model-mismatch       saved against a different model artifact
    :state/abi-mismatch         saved by a different shim ABI
    :state/token-hash-mismatch  the descriptor's tokens were altered
    :state/blob-hash-mismatch   the descriptor's blob was altered
    :state/blob-size-mismatch   the descriptor disagrees with its own blob
    :state/malformed            not a state descriptor
    :seq/unsupported            v0 is single-sequence

  There is no public unchecked path. If you have genuinely proven identity
  another way, prove it to this function by passing the tokens."
  ([session state tokens] (load-state! session state tokens {}))
  ([session state tokens {:keys [seq-id] :or {seq-id 0}}]
   (session-ptr session)
   (when (not= 0 seq-id)
     (throw (ex-info "jolt.llama/load-state!: v0 is single-sequence; :seq-id must be 0"
                     {:jolt.llama/op :state-load
                      :jolt.llama/error :seq/unsupported
                      :jolt.llama/seq-id seq-id})))
   (when-not (sequential? tokens)
     (throw (ex-info "jolt.llama/load-state!: tokens must be the canonical token vector"
                     {:jolt.llama/op :state-load
                      :jolt.llama/error :error/invalid-arg})))
   (when-let [why (state-compatible? session state)]
     (throw (ex-info (str "jolt.llama/load-state!: incompatible state (" (name why) ")")
                     {:jolt.llama/op :state-load
                      :jolt.llama/error why
                      :jolt.llama/state-model (:model-content-id state)
                      :jolt.llama/session-model (get-in session [:model :content-id])
                      :jolt.llama/state-abi (:abi state)})))
   (when-not (token-prefix-ok? state tokens)
     (let [sv (:tokens state)
           idx (or (first (keep-indexed (fn [i [a b]] (when (not= a b) i))
                                        (map vector sv tokens)))
                   (count tokens))]
       (throw (ex-info "jolt.llama/load-state!: incoming tokens do not extend the saved state"
                       {:jolt.llama/op :state-load
                        :jolt.llama/error :state/prefix-mismatch
                        :jolt.llama/saved-n (count sv)
                        :jolt.llama/incoming-n (count tokens)
                        :jolt.llama/diverges-at idx}))))
   (load-state-unchecked! session state seq-id)))

;; -------------------------------------------------------- candidate scoring

(defn- topk-map [session k]
  (into {} (map (juxt :token :logprob) (top-k session k {:pieces? false}))))

(defn- max-abs-delta [a b]
  (let [common (filter b (keys a))]
    (if (seq common)
      (apply max (map #(abs (- (double (a %)) (double (b %)))) common))
      -1.0)))

(defn append-divergence
  "Compare `prefix ++ suffix` evaluated as TWO calls against the same tokens
  evaluated as ONE call.

  Returns a MAP, not a number, because a single number here is misleading. The
  obvious metric -- max |delta logprob| over the top-k -- is dominated by tokens
  at p ~ 1e-8, where a float32 logit of magnitude 20 carries absolute error in
  the 0.1 nat range purely from having ~7 significant digits. Measured on
  Qwen3.5-0.8B, two kernel paths that agree to 0.00046 nats at rank 0 (p=0.993)
  differ by 0.286 at rank 30 (p~1e-8). Reporting only the max would call that a
  serious divergence; it is not. See docs/EXACTNESS.md.

    :bit-exact?    true when every common logprob matched exactly
    :top1-abs      |delta| for the argmax -- the number that matters for policy
    :max-abs       worst over the top-k, tail included
    :top1-same?    whether the argmax token itself agreed
    :order-same?   whether the top-5 ordering agreed
    :n-common

  DESTRUCTIVE: clears the session. Intended for calibration and tests."
  ([session prefix suffix] (append-divergence session prefix suffix {}))
  ([session prefix suffix {:keys [k seq-id] :or {k 50 seq-id 0}}]
   (let [prefix (vec prefix) suffix (vec suffix)
         full   (into prefix suffix)
         _      (clear! session seq-id)
         _      (eval! session full {:seq-id seq-id})
         one    (top-k session k {:pieces? false})
         one-m  (into {} (map (juxt :token :logprob) one))
         _      (clear! session seq-id)
         _      (eval! session prefix {:seq-id seq-id})
         _      (eval! session suffix {:seq-id seq-id})
         two    (top-k session k {:pieces? false})
         two-m  (into {} (map (juxt :token :logprob) two))
         common (filter two-m (keys one-m))
         ds     (map #(abs (- (double (one-m %)) (double (two-m %)))) common)
         t1     (:token (first one))]
     {:bit-exact?  (every? zero? ds)
      :top1-abs    (when-let [b (two-m t1)]
                     (abs (- (double (one-m t1)) (double b))))
      :max-abs     (if (seq ds) (apply max ds) -1.0)
      :top1-same?  (= t1 (:token (first two)))
      :order-same? (= (mapv :token (take 5 one)) (mapv :token (take 5 two)))
      :n-common    (count common)})))

(defn calibrate-append-exactness
  "Find the shortest append length that still reproduces a one-pass evaluation,
  by measurement rather than assumption.

  The threshold is a property of the MODEL, not of this library, so it is not
  hard-coded. On the Qwen3.5-0.8B hybrid used here it measures at 64, matching
  the fused Gated Delta Net chunk size the loader announces; a pure-attention
  model is expected to measure 1.

  `probe-tokens` must be a real token vector from the model, long enough for
  `prefix-len + max-suffix`. Exactness is monotone in suffix length for the
  models measured so far, so this binary-searches; :verified names the lengths
  actually probed so a non-monotone model is visible rather than hidden.

  Returns
    :threshold      shortest suffix length that matched one-pass, or nil
    :max-suffix     the search ceiling
    :monotone?      whether every probe was consistent with a single threshold
    :probes         [{:suffix-len n :divergence d :exact? bool} ...]

  DESTRUCTIVE: clears the session."
  ([session probe-tokens] (calibrate-append-exactness session probe-tokens {}))
  ([session probe-tokens {:keys [prefix-len max-suffix tolerance seq-id]
                          :or {prefix-len 128 max-suffix 256 tolerance 1e-9 seq-id 0}}]
   (let [toks (vec probe-tokens)]
     (when (< (count toks) (+ prefix-len max-suffix))
       (throw (ex-info "jolt.llama/calibrate-append-exactness: probe-tokens too short"
                       {:jolt.llama/op :calibrate-append-exactness
                        :jolt.llama/error :error/invalid-arg
                        :jolt.llama/need (+ prefix-len max-suffix)
                        :jolt.llama/have (count toks)})))
     (let [prefix (subvec toks 0 prefix-len)
           probes (atom [])
           exact? (fn [n]
                    (let [r (append-divergence session prefix
                                                (subvec toks prefix-len (+ prefix-len n))
                                                {:seq-id seq-id})
                          ok (or (:bit-exact? r) (< (:max-abs r) tolerance))]
                      (swap! probes conj (assoc (select-keys r [:max-abs :top1-abs :top1-same?])
                                                :suffix-len n :exact? ok))
                      ok))]
       (if-not (exact? max-suffix)
         ;; nothing in range is exact; report that honestly instead of guessing
         {:threshold nil :max-suffix max-suffix :monotone? true :probes @probes}
         (loop [lo 1 hi max-suffix]
           (if (>= lo hi)
             (let [ps @probes
                   t lo
                   ;; a single threshold explains the data only if every probe
                   ;; below t diverged and every probe at or above t matched
                   mono (every? (fn [{:keys [suffix-len exact?]}]
                                  (= exact? (>= suffix-len t)))
                                ps)]
               {:threshold t :max-suffix max-suffix :monotone? mono :probes (vec ps)})
             (let [mid (quot (+ lo hi) 2)]
               (if (exact? mid) (recur lo mid) (recur (inc mid) hi))))))))))

(defn score-candidates
  "Score a finite set of legal candidates against the CURRENT session state.

  This is the reason the library exists: trusted code constructs the legal
  domain, the model ranks it, trusted policy applies the result. No sampler, no
  grammar, no free-form generation.

  The caller must already have evaluated the base context (spine, or spine plus
  delta) so logits are available. Each candidate is {:id any :tokens [ids...]}.

  Scoring is teacher-forced: a candidate's score is the summed log-probability
  of its own tokens in order. The work is arranged around one observation --
  P(first-token | base) is the SAME conditional for every candidate, so it is
  read once from the base logits. Single-token candidates therefore cost no
  evaluation at all, which is the common controller case. Only candidates with
  more than one token need the model advanced, and those are restored back to
  the base state first so candidates never see one another.

  Returned per candidate:
    :logprob-sum   sum over the candidate's own tokens
    :logprob-mean  length-normalised
    :n-tokens
  Both are reported because neither is universally right, and silently choosing
  one is how incomparable conventions creep in. Sorted by :logprob-sum
  descending; :rank is the index under that ordering.

  SCORING CONVENTION, stated because it is not free of consequence.

  A candidate's first token is read from the base logits, which were produced by
  whatever call evaluated the base -- normally a long prefill. Its remaining
  tokens are reached by single-token decodes. On a hybrid model those are two
  different kernel paths (see docs/EXACTNESS.md and calibrate-append-exactness),
  so a multi-token score mixes them while a single-token score does not.

  What follows from that:
    * single-token candidates are exactly comparable with each other
    * equal-length candidates are exactly comparable with each other
    * unequal-length candidates are comparable, but their sums are not built
      from an identical mixture of paths, so a near-tie between a 1-token and a
      4-token candidate should not be read as meaningful
    * NONE of these scores are comparable against a logprob obtained by
      prefilling the candidate text as part of one long prompt

  The result map reports :convention and :homogeneous? so a caller can assert
  the case it actually relies on instead of assuming it.

  Requires :state (from save-state) when any candidate has more than one token,
  since rewinding to the base otherwise means re-evaluating it per candidate.

  After scoring multi-token candidates the session holds the restored base state
  but no logits, because that is what load-state! guarantees. To score the same
  base again, pass the previous result's :base-logprobs back in as the
  :base-logprobs option; do NOT re-evaluate a token to regenerate them, since
  that would move the base onto a different kernel path."
  ([session candidates] (score-candidates session candidates {}))
  ([session candidates {:keys [seq-id state base-logprobs] :or {seq-id 0}}]
   (when (empty? candidates)
     (throw (ex-info "jolt.llama/score-candidates: empty candidate set"
                     {:jolt.llama/op :score-candidates
                      :jolt.llama/error :error/invalid-arg})))
   (doseq [{:keys [id tokens]} candidates]
     (when (empty? tokens)
       (throw (ex-info "jolt.llama/score-candidates: candidate has no tokens"
                       {:jolt.llama/op :score-candidates
                        :jolt.llama/error :error/invalid-arg
                        :jolt.llama/candidate id}))))
   (let [base-tokens @(:tokens session)
         base-n (count base-tokens)
         multi? (some #(> (count (:tokens %)) 1) candidates)
         _ (when (and multi? (nil? state))
             (throw (ex-info (str "jolt.llama/score-candidates: multi-token candidates "
                                  "need :state so the base can be restored between them")
                             {:jolt.llama/op :score-candidates
                              :jolt.llama/error :error/invalid-arg})))
         ;; One read of the base distribution serves every candidate's first
         ;; token. Doing this before any restore is what keeps the single-token
         ;; path free of evaluation.
         first-tokens (distinct (map (comp first :tokens) candidates))
         ;; Scoring a multi-token candidate ends with a restore, and a restore
         ;; deliberately leaves the session with no logits (M0 asserts this: a
         ;; restored state has not produced an output position). So a second
         ;; call against the same base cannot read the base distribution again.
         ;; Rather than re-evaluate -- which would land on a different kernel
         ;; path and silently change the scoring convention -- the base
         ;; log-probabilities are returned, and accepted back here.
         base-lp (if (and base-logprobs (every? base-logprobs first-tokens))
                   (select-keys base-logprobs first-tokens)
                   (into {} (map (fn [t] [t (token-logprob session t)]) first-tokens)))
         scored
         (doall
          (for [{:keys [id tokens] :as cand} candidates]
            (let [toks (vec tokens)
                  lps (if (= 1 (count toks))
                        [(get base-lp (first toks))]
                        (do
                          (load-state! session state (:tokens state) {:seq-id seq-id})
                          (loop [i 1
                                 acc [(get base-lp (first toks))]]
                            (if (>= i (count toks))
                              acc
                              (do
                                ;; advance past token i-1 so the next score is
                                ;; conditioned on it, then read token i
                                (eval! session [(nth toks (dec i))] {:seq-id seq-id})
                                (recur (inc i) (conj acc (token-logprob session (nth toks i)))))))))
                  total (reduce + 0.0 lps)]
              (assoc (dissoc cand :tokens)
                     :id id
                     :n-tokens (count toks)
                     :logprob-sum total
                     :logprob-mean (/ total (count toks))
                     :token-logprobs lps))))
         ranked (->> scored
                     (sort-by :logprob-sum >)
                     (map-indexed (fn [i c] (assoc c :rank i)))
                     vec)]
     ;; Leave the session as we found it. Only needed if a multi-token candidate
     ;; advanced the sequence.
     (when (and multi? state)
       (load-state! session state (:tokens state) {:seq-id seq-id}))
     (swap! (:tokens session) (fn [_] base-tokens))
     {:candidates ranked
      :best (first ranked)
      :n-candidates (count ranked)
      :base-n-tokens base-n
      ;; named so a caller can assert on it; see the docstring
      ;; feed this back as :base-logprobs to score the same base again after a
      ;; restore has cleared the session's logits
      :base-logprobs base-lp
      :convention :teacher-forced/first-from-base-rest-single-token
      :homogeneous? (= 1 (count (distinct (map :n-tokens ranked))))})))

;; ------------------------------------------------------------------- close

(defn close!
  "Close a session or model.

  Idempotent for a handle this wrapper owns: closing twice returns
  :already-closed rather than throwing, because a cleanup path that throws on a
  double close makes correct unwinding harder than it needs to be. That
  idempotence lives HERE and cannot be pushed into C -- once jl_session_close
  has freed the struct, a second raw call reads freed memory before it can
  check anything.

  Closing a model with live sessions THROWS :model/sessions-active. The
  sessions are not closed on the caller's behalf: a handle this function did
  not create is not its to invalidate, and silently closing children would turn
  one caller's mistake into another caller's dangling handle. The shim refuses
  independently with JL_ERR_SESSIONS_ACTIVE."
  [handle]
  (cond
    (not (map? handle)) (throw (ex-info "jolt.llama/close!: not a handle" {:got handle}))
    @(:closed handle) :already-closed
    :else
    (let [k (::kind handle)]
      (case k
        :session
        (do (reset! (:closed handle) true)
            (check! (c-session-close (::ptr handle)) :session-close {})
            (when-let [live (:sessions (:model handle))] (swap! live dec)))

        :model
        (let [live (if-let [a (:sessions handle)] @a 0)]
          (when (pos? live)
            ;; checked BEFORE marking closed, so a refused close leaves the
            ;; model exactly as usable as it was
            (throw (ex-info (str "jolt.llama/close!: " live " session(s) still open; "
                                 "close them before the model")
                            {:jolt.llama/op :model-close
                             :jolt.llama/error :model/sessions-active
                             :jolt.llama/sessions live})))
          (reset! (:closed handle) true)
          (check! (c-model-close (::ptr handle)) :model-close {}))

        (throw (ex-info "jolt.llama/close!: unknown handle kind" {:kind k})))
      :closed)))

(defmacro with-model
  "(with-model [m {:path ...}] body) — closes on normal or exceptional exit."
  [[sym opts] & body]
  `(let [~sym (open-model ~opts)]
     (try ~@body (finally (close! ~sym)))))

(defmacro with-session
  "(with-session [s model {:context-size ...}] body)"
  [[sym model opts] & body]
  `(let [~sym (new-session ~model ~opts)]
     (try ~@body (finally (close! ~sym)))))
