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
(ffi/defcfn ^:private c-runtime-build-id "jl_runtime_build_id" [] :pointer)
(ffi/defcfn ^:private c-sha256-hex "jl_sha256_hex"
  [:pointer :size_t :pointer :size_t] :int32 :blocking)
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
   -13 :state/not-append
   -14 :session/poisoned})

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

(defn runtime-build-id
  "Identity of the native runtime that will serialize this session's state.

  Embedded in the shim at BUILD time, e.g.
  \"llama.cpp:b81c99b479d4c24e5eeca10de99032ebd343ef8f:clean\". Not decoration:
  the same model serialized to 20263632 bytes on one llama.cpp build and
  20263652 on another, so native state is NOT portable across builds -- and the
  shim ABI cannot stand in for that, since the shim can be byte-identical
  across two different llama.cpp trees.

  A build with no coordinate reports \"unknown:...\". That string is never
  treated as an identity -- see `unattributed?` -- because equality alone would
  let two shims built from two unidentified trees exchange state, which is the
  case the check exists to stop."
  []
  (ffi/ptr->string (c-runtime-build-id)))

(defn unattributed?
  "Whether a runtime id fails to identify anything.

  Anything that is nil, blank, or announces itself as unknown. Checked
  explicitly rather than relying on inequality: \"unknown:x\" equals
  \"unknown:x\", so a pure equality test would ACCEPT a state saved by an
  unidentified build into another unidentified build."
  [id]
  (or (nil? id)
      (not (string? id))
      (clojure.string/blank? id)
      (clojure.string/starts-with? id "unknown:")))

(defn- ensure-abi! []
  (let [got (c-abi-version)]
    (when-not (= got abi-expected)
      (throw (ex-info (str "jolt.llama: shim ABI " got " != expected " abi-expected
                           "; rebuild native/libjolt_llama.so")
                      {:jolt.llama/op :runtime-init
                       :jolt.llama/abi-found got
                       :jolt.llama/abi-expected abi-expected})))))

;; Process-wide native runtime lifecycle:
;;
;;     :uninitialized --CAS--> :initializing --> :initialized
;;                                           \-> :failed
;;
;; The previous boolean was read-then-write, so N threads opening their first
;; model could all observe false and all call jl_runtime_init -- and the native
;; refcount is a plain int, so those increments race too. Only the thread
;; winning :uninitialized -> :initializing calls in; the rest wait and observe
;; the same outcome. defonce takes no docstring in Jolt, hence the comment.
(defonce ^:private runtime-state (atom :uninitialized))

(defn init-runtime!
  "Initialise the native runtime exactly once per process.

  Idempotent and safe to call concurrently. The native side refcounts, but
  repeating it from Jolt would leak a reference per call AND the native
  refcount is a plain int with no atomicity of its own -- so this admits
  exactly one caller rather than relying on the C side to survive a race.

  Every waiter observes the same outcome: a failed initialisation is latched as
  :failed and re-thrown for all of them, rather than letting the loser of the
  race proceed as if the runtime were up.

  The runtime is never freed. It is not part of ordinary handle lifecycle, and
  that is the v0 contract."
  []
  (loop []
    (case @runtime-state
      :initialized :ok
      :failed (throw (ex-info "jolt.llama: native runtime failed to initialise"
                              {:jolt.llama/op :runtime-init
                               :jolt.llama/error :runtime/init-failed}))
      :initializing (do (Thread/sleep 1) (recur))
      :uninitialized
      (if (compare-and-set! runtime-state :uninitialized :initializing)
        (try
          (ensure-abi!)
          (check! (c-runtime-init) :runtime-init {})
          (reset! runtime-state :initialized)
          :ok
          (catch Throwable e
            ;; latch the failure so every waiter sees it rather than spinning
            (reset! runtime-state :failed)
            (throw e)))
        ;; lost the race; fall through and observe the winner's outcome
        (recur)))))

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
           ;; :open -> :closing -> :closed, moved only by compare-and-set!.
           ;; See close! for why a boolean was not enough.
           :state (atom :open)})))))

(defn- model-ptr [m]
  (when-not (and (map? m) (= :model (::kind m))) (throw (ex-info "not a model handle" {:got m})))
  (when (not= :open @(:state m)) (throw (ex-info "jolt.llama: model is closed"
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
           ;; identifies THIS session instance, so a checkpoint cannot be
           ;; mistaken for one taken from another session over the same model
           :session-id (str (random-uuid))
           ::ptr ptr
           :model model
           :n-ctx (c-session-n-ctx ptr)
           ;; The token vector currently resident in seq 0. This is the Jolt-side
           ;; half of the token-identity contract; the C layer has no opinion.
           :tokens (atom [])
           ;; STRICTLY MONOTONIC counter of state-moving operations. Never
           ;; rewound. An earlier version reset it when restoring this session's
           ;; own checkpoint, which let two DIVERGENT evaluations carry the same
           ;; number: eval to 6, restore to 5, evaluate something different, and
           ;; the session is at "6" again describing different logits. A
           ;; checkpoint saved at the first 6 would then match the second.
           :revision (atom 0)
           ;; The checkpoint this session is currently sitting ON, by unique
           ;; state id, or nil once anything has moved it. This -- not the
           ;; counter -- is what candidate rewind checks, because it answers the
           ;; question that actually matters: is the session right now at the
           ;; exact evaluation this state was captured from?
           :at-checkpoint (atom nil)
           :state (atom :open)})))))

(defn- session-ptr [s]
  (when-not (and (map? s) (= :session (::kind s))) (throw (ex-info "not a session handle" {:got s})))
  (when (not= :open @(:state s)) (throw (ex-info "jolt.llama: session is closed"
                                     {:jolt.llama/op :session-use
                                      :jolt.llama/error :handle/closed})))
  (::ptr s))

(defn clear!
  "Drop the session's cached state and forget its token vector.

  No :seq-id option. v0 is single-sequence, so the only legal value was 0 and
  advertising the parameter offered a capability that did not exist -- a caller
  passing -1 used to get a clear-everything operation, a multi-sequence concept,
  instead of an error."
  [session]
  (let [p (session-ptr session)]
    (check! (c-session-clear p 0) :session-clear {})
    (reset! (:tokens session) [])
    (swap! (:revision session) inc)
    (reset! (:at-checkpoint session) nil)
    :ok))

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
     (swap! (:revision session) inc)
     (reset! (:at-checkpoint session) nil)
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
  over the whole vocabulary so scores are comparable across calls.

  With `:bits? true` each entry also carries `:bits`, the raw IEEE-754
  representation of its score as 8 hex digits, read from the SAME native buffer
  rather than reconstructed. Printed decimals cannot establish bit identity --
  two different floats render identically at six places -- so any gate that
  claims exactness has to compare representations. Jolt runs on Chez, not the
  JVM, so Float/floatToRawIntBits is not available and the bits are taken by
  reading the buffer as uint32."
  ([session k] (top-k session k {}))
  ([session k {:keys [pieces? bits?] :or {pieces? true bits? false}}]
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
                         bits? (assoc :bits (format "%08x"
                                                    (bit-and (ffi/read lbuf :uint32 (* i 4))
                                                             0xffffffff)))
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

(defn- sha256-bytes
  "SHA-256 of a byte array, as 64 lowercase hex characters. A REAL content
  digest, not a sentinel.

  Computed by the shim because the bytes are already in native memory there.
  The alternative was measured rather than assumed: feeding a 52 MiB state blob
  to jolt-crypto's OpenSSL FFI took ~47 000 ms, essentially all of it
  marshalling the array across the boundary rather than hashing -- against
  ~100 ms computing it natively. jolt-crypto is a perfectly good library and
  this is not an argument against depending on it; it is an argument against
  moving 52 MiB through an FFI to reach it."
  [^bytes arr]
  (let [n (alength arr)]
    (ffi/with-alloc [buf (max 1 n)]
      (ffi/write-array buf arr)
      (ffi/with-alloc [out 65]
        (check! (c-sha256-hex buf n out 65) :sha256 {})
        (ffi/ptr->string out)))))

(defn- sha256-tokens
  "SHA-256 of a token vector over its canonical little-endian int32 encoding.

  Canonical so the digest means the same thing for the same tokens regardless
  of how they were held in memory."
  [toks]
  (let [v (vec toks)
        n (count v)
        arr (byte-array (* 4 n))]
    (dotimes [i n]
      (let [t (int (nth v i))]
        (aset arr (* 4 i)       (unchecked-byte t))
        (aset arr (+ (* 4 i) 1) (unchecked-byte (bit-shift-right t 8)))
        (aset arr (+ (* 4 i) 2) (unchecked-byte (bit-shift-right t 16)))
        (aset arr (+ (* 4 i) 3) (unchecked-byte (bit-shift-right t 24)))))
    (sha256-bytes arr)))

(defn save-state
  "Capture exact native state for a sequence, bound to its token vector.

  The returned descriptor carries everything needed to decide later whether
  reuse is legal: the token vector itself, its count, a hash, the native byte
  count, and the model coordinate. The raw blob is held in memory here; a
  content-addressed store is a later concern."
  ([session] (save-state session {}))
  ([session _opts]
   ;; no :seq-id option: v0 is single-sequence and 0 was the only legal value
   (let [seq-id 0
         p (session-ptr session)
         toks @(:tokens session)]
     (ffi/with-out [np :size_t]
       (check! (c-state-save p (int seq-id) ffi/null 0 np) :state-save {})
       (let [n (ffi/read np :size_t)
             blob (byte-array n)]
         (ffi/with-alloc [buf n]
           (check! (c-state-save p (int seq-id) buf n np) :state-save {})
           (let [written (ffi/read np :size_t)
                 state-id (str (random-uuid))]
             (ffi/read-into! buf blob 0 written)
             ;; Saving does not move native state, so the session IS at this
             ;; checkpoint the moment it is taken. Recording that is what lets
             ;; the very next score-candidates rewind to it.
             (reset! (:at-checkpoint session) state-id)
             {::kind :state
              ;; THE compatibility coordinate. :model-content-id is the one that
              ;; decides; path and desc are kept for humans reading a descriptor
              ;; and are explicitly NOT trusted -- a path can point at different
              ;; bytes between runs and descriptions collide across
              ;; quantizations of the same model.
              ;; the NATIVE RUNTIME that produced this blob. The model and the
              ;; ABI together do not identify it: two llama.cpp builds behind
              ;; one shim ABI serialize different bytes.
              ;; WHICH EVALUATION this state was taken from. Two states with
              ;; identical tokens can represent different numerical bases, so a
              ;; checkpoint used for candidate rewind must match the session
              ;; and revision it claims to be a checkpoint OF.
              ;; unique per save: two saves of the same tokens at the same
              ;; revision are still different checkpoints, and a counter alone
              ;; could not say so
              :state-id state-id
              :session-id (:session-id session)
              :revision @(:revision session)
              :runtime-id (runtime-build-id)
              :model-content-id (get-in session [:model :content-id])
              :model-desc (get-in session [:model :desc])
              :model-path (get-in session [:model :path])
              :seq-id seq-id
              :tokens toks
              :n-tokens (count toks)
              :token-sha256 (sha256-tokens toks)
              :state-bytes written
              :state-sha256 (sha256-bytes blob)
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
      ;; UNATTRIBUTABLE first. String equality would let "unknown:x" match
      ;; "unknown:x", so two shims built from two unidentified trees would
      ;; happily exchange state -- which is the exact case the runtime check
      ;; exists to stop. An unknown coordinate is not a coordinate.
      (or (unattributed? (:runtime-id state))
          (unattributed? (runtime-build-id)))   :state/runtime-unattributed
      ;; distinct from the ABI check on purpose: the shim can be byte-identical
      ;; across two llama.cpp trees that serialize state differently
      (not= (:runtime-id state) (runtime-build-id)) :state/runtime-mismatch
      (not= 0 (:seq-id state))                 :seq/unsupported
      ;; The TOKEN digest is always verified. It is ~2 ms and it catches the
      ;; case that actually matters: a descriptor whose token vector no longer
      ;; describes its blob, which is what the whole prefix contract rests on.
      (not= (:token-sha256 state) (sha256-tokens (:tokens state)))
      :state/token-digest-mismatch

      ;; The BLOB digest is verified when the state did not originate in THIS
      ;; session -- a cross-session restore, or anything that will later cross a
      ;; process boundary. It is ~345 ms over a 52 MiB blob, and candidate
      ;; rewind restores once per multi-token candidate, so verifying it on
      ;; every same-session restore put a third of a second on the hot path and
      ;; took the exact-spine speedup from 3.93x to 3.00x.
      ;;
      ;; The trade, stated rather than hidden: within one session the blob is an
      ;; immutable in-process byte array this library created and never hands
      ;; out, so re-digesting it on each rewind detects only memory corruption.
      ;; Across sessions the descriptor may have come from anywhere and is
      ;; checked in full. `verify-state-digest` forces the full check when a
      ;; caller wants it regardless.
      (and (not= (:session-id state) (:session-id session))
           (not= (:state-sha256 state) (sha256-bytes (::blob state))))
      :state/blob-digest-mismatch
      (not= (:state-bytes state) (alength ^bytes (::blob state)))
      :state/blob-size-mismatch
      (not= (:model-content-id state) (:content-id model))
      :state/model-mismatch
      :else nil)))

(defn verify-state-digest
  "Recompute and check the state blob's SHA-256, whatever its origin.

  state-compatible? skips this for a state that originated in the calling
  session, because candidate rewind restores once per multi-token candidate and
  a full re-digest there costs more than the restore. Call this explicitly
  before trusting a descriptor that has been held for a long time, handed
  between components, or is about to cross a process boundary."
  [state]
  (= (:state-sha256 state) (sha256-bytes (::blob state))))

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
        ;; The counter always ADVANCES -- restoring is a state-moving operation
        ;; like any other, and rewinding it would let two divergent evaluations
        ;; share a number. What identifies the position is the checkpoint the
        ;; session is now sitting on, which is unique per save and cannot
        ;; collide. A checkpoint from ANOTHER session loads its bytes but does
        ;; not put this session "on" it: this session is not that session.
        (swap! (:revision session) inc)
        (reset! (:at-checkpoint session)
                (when (= (:session-id state) (:session-id session))
                  (:state-id state)))
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
    :state/token-digest-mismatch the descriptor's tokens were altered
    :state/blob-digest-mismatch  the descriptor's blob was altered
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

(defn- base-identity
  "What identifies the evaluation a session is sitting on right now.

  Two cases, and neither alone is sufficient:

    a checkpoint has been taken -> its unique id. Stable across a restore TO
      it, which is exactly the legitimate rewind score-candidates performs, so
      a counter would reject the descriptor it had just returned.

    no checkpoint               -> the monotonic revision. Two different
      evaluations always carry different numbers, so a caller that never saved
      still cannot replay one base's scores into another.

  The session id is carried alongside so a descriptor from a different session
  can never match by coincidence."
  [session]
  [(:session-id session)
   (or @(:at-checkpoint session) [:rev @(:revision session)])])

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
  differ by 0.286 at rank 30 (p~1e-8). See docs/EXACTNESS.md.

  BIT EXACTNESS IS DEFINED OVER THE WHOLE COMPARED VECTOR, not an intersection.
  It requires the same number of entries, the same token ids IN THE SAME ORDER,
  and identical raw float bits for every one. An earlier version compared only
  the ids present in both maps, so a 90-of-100 overlap could report
  :bit-exact? true -- and an EMPTY intersection reported it too, because
  `every?` over nothing is true. A gate documented as bit-exact over 100
  entries must not pass on 90, and must never pass on 0.

    :bit-exact?    identical length, order and raw bits
    :order-same?   the top-5 ids agreed
    :top1-same?    the argmax agreed
    :top1-abs      |delta| for the argmax -- the number that matters for policy
    :max-abs       worst over the compared entries, tail included
    :n-a :n-b      entries returned by each arm
    :n-common      ids present in both

  DESTRUCTIVE: clears the session. Intended for calibration and tests."
  ([session prefix suffix] (append-divergence session prefix suffix {}))
  ([session prefix suffix {:keys [k seq-id] :or {k 50 seq-id 0}}]
   (let [prefix (vec prefix) suffix (vec suffix)
         full   (into prefix suffix)
         _      (clear! session)
         _      (eval! session full {:seq-id seq-id})
         one    (top-k session k {:pieces? false :bits? true})
         _      (clear! session)
         _      (eval! session prefix {:seq-id seq-id})
         _      (eval! session suffix {:seq-id seq-id})
         two    (top-k session k {:pieces? false :bits? true})
         one-m  (into {} (map (juxt :token :logprob) one))
         two-m  (into {} (map (juxt :token :logprob) two))
         common (filter two-m (keys one-m))
         ds     (map #(abs (- (double (one-m %)) (double (two-m %)))) common)
         t1     (:token (first one))
         ;; the whole vector, in order, by raw bits
         exact? (and (pos? (count one))
                     (= (count one) (count two))
                     (= (mapv :token one) (mapv :token two))
                     (= (mapv :bits one) (mapv :bits two)))]
     {:bit-exact?  exact?
      :top1-abs    (when-let [b (two-m t1)]
                     (abs (- (double (one-m t1)) (double b))))
      :max-abs     (if (seq ds) (apply max ds) -1.0)
      :top1-same?  (= t1 (:token (first two)))
      :order-same? (= (mapv :token (take 5 one)) (mapv :token (take 5 two)))
      :n-a         (count one)
      :n-b         (count two)
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
                          ;; STRICT: the whole vector by raw bits. Falling back
                          ;; to a tolerance on max-abs would let a partial or
                          ;; reordered comparison satisfy a bit-exact claim.
                          ok (:bit-exact? r)]
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

  PROMOTION SCOPE, v0.

  The SINGLE-TOKEN path is promoted. Every candidate of exactly one token is
  scored from one base distribution, needs no candidate evaluation, no saved
  state and no rewind, touches neither native state nor the token ledger, and
  is exactly comparable under the validated exactness contract.

  The MULTI-TOKEN path remains EXPERIMENTAL, but the numerical-base hole is now
  closed. A rewind checkpoint must name THIS session and THIS evaluation
  revision, not merely carry matching tokens -- because token equality is not
  numerical equality: the same token vector reached by a different call
  structure produces different logits below the calibrated append threshold.
  Property C5 constructs exactly that case and asserts the refusal.
  :base-logprobs is likewise a descriptor bound to its origin rather than a
  bare map that could carry scores from another base.

  Failure atomicity is now defined too: a failure during candidate realisation
  restores the base, and if the restore itself fails the session is marked
  :poisoned and refuses further use rather than being handed back looking
  healthy. Property C7 injects a failure after a candidate has already moved
  native state and asserts one of those two outcomes.

  It remains EXPERIMENTAL because its guarantees rest on identity checks and a
  recovery path rather than on the structural argument the single-token path
  has -- that path cannot desynchronise because it never moves native state at
  all. Prefer single-token action vocabularies where the choice exists.

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
         ;; THE SAVED STATE MUST BE THIS EXACT BASE, not merely a compatible
         ;; one. Every candidate's FIRST token is read from the logits the
         ;; session is holding right now (base A); its later tokens are reached
         ;; by restoring `state` and advancing. If `state` were some other valid
         ;; checkpoint B, the score would mix P(t1 | A) with P(t2 | B, t1) --
         ;; a conditional that describes no sequence at all, and one that
         ;; state-compatible? happily accepts because B is a perfectly good
         ;; state for this model, runtime and ABI.
         ;;
         ;; Equality, not prefix: a strict prefix of the base is individually
         ;; reusable as a checkpoint and still wrong here, because the tokens
         ;; between it and the base would silently vanish from the conditional.
         ;;
         ;; Checked BEFORE any candidate touches native state, so a refusal
         ;; leaves the session exactly as it was found.
         ;; TOKEN EQUALITY IS NOT NUMERICAL EQUALITY. This library's own
         ;; measurements show the same token vector reached by a different call
         ;; structure produces different logits below the calibrated append
         ;; threshold, so a state carrying the right tokens can still represent
         ;; a different numerical base than the logits the first token is read
         ;; from -- composing P(t1 | A) with P(t2 | B, t1), which is the same
         ;; defect as before one level down.
         ;;
         ;; The checkpoint must therefore be THIS session at THIS evaluation
         ;; revision, not merely a state with matching tokens. Ordinary
         ;; load-state! portability between two handles over the same artifact
         ;; is deliberately NOT tightened; the stricter identity applies only to
         ;; a checkpoint claiming to represent the current logits.
         _ (when multi?
             (let [why (cond
                         (not= (:session-id state) (:session-id session))
                         :score/base-session-mismatch
                         ;; The session must be sitting ON this exact
                         ;; checkpoint. A counter cannot express it: restoring
                         ;; is itself a state-moving operation so the number
                         ;; always advances, and an earlier version that rewound
                         ;; the counter let two DIVERGENT evaluations share one.
                         ;; Only a unique per-save id says "this is the
                         ;; evaluation you captured".
                         (not= (:state-id state) @(:at-checkpoint session))
                         :score/base-checkpoint-mismatch
                         (not= (vec (:tokens state)) (vec base-tokens))
                         :score/base-state-mismatch
                         :else nil)]
               (when why
                 (throw (ex-info (str "jolt.llama/score-candidates: :state is not this "
                                      "session's current scoring base (" (name why) ")")
                                 {:jolt.llama/op :score-candidates
                                  :jolt.llama/error why
                                  :jolt.llama/base-n base-n
                                  :jolt.llama/state-n (count (:tokens state))
                                  :jolt.llama/state-revision (:revision state)
                                  :jolt.llama/session-revision @(:revision session)})))))
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
         ;; :base-logprobs is a DESCRIPTOR, not a bare token->score map. A bare
         ;; map can silently come from another prompt, session or evaluation,
         ;; giving first-token scores from one base and continuations from
         ;; another. It is accepted only when it names the same session and
         ;; revision it is being replayed into.
         _ (when base-logprobs
             (let [why (cond
                         (not (map? (:scores base-logprobs))) :score/base-logprobs-malformed
                         (not= (:base-identity base-logprobs) (base-identity session))
                         :score/base-logprobs-mismatch
                         (not (every? (:scores base-logprobs) first-tokens))
                         :score/base-logprobs-incomplete
                         :else nil)]
               (when why
                 (throw (ex-info (str "jolt.llama/score-candidates: :base-logprobs does not "
                                      "describe this scoring base (" (name why) ")")
                                 {:jolt.llama/op :score-candidates
                                  :jolt.llama/error why
                                  :jolt.llama/given-revision (:revision base-logprobs)
                                  :jolt.llama/session-revision @(:revision session)})))))
         base-lp (if base-logprobs
                   (select-keys (:scores base-logprobs) first-tokens)
                   (into {} (map (fn [t] [t (token-logprob session t)]) first-tokens)))
         ;; FAILURE ATOMICITY. Candidate scoring moves native state, so an
         ;; exception partway through used to skip the final restore and leave
         ;; the session advanced past its base -- with the token ledger still
         ;; claiming the base, which is the desynchronisation every later
         ;; append-only check reads. The whole realisation is wrapped: on any
         ;; failure the base is restored, and if the restore ITSELF fails the
         ;; session is marked poisoned rather than handed back looking healthy.
         restore-base!
         (fn [] (when multi? (load-state! session state (:tokens state) {:seq-id seq-id})))
         scored
         (try
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
          (catch Throwable e
            (let [recovered?
                  (try (restore-base!) true
                       (catch Throwable _ false))]
              (when-not recovered?
                ;; the session's ledger and native state may now disagree and
                ;; nothing here can reconcile them; say so loudly
                (reset! (:state session) :poisoned))
              (throw (ex-info (str "jolt.llama/score-candidates: failed during candidate "
                                   "realisation; base "
                                   (if recovered? "restored" "NOT restored, session poisoned"))
                              {:jolt.llama/op :score-candidates
                               :jolt.llama/error (if recovered?
                                                   :score/failed-base-restored
                                                   :score/failed-session-poisoned)
                               :jolt.llama/cause (ex-message e)}
                              e)))))
         ranked (->> scored
                     (sort-by :logprob-sum >)
                     (map-indexed (fn [i c] (assoc c :rank i)))
                     vec)]
     ;; Leave the session as we found it, through ONE operation that moves the
     ;; native state and the token ledger together. The previous code restored
     ;; native state and then separately forced the ledger back to base-tokens,
     ;; so if the two ever disagreed the ledger would confidently describe a
     ;; sequence the session was not holding -- and load-state!'s prefix check,
     ;; the thing that would have caught it, reads that same ledger.
     ;;
     ;; Safe to drop the manual swap because state.tokens == base-tokens is
     ;; enforced above, and load-state! sets the ledger from the state it
     ;; restored. Only needed at all if a multi-token candidate advanced the
     ;; sequence; the single-token path never moves native state.
     (restore-base!)
     {:candidates ranked
      :best (first ranked)
      :n-candidates (count ranked)
      :base-n-tokens base-n
      ;; named so a caller can assert on it; see the docstring
      ;; feed this back as :base-logprobs to score the same base again after a
      ;; restore has cleared the session's logits. Bound to the session and
      ;; revision it came from, so it cannot be replayed into a different base.
      :base-logprobs {:base-identity (base-identity session)
                      :session-id (:session-id session)
                      :scores base-lp}
      ;; The promoted/experimental split was documentation only, so a caller
      ;; could not tell which guarantee it was relying on without reading a
      ;; docstring. Reported per call, from what the candidates actually are.
      :promoted? (not multi?)
      :convention :teacher-forced/first-from-base-rest-single-token
      :homogeneous? (= 1 (count (distinct (map :n-tokens ranked))))})))

;; ------------------------------------------------------------------- close

(defn close!
  "Close a session or model. Returns :closed, :already-closed, or throws.

  CONCURRENCY CONTRACT, v0. Handles are otherwise THREAD-CONFINED: one session
  must not be evaluated from two threads at once, and this library does not
  claim concurrent inference safety. CLOSE is the deliberate exception, hardened
  because cleanup is exactly where accidental concurrent calls happen -- two
  finally blocks, a shutdown hook racing a worker.

  The handle carries :state, moved only by compare-and-set!:

      :open --CAS--> :closing --> :closed
                              or  :close-failed   (native close threw)

  Only the thread that wins :open -> :closing calls native free. Every other
  caller observes :closing or :closed and returns :already-closed without
  touching the pointer. A boolean could not express this: read-then-write left
  a window where two threads both saw false, both wrote true, and both called
  free on one pointer.

  This is idempotence that CANNOT live in C. Once jl_session_close has freed the
  struct a second raw call reads freed memory before it can check anything, so
  the guard has to sit above the pointer -- which is what this atom is.

  A native close that throws leaves the handle at :close-failed, which is
  terminal and observable, rather than parked at :closing where every later
  caller would be told :already-closed about a resource that was never freed.

  Closing a model with live sessions THROWS :model/sessions-active and leaves
  the model :open and fully usable. Child sessions are never closed on the
  caller's behalf: a handle this function did not create is not its to
  invalidate.

  The native runtime refcount is NOT thread-safe and is not part of ordinary
  handle lifecycle -- Jolt initialises the runtime once, on first model open,
  and never frees it. That is the v0 contract, stated rather than defended with
  a lock nobody needs yet."
  [handle]
  (if-not (map? handle)
    (throw (ex-info "jolt.llama/close!: not a handle" {:got handle}))
    (let [k (::kind handle)
          st (:state handle)]
      (case k
        :session
        (if-not (compare-and-set! st :open :closing)
          :already-closed
          (try
            (check! (c-session-close (::ptr handle)) :session-close {})
            ;; released only by the thread that actually closed, so the model's
            ;; count can never go negative or be decremented twice
            (when-let [live (:sessions (:model handle))] (swap! live dec))
            (reset! st :closed)
            :closed
            (catch Throwable e
              ;; A native close that FAILED must not leave the handle parked at
              ;; :closing, where every later caller gets :already-closed for a
              ;; resource that was never released and the model can never close.
              ;; :close-failed is terminal and observable.
              (reset! st :close-failed)
              (throw e))))

        :model
        ;; the live-session check happens BEFORE the CAS, so a refused close
        ;; leaves the model :open rather than stranded in :closing
        (let [live (if-let [a (:sessions handle)] @a 0)]
          (when (pos? live)
            (throw (ex-info (str "jolt.llama/close!: " live " session(s) still open; "
                                 "close them before the model")
                            {:jolt.llama/op :model-close
                             :jolt.llama/error :model/sessions-active
                             :jolt.llama/sessions live})))
          (if-not (compare-and-set! st :open :closing)
            :already-closed
            (try
              (check! (c-model-close (::ptr handle)) :model-close {})
              (reset! st :closed)
              :closed
              (catch Throwable e
                (reset! st :close-failed)
                (throw e)))))

        (throw (ex-info "jolt.llama/close!: unknown handle kind" {:kind k}))))))

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
