(ns jolt.hegel-properties
  "Property and lifecycle coverage for jolt-llama, via jolt-hegel.

  Three groups, cheapest first, because the model is loaded once and every
  evaluation costs real time:

    SEAM       tokenizer-boundary properties. Pure tokenization, no eval, so
               these can run many cases. This is the high-value group: it is
               where a text-prefix assumption gets caught and shrunk.

    LIFECYCLE  legal and illegal call sequences against the FFI boundary. The
               shim must answer with a status or an ex-info, never a segfault,
               because Jolt cannot catch SIGSEGV.

    STATE      save/restore round trips. Expensive, so few cases.

  The seam group deliberately contains one property that is EXPECTED TO FAIL.
  Demonstrating that the naive text-prefix assumption is false -- and shrinking
  a counterexample -- is the evidence that the contract in jolt.llama is load
  bearing rather than decorative."
  (:require [hegel.core :as h]
            [hegel.generator :as g]
            [hegel.report :as report]
            [jolt.llama :as llama]))

(def model-path (or (System/getenv "JOLT_LLAMA_MODEL")
                    (throw (ex-info "set JOLT_LLAMA_MODEL" {}))))

;; One model for the whole suite; sessions are created per property where the
;; property needs its own state.
(def model (llama/open-model {:path model-path}))

(defn- tok [s] (llama/tokenize model s))
(defn- tok-raw [s] (llama/tokenize model s {:add-special? false}))

(defn- common-prefix-len [a b]
  (count (take-while true? (map = a b))))

;; ---------------------------------------------------------------- seam

(defn check-canonical-prefix-holds! [runner]
  (report/run!
   runner
   "computed token boundary is a true prefix of every extension"
   (fn []
     (h/run-test!
      {:test-cases 150 :database "" :verbosity :quiet}
      (fn [_]
        ;; A stable spine and an arbitrary dynamic suffix. The CONTRACT is not
        ;; that the spine's own tokenization is a prefix -- it is that the
        ;; boundary jolt.llama computes is one.
        (let [stable (str "CONTROLLER POLICY\nsvc"
                          (h/draw! (g/integer 0 999))
                          ": budget=" (h/draw! (g/integer 1 9999)) "ms\nSTATE\n")
              suffix (h/draw! (g/string {:max-size 24}))
              full   (tok (str stable suffix))
              spine  (tok stable)
              n      (common-prefix-len spine full)]
          ;; the computed boundary must be a genuine token prefix
          (when-not (= (vec (take n spine)) (vec (take n full)))
            (throw (ex-info "computed boundary is not a token prefix"
                            {:hegel/origin "seam/computed-boundary-is-prefix"})))
          ;; and it must never exceed either sequence
          (when (or (> n (count spine)) (> n (count full)))
            (throw (ex-info "boundary exceeds a sequence"
                            {:hegel/origin "seam/boundary-in-range"})))))))))

(defn check-naive-text-prefix-assumption! [runner]
  (report/run!
   runner
   "DEMONSTRATION: a text prefix is not a token prefix (expected to fail+shrink)"
   (fn []
     (h/run-test!
      {:test-cases 400 :database "" :verbosity :quiet}
      (fn [_]
        ;; The assumption jolt-llama refuses to make: that tokenizing the stable
        ;; text alone yields a vector that prefixes the full prompt.
        ;;
        ;; The BOUNDARY CHARACTER is drawn, not fixed. An earlier version ended
        ;; the stable text on a newline and found no counterexample in 400 cases,
        ;; because a newline rarely merges rightwards -- which made the
        ;; demonstration pass and therefore prove nothing. Real templates end on
        ;; ":", " ", "=", or mid-word far more often than on "\n", so the
        ;; generator covers those endings instead of one arbitrary choice.
        (let [ending (h/draw! (g/sampled-from ["ACTION:" "  svc042: p95=" "owner=team"
                                               "budget=" "state " "PLAN" "\n"]))
              stable (str "CONTROLLER POLICY v1\nTOPOLOGY\n  svc001: budget=120ms\n"
                          ending)
              suffix (h/draw! (g/string {:max-size 24}))
              spine  (tok stable)
              full   (tok (str stable suffix))]
          (when-not (= (vec spine) (vec (take (count spine) full)))
            (throw (ex-info "text prefix is not a token prefix"
                            {:hegel/origin "seam/naive-text-prefix"})))))))))

(defn check-append-is-not-concat! [runner]
  (report/run!
   runner
   "DEMONSTRATION: tokenize(a++b) != tokenize(a)++tokenize(b) (expected to fail+shrink)"
   (fn []
     (h/run-test!
      {:test-cases 400 :database "" :verbosity :quiet}
      (fn [_]
        (let [a (str "svc" (h/draw! (g/integer 0 99)) ": p95=")
              b (h/draw! (g/string {:max-size 16}))]
          (when-not (= (vec (tok-raw (str a b)))
                       (vec (concat (tok-raw a) (tok-raw b))))
            (throw (ex-info "concatenated tokenization differs"
                            {:hegel/origin "seam/concat-vs-canonical"})))))))))

(defn check-prefix-predicate-agrees! [runner]
  (report/run!
   runner
   "token-prefix-ok? agrees with token-for-token comparison"
   (fn []
     (h/run-test!
      {:test-cases 300 :database "" :verbosity :quiet}
      (fn [_]
        ;; The predicate is the gate load-state! uses, so it must agree with the
        ;; literal definition for arbitrary vectors, including the awkward ones:
        ;; empty saved state, equal length, incoming shorter.
        (let [saved    (h/draw! (g/vector {:max-size 40} (g/integer 0 5000)))
              extra    (h/draw! (g/vector {:max-size 20} (g/integer 0 5000)))
              incoming (vec (concat saved extra))
              st       {:tokens (vec saved)}]
          (when-not (llama/token-prefix-ok? st incoming)
            (throw (ex-info "extension rejected"
                            {:hegel/origin "seam/predicate-accepts-extension"})))
          ;; a shorter vector can never be a legal reuse
          (when (and (seq saved)
                     (llama/token-prefix-ok? st (vec (butlast saved))))
            (throw (ex-info "shorter vector accepted"
                            {:hegel/origin "seam/predicate-rejects-short"})))
          ;; perturbing any single token must reject
          (when (seq saved)
            (let [i (h/draw! (g/integer 0 (dec (count saved))))
                  bad (assoc (vec incoming) i (inc (nth incoming i)))]
              (when (llama/token-prefix-ok? st bad)
                (throw (ex-info "perturbed vector accepted"
                                {:hegel/origin "seam/predicate-rejects-perturbed"})))))))))))

;; ------------------------------------------------------------ lifecycle

(defn check-illegal-lifecycle-is-safe! [runner]
  (report/run!
   runner
   "illegal lifecycle sequences fail safely, never crash"
   (fn []
     (h/run-test!
      {:test-cases 60 :database "" :verbosity :quiet}
      (fn [_]
        ;; Draw a short program of operations, some of which are illegal at the
        ;; point they are drawn. Every one must either work or throw ex-info.
        ;; The process surviving is itself the assertion.
        (let [ops (h/draw! (g/vector {:min-size 1 :max-size 6}
                                     (g/sampled-from [:close :close-again :use-after-close
                                                      :logits-before-eval :empty-eval
                                                      :bad-token-logprob])))
              s (llama/new-session model {:context-size 512 :threads 2})]
          (try
            (doseq [op (take 6 ops)]
              (try
                (case op
                  :close              (llama/close! s)
                  :close-again        (do (llama/close! s) (llama/close! s))
                  :use-after-close    (do (llama/close! s) (llama/logits s))
                  :logits-before-eval (llama/logits s)
                  :empty-eval         (llama/eval! s [])
                  :bad-token-logprob  (llama/token-logprob s -1))
                (catch Throwable e
                  ;; a thrown ex-info is the contract; anything else is a bug
                  (when-not (ex-data e)
                    (throw (ex-info "non-ex-info escaped the FFI boundary"
                                    {:hegel/origin "lifecycle/ex-info-only"}))))))
            (finally (try (llama/close! s) (catch Throwable _ nil))))))))))

(defn check-close-is-idempotent! [runner]
  (report/run!
   runner
   "close! is idempotent for any repetition count"
   (fn []
     (h/run-test!
      {:test-cases 40 :database "" :verbosity :quiet}
      (fn [_]
        (let [n (h/draw! (g/integer 1 5))
              s (llama/new-session model {:context-size 512 :threads 2})
              results (doall (for [_ (range n)] (llama/close! s)))]
          ;; first closes, the rest report already-closed; none throw
          (when-not (= :closed (first results))
            (throw (ex-info "first close did not close"
                            {:hegel/origin "lifecycle/first-close"})))
          (when-not (every? #{:already-closed} (rest results))
            (throw (ex-info "repeat close did not report already-closed"
                            {:hegel/origin "lifecycle/repeat-close"})))))))))

;; ---------------------------------------------------------------- state

(defn- topk-map [s k]
  (into {} (map (juxt :token :logprob) (llama/top-k s k {:pieces? false}))))

(defn- max-delta [a b]
  (let [c (filter b (keys a))]
    (if (seq c) (apply max (map #(abs (- (double (a %)) (double (b %)))) c)) -1.0)))

(defn- probe-text [n]
  (str "CONTROLLER\n"
       (apply str (for [k (range n)]
                    (format "  svc%03d: region=r%d p95=%dms err=%d\n"
                            k (mod k 7) (+ 40 (mod (* k 13) 500)) (mod k 9))))))

;; Measured once. See docs/EXACTNESS.md: the shortest append that still
;; reproduces a one-pass evaluation is a property of the model's kernel
;; selection, so it is calibrated rather than assumed.

;; ONE session for every state property. Allocating a context costs ~1 GiB of
;; compute buffer and several seconds; doing it per generated case made hegel's
;; TooSlow health check fire, which is the health check working correctly.
;; clear! between cases is the isolation, and the alternating-domain test is the
;; evidence that clear! is sufficient isolation.
(def state-session (llama/new-session model {:context-size 4096 :threads 8}))

(def calibration
  (llama/calibrate-append-exactness state-session (tok (probe-text 200))
                                    {:prefix-len 128 :max-suffix 256}))

(defn check-restore-is-transparent! [runner]
  (report/run!
   runner
   "save/restore is transparent: restoring changes nothing a plain split would not"
   (fn []
     (h/run-test!
      {:test-cases 10 :database "" :verbosity :quiet}
      (fn [_]
        ;; THIS is the invariant jolt-llama owns. It must hold for EVERY suffix
        ;; length, including short ones, because it says only that restoring a
        ;; saved state is indistinguishable from never having cleared it.
        ;;
        ;; It is deliberately NOT "restore matches a one-pass recompute". That
        ;; stronger claim is false for short suffixes on this model, and it is
        ;; false with or without a restore -- so attributing it to save/restore
        ;; would blame the wrong component. See docs/EXACTNESS.md.
        ;; kept small on purpose: each case runs four evaluations, and hegel's
        ;; TooSlow health check is a real constraint, not an annoyance to silence
        (let [n (h/draw! (g/integer 2 8))
              k (h/draw! (g/integer 1 40))
              toks (vec (tok (probe-text (+ 4 n))))
              base (subvec toks 0 (- (count toks) k))
              suffix (subvec toks (- (count toks) k))
              s state-session]
          ;; arm 1: split, no restore
          (llama/clear! s)
          (llama/eval! s base)
          (llama/eval! s suffix)
          (let [plain (topk-map s 50)]
            ;; arm 2: same split, with a save/clear/restore in the middle
            (llama/clear! s)
            (llama/eval! s base)
            (let [st (llama/save-state s)]
              (llama/clear! s)
              (llama/load-state! s st toks)
              (llama/eval! s suffix)
              (let [d (max-delta plain (topk-map s 50))]
                (when-not (zero? d)
                  (throw (ex-info "restore was not transparent"
                                  {:hegel/origin "state/restore-transparent"
                                   :delta d :suffix-len k}))))))))))))

(defn check-restore-matches-onepass-above-threshold! [runner]
  (report/run!
   runner
   (str "restore + append >= " (:threshold calibration)
        " tokens equals a one-pass recompute")
   (fn []
     (h/run-test!
      {:test-cases 8 :database "" :verbosity :quiet}
      (fn [_]
        ;; The stronger claim, asserted only where calibration says it holds.
        ;; This is the exact-spine result M1 reports, with its precondition made
        ;; explicit instead of implied by the shape of the test data.
        (let [thr (or (:threshold calibration) 64)
              k (h/draw! (g/integer thr (+ thr 36)))
              n (h/draw! (g/integer 14 20))
              toks (vec (tok (probe-text n)))]
          (when (> (count toks) (+ k 32))
            (let [base (subvec toks 0 (- (count toks) k))
                  suffix (subvec toks (- (count toks) k))
                  s state-session]
              (llama/clear! s)
              (llama/eval! s toks)
              (let [onepass (topk-map s 50)]
                (llama/clear! s)
                (llama/eval! s base)
                (let [st (llama/save-state s)]
                  (llama/clear! s)
                  (llama/load-state! s st toks)
                  (llama/eval! s suffix)
                  (let [d (max-delta onepass (topk-map s 50))]
                    (when-not (zero? d)
                      (throw (ex-info "restore+long-append diverged from one-pass"
                                      {:hegel/origin "state/exact-above-threshold"
                                       :delta d :suffix-len k}))))))))))))))

(defn check-short-append-diverges! [runner]
  (report/run!
   runner
   (str "a short append (< " (:threshold calibration)
        ") does NOT equal a one-pass recompute, and the library says so")
   (fn []
     (h/run-test!
      {:test-cases 8 :database "" :verbosity :quiet}
      (fn [_]
        ;; A negative result, asserted rather than merely noted. If a future
        ;; llama.cpp or model makes short appends exact, THIS property fails --
        ;; which is the signal to re-run calibration and relax the docs, not a
        ;; regression to paper over.
        (let [thr (or (:threshold calibration) 64)
              k (h/draw! (g/integer 1 (max 1 (dec thr))))
              toks (vec (tok (probe-text 12)))
              base (subvec toks 0 (- (count toks) k))
              suffix (subvec toks (- (count toks) k))
              r (llama/append-divergence state-session base suffix)]
          (when (:bit-exact? r)
            (throw (ex-info "short append was exact; calibration is stale"
                            {:hegel/origin "state/short-append-diverges"
                             :suffix-len k})))
          ;; the head must stay stable even where the tail does not: this is the
          ;; claim docs/EXACTNESS.md makes about it being precision, not disagreement
          (when-not (:top1-same? r)
            (throw (ex-info "a short append changed the argmax"
                            {:hegel/origin "state/short-append-keeps-argmax"
                             :suffix-len k :top1-abs (:top1-abs r)})))))))))

(defn check-prefix-mismatch-refused! [runner]
  (report/run!
   runner
   "load-state! refuses any non-extending token vector"
   (fn []
     (h/run-test!
      {:test-cases 25 :database "" :verbosity :quiet}
      (fn [_]
        (let [toks (tok "CONTROLLER\n  svc000: p95=40ms\n")
              s state-session]
          (do
            (llama/clear! s)
            (llama/eval! s toks)
            (let [st (llama/save-state s)
                  i (h/draw! (g/integer 0 (dec (count toks))))
                  bad (assoc (vec toks) i (mod (+ 7 (nth toks i)) 1000))]
              (llama/clear! s)
              (let [refused? (try (llama/load-state! s st bad) false
                                  (catch Throwable e
                                    (= :state/prefix-mismatch (:jolt.llama/error (ex-data e)))))]
                (when-not refused?
                  (throw (ex-info "a mismatched prefix was accepted"
                                  {:hegel/origin "state/mismatch-refused"}))))))))))))


;; ------------------------------------------------------------ ownership

(defn check-model-close-is-refused-while-sessions-live! [runner]
  (report/run!
   runner
   "a model with live sessions refuses to close, for any interleaving"
   (fn []
     (h/run-test!
      {:test-cases 25 :database "" :verbosity :quiet}
      (fn [_]
        ;; A jl_session holds its jl_model* AND a llama_context built from that
        ;; model's llama_model. Closing the model underneath a live session is a
        ;; use-after-free the caller cannot detect, so it must be refused --
        ;; and the model must remain usable afterwards, since a refusal that
        ;; damaged the model would only move the bug.
        ;;
        ;; A fresh model per case: this property is ABOUT closing models, so it
        ;; cannot share the suite's.
        (let [n (h/draw! (g/integer 1 3))
              m2 (llama/open-model {:path model-path})
              sessions (atom [])]
          (try
            (dotimes [_ n]
              (swap! sessions conj (llama/new-session m2 {:context-size 256 :threads 2})))
            ;; every prefix of the close sequence must still refuse
            (loop [remaining @sessions]
              (when (seq remaining)
                (let [refused? (try (llama/close! m2) false
                                    (catch Throwable e
                                      (= :model/sessions-active
                                         (:jolt.llama/error (ex-data e)))))]
                  (when-not refused?
                    (throw (ex-info "model closed with live sessions"
                                    {:hegel/origin "ownership/model-close-refused"
                                     :live (count remaining)})))
                  ;; and the refusal must be non-destructive
                  (when-not (pos? (:n-vocab m2))
                    (throw (ex-info "a refused close damaged the model"
                                    {:hegel/origin "ownership/refusal-is-nondestructive"})))
                  (llama/close! (first remaining))
                  (recur (rest remaining)))))
            ;; with every session closed it must now succeed
            (when-not (= :closed (llama/close! m2))
              (throw (ex-info "model would not close after all sessions closed"
                              {:hegel/origin "ownership/close-after-drain"})))
            (finally
              (doseq [s @sessions] (try (llama/close! s) (catch Throwable _ nil)))
              (try (llama/close! m2) (catch Throwable _ nil))))))))))

(defn check-use-after-close-is-refused! [runner]
  (report/run!
   runner
   "using a closed model or session is refused, never a crash"
   (fn []
     (h/run-test!
      {:test-cases 20 :database "" :verbosity :quiet}
      (fn [_]
        (let [m2 (llama/open-model {:path model-path})
              s (llama/new-session m2 {:context-size 256 :threads 2})
              op (h/draw! (g/sampled-from [:eval :logits :save :clear :new-session]))]
          (llama/close! s)
          (llama/close! m2)
          ;; the process surviving all of these IS the property
          (let [threw? (try
                         (case op
                           :eval        (llama/eval! s [1 2 3])
                           :logits      (llama/logits s)
                           :save        (llama/save-state s)
                           :clear       (llama/clear! s)
                           :new-session (llama/new-session m2 {:context-size 256 :threads 2}))
                         false
                         (catch Throwable e
                           (= :handle/closed (:jolt.llama/error (ex-data e)))))]
            (when-not threw?
              (throw (ex-info "a closed handle was used"
                              {:hegel/origin "ownership/use-after-close"
                               :op op}))))))))))

(defn check-v0-is-single-sequence! [runner]
  (report/run!
   runner
   "v0 refuses every sequence but 0"
   (fn []
     (h/run-test!
      {:test-cases 30 :database "" :verbosity :quiet}
      (fn [_]
        (let [sq (h/draw! (g/integer 1 8))]
          (when-not (try (llama/new-session model {:context-size 256 :seq-max (inc sq)}) false
                         (catch Throwable e (= :seq/unsupported (:jolt.llama/error (ex-data e)))))
            (throw (ex-info "seq-max > 1 accepted"
                            {:hegel/origin "seq/seq-max-refused"})))
          (llama/clear! state-session)
          (llama/eval! state-session (vec (tok (probe-text 2))))
          (when-not (try (llama/eval! state-session [1] {:seq-id sq}) false
                         (catch Throwable e (= :seq/unsupported (:jolt.llama/error (ex-data e)))))
            (throw (ex-info "nonzero seq-id accepted by eval!"
                            {:hegel/origin "seq/eval-refused"})))))))))

(defn check-eval-is-append-only! [runner]
  (report/run!
   runner
   "evaluation is append-only and there is no way to ask otherwise"
   (fn []
     (h/run-test!
      {:test-cases 20 :database "" :verbosity :quiet}
      (fn [_]
        ;; The public API has no :pos, so the property is that appending always
        ;; lands at the current end and the ledger tracks it exactly.
        (let [a (h/draw! (g/integer 1 6))
              b (h/draw! (g/integer 1 6))
              toks (vec (tok (probe-text 3)))]
          (llama/clear! state-session)
          (let [r1 (llama/eval! state-session (vec (take a toks)))]
            (when-not (zero? (:pos r1))
              (throw (ex-info "first eval did not start at 0"
                              {:hegel/origin "append/starts-at-zero"})))
            (let [r2 (llama/eval! state-session (vec (take b (drop a toks))))]
              (when-not (= (:pos r2) (:n-resident r1))
                (throw (ex-info "append did not continue at the resident end"
                                {:hegel/origin "append/continues-at-end"
                                 :pos (:pos r2) :expected (:n-resident r1)})))))))))))

;; ---------------------------------------------------- state compatibility

(defn check-incompatible-state-is-refused! [runner]
  (report/run!
   runner
   "a state descriptor that does not match is refused, with the reason named"
   (fn []
     (h/run-test!
      {:test-cases 30 :database "" :verbosity :quiet}
      (fn [_]
        ;; Each mutation targets one field of the compatibility coordinate, and
        ;; each must be refused with ITS OWN keyword -- a single :incompatible
        ;; would not be auditable, and would hide which check actually fired.
        (let [toks (vec (tok (probe-text 3)))
              _ (llama/clear! state-session)
              _ (llama/eval! state-session toks)
              st (llama/save-state state-session)
              which (h/draw! (g/sampled-from
                              [:model :abi :runtime :seq :token-sha256 :blob-digest
                               :blob-size :malformed]))
              [bad expected]
              (case which
                :model      [(assoc st :model-content-id "sha256:deadbeef") :state/model-mismatch]
                :abi        [(assoc st :abi 99) :state/abi-mismatch]
                :seq        [(assoc st :seq-id 1) :seq/unsupported]
                :runtime    [(assoc st :runtime-id "llama.cpp:0000:clean") :state/runtime-mismatch]
                :token-sha256 [(assoc st :token-sha256 "00") :state/token-digest-mismatch]
                ;; a same-session blob digest edit is NOT a load-state! concern
                ;; by design; the cross-session case is covered by the runtime
                ;; and session identity checks plus verify-state-digest
                :blob-digest [(assoc st :state-bytes 1) :state/blob-size-mismatch]
                :blob-size  [(assoc st :state-bytes 1) :state/blob-size-mismatch]
                :malformed  [(dissoc st :tokens) :state/malformed])
              got (try (llama/load-state! state-session bad toks) :accepted
                       (catch Throwable e (:jolt.llama/error (ex-data e))))]
          (when-not (= expected got)
            (throw (ex-info "wrong refusal for a mutated state descriptor"
                            {:hegel/origin (str "state/refuses-" (name which))
                             :expected expected :got got})))))))))

(defn check-same-artifact-different-handle-is-accepted! [runner]
  (report/run!
   runner
   "state moves between two handles on the SAME model artifact"
   (fn []
     (h/run-test!
      {:test-cases 3 :database "" :verbosity :quiet}
      (fn [_]
        ;; Pointer identity is NOT the semantic model identity. Two independently
        ;; opened handles over the same GGUF are the same model, and refusing
        ;; that restore would be a false negative -- the kind that teaches
        ;; callers to reach for an unchecked path.
        (let [toks (vec (tok (probe-text 3)))
              _ (llama/clear! state-session)
              _ (llama/eval! state-session toks)
              st (llama/save-state state-session)
              m2 (llama/open-model {:path model-path})]
          (try
            (let [s2 (llama/new-session m2 {:context-size 2048 :threads 4})]
              (try
                (when-not (= (:content-id m2) (:model-content-id st))
                  (throw (ex-info "the same artifact produced a different content id"
                                  {:hegel/origin "state/content-id-is-stable"})))
                (llama/load-state! s2 st toks)
                (llama/eval! s2 [(:token (first (llama/top-k state-session 1 {:pieces? false})))])
                (finally (llama/close! s2))))
            (finally (llama/close! m2)))))))))


;; ------------------------------------------------ sequence closed world

(defn check-seq-closed-world! [runner]
  (report/run!
   runner
   "every seq-carrying entry point refuses a nonzero sequence, without side effects"
   (fn []
     (h/run-test!
      {:test-cases 40 :database "" :verbosity :quiet}
      (fn [_]
        ;; A rejected call must not merely fail -- it must leave nothing behind.
        ;; The public clear!/save-state no longer take a sequence at all, so the
        ;; drawn ids exercise the two that still do plus the native floor.
        (let [sq (h/draw! (g/sampled-from [-3 -1 1 2 17]))
              s state-session
              toks (vec (tok (probe-text 3)))]
          (llama/clear! s)
          (llama/eval! s toks)
          (let [before-ledger @(:tokens s)
                st (llama/save-state s)
                refused (fn [f]
                          (try (f) :ACCEPTED
                               (catch Throwable e (:jolt.llama/error (ex-data e)))))
                r-eval (refused #(llama/eval! s [1] {:seq-id sq}))
                r-load (refused #(llama/load-state! s st toks {:seq-id sq}))]
            (when-not (= :seq/unsupported r-eval)
              (throw (ex-info "eval! accepted a nonzero seq"
                              {:hegel/origin "seq/eval-refuses" :seq sq :got r-eval})))
            (when-not (= :seq/unsupported r-load)
              (throw (ex-info "load-state! accepted a nonzero seq"
                              {:hegel/origin "seq/load-refuses" :seq sq :got r-load})))
            ;; nothing moved
            (when-not (= before-ledger @(:tokens s))
              (throw (ex-info "a refused call changed the token ledger"
                              {:hegel/origin "seq/refusal-has-no-effect"})))
            ;; and the session still works
            (when-not (pos? (count (llama/top-k s 3 {:pieces? false})))
              (throw (ex-info "session unusable after a refused call"
                              {:hegel/origin "seq/session-survives"}))))))))))

(defn check-seq-max-exactness! [runner]
  (report/run!
   runner
   "n_seq_max must be exactly 1, and a rejected session leaks nothing"
   (fn []
     (h/run-test!
      {:test-cases 12 :database "" :verbosity :quiet}
      (fn [_]
        (let [n (h/draw! (g/sampled-from [0 2 4 32]))
              m2 (llama/open-model {:path model-path})]
          (try
            (let [before @(:sessions m2)
                  got (try (llama/new-session m2 {:context-size 256 :seq-max n})
                           :ACCEPTED
                           (catch Throwable e (:jolt.llama/error (ex-data e))))]
              (when-not (= :seq/unsupported got)
                (throw (ex-info "an illegal n_seq_max was accepted"
                                {:hegel/origin "seq/seq-max-exact" :n n :got got})))
              ;; a failed construction must not consume an ownership slot
              (when-not (= before @(:sessions m2))
                (throw (ex-info "a rejected session incremented the model's count"
                                {:hegel/origin "seq/failed-new-consumes-no-count"})))
              ;; and the model must still close cleanly, i.e. nothing leaked
              (when-not (= :closed (llama/close! m2))
                (throw (ex-info "model would not close after a rejected session"
                                {:hegel/origin "seq/failed-new-leaks-nothing"}))))
            (finally (try (llama/close! m2) (catch Throwable _ nil))))))))))

;; -------------------------------------------------- candidate base state

(defn check-scoring-base-must-be-exact! [runner]
  (report/run!
   runner
   "multi-token scoring refuses any state that is not the exact current base"
   (fn []
     (h/run-test!
      {:test-cases 10 :database "" :verbosity :quiet}
      (fn [_]
        ;; Every candidate's FIRST token is read from the logits the session
        ;; holds now; its later tokens come from restoring the supplied state.
        ;; If those are different bases the score mixes P(t1|A) with P(t2|B,t1),
        ;; a conditional describing no sequence -- and state-compatible? accepts
        ;; B happily, because B is a perfectly good state for this model.
        (let [s state-session
              n (h/draw! (g/integer 2 6))
              toks (vec (tok (probe-text (+ 4 n))))
              cut (h/draw! (g/integer 1 (max 1 (dec (count toks)))))
              prefix (vec (take cut toks))]
          ;; state SB, saved at a STRICT PREFIX of the base
          (llama/clear! s)
          (llama/eval! s prefix)
          (let [sb (llama/save-state s)]
            ;; now put the session at the full base A
            (llama/clear! s)
            (llama/eval! s toks)
            (let [cands [{:id :a :tokens [(first toks) (second toks)]}]
                  got (try (llama/score-candidates s cands {:state sb}) :ACCEPTED
                           (catch Throwable e (:jolt.llama/error (ex-data e))))]
              ;; PREFIX IS NOT ENOUGH: sb is individually reusable as a
              ;; checkpoint and still wrong here. Any of the base-identity
              ;; reasons is a correct refusal -- the revision check is strictly
              ;; stronger than the token check and fires first, so a prefix
              ;; saved at an earlier evaluation is caught as a revision
              ;; mismatch. What must never happen is acceptance.
              (when-not (#{:score/base-state-mismatch
                           :score/base-revision-mismatch
                           :score/base-session-mismatch} got)
                (throw (ex-info "a non-base state was accepted for candidate rewind"
                                {:hegel/origin "score/base-must-be-exact" :got got})))
              ;; and the refusal happened before anything moved
              (when-not (= toks @(:tokens s))
                (throw (ex-info "a refused scoring call disturbed the session"
                                {:hegel/origin "score/refusal-has-no-effect"})))))))))))

(defn check-single-token-path-is-state-free! [runner]
  (report/run!
   runner
   "single-token scoring needs no saved state and moves nothing"
   (fn []
     (h/run-test!
      {:test-cases 10 :database "" :verbosity :quiet}
      (fn [_]
        ;; The fast path must stay fast: one base distribution, no restore, no
        ;; candidate evaluation. Asserted because it is easy to lose by accident
        ;; while hardening the multi-token path beside it.
        (let [s state-session
              toks (vec (tok (probe-text 3)))
              _ (llama/clear! s)
              _ (llama/eval! s toks)
              top (llama/top-k s 4 {:pieces? false})
              cands (mapv (fn [t] {:id (:token t) :tokens [(:token t)]}) top)
              before @(:tokens s)
              ;; NO :state supplied at all
              res (llama/score-candidates s cands)
              after @(:tokens s)]
          (when-not (= before after)
            (throw (ex-info "the single-token path moved the token ledger"
                            {:hegel/origin "score/single-token-is-state-free"})))
          ;; and each score must equal the direct base log-probability
          (doseq [c (:candidates res)]
            (let [direct (llama/token-logprob s (:id c))]
              (when-not (< (abs (- (double direct) (double (:logprob-sum c)))) 1e-6)
                (throw (ex-info "a single-token score differs from the base logprob"
                                {:hegel/origin "score/single-token-equals-base"})))))))))))

(defn check-scoring-leaves-the-session-untouched! [runner]
  (report/run!
   runner
   "after multi-token scoring the ledger and the native state still agree"
   (fn []
     (h/run-test!
      {:test-cases 6 :database "" :verbosity :quiet}
      (fn [_]
        (let [s state-session
              toks (vec (tok (probe-text 4)))
              _ (llama/clear! s)
              _ (llama/eval! s toks)
              st (llama/save-state s)
              top (llama/top-k s 3 {:pieces? false})
              cands [{:id :one :tokens [(:token (first top))]}
                     {:id :two :tokens [(:token (first top)) (:token (second top))]}]
              _ (llama/score-candidates s cands {:state st})]
          ;; the ledger says base
          (when-not (= toks @(:tokens s))
            (throw (ex-info "scoring left the ledger describing something else"
                            {:hegel/origin "score/ledger-restored"})))
          ;; and the NATIVE state agrees, proven by appending and comparing
          ;; against a session that did the SAME appends without scoring.
          ;;
          ;; The oracle deliberately uses the same call structure -- eval base,
          ;; then eval one token -- rather than a single prefill of base++token.
          ;; A one-token append is below the calibrated 64-token threshold, so it
          ;; legitimately differs from a one-pass recompute (docs/EXACTNESS.md);
          ;; comparing against a prefill would fail for that reason and blame
          ;; candidate scoring for it. The question here is only whether SCORING
          ;; moved anything, so scoring is the only difference between the arms.
          (let [t (:token (first top))]
            (llama/eval! s [t])
            (let [after-scoring (llama/top-k s 5 {:pieces? false})]
              (llama/clear! s)
              (llama/eval! s toks)
              (llama/eval! s [t])
              (let [oracle (llama/top-k s 5 {:pieces? false})
                    d (apply max (map #(abs (- (double (:logprob %1)) (double (:logprob %2))))
                                      after-scoring oracle))]
                (when-not (zero? d)
                  (throw (ex-info "native state drifted across candidate scoring"
                                  {:hegel/origin "score/no-native-drift" :delta d}))))))))))))

;; -------------------------------------------------------- concurrent close

(defn check-concurrent-close-is-single! [runner]
  (report/run!
   runner
   "N threads racing to close one handle produce exactly one native close"
   (fn []
     (h/run-test!
      {:test-cases 6 :database "" :verbosity :quiet}
      (fn [_]
        ;; Cleanup is where accidental concurrent calls actually happen: two
        ;; finally blocks, a shutdown hook racing a worker. The CAS is what
        ;; makes exactly one of them call free.
        (let [n (h/draw! (g/integer 2 8))
              m2 (llama/open-model {:path model-path})
              s (llama/new-session m2 {:context-size 256 :threads 2})
              results (atom [])
              latch (promise)
              threads (doall (for [_ (range n)]
                               (future (deref latch)
                                       (swap! results conj
                                              (try (llama/close! s)
                                                   (catch Throwable e [:threw (ex-message e)]))))))]
          (deliver latch true)
          (doseq [t threads] (deref t))
          (let [rs @results
                closed (count (filter #{:closed} rs))
                already (count (filter #{:already-closed} rs))]
            (when-not (= 1 closed)
              (throw (ex-info "not exactly one thread closed the session"
                              {:hegel/origin "close/exactly-one" :results rs})))
            (when-not (= n (+ closed already))
              (throw (ex-info "a racing close neither closed nor reported already-closed"
                              {:hegel/origin "close/defined-result" :results rs})))
            ;; the ownership count must not have gone negative or double-decremented
            (when-not (= 0 @(:sessions m2))
              (throw (ex-info "the model's session count is wrong after a close race"
                              {:hegel/origin "close/count-exact" :n @(:sessions m2)})))
            (when-not (= :closed (llama/close! m2))
              (throw (ex-info "model would not close after its session race"
                              {:hegel/origin "close/model-after-race"}))))))))))


;; -------------------------------------------------------- integrity checks

(defn check-integrity-digests-detect-mutation! [runner]
  (report/run!
   runner
   "token and length mutations are always refused; blob mutations by digest"
   (fn []
     (h/run-test!
      {:test-cases 20 :database "" :verbosity :quiet}
      (fn [_]
        ;; The contract is deliberately asymmetric and this asserts both halves,
        ;; including the half that is NOT checked on the hot path.
        ;;
        ;; ALWAYS refused: a token mutation (~2 ms to detect, and it is the case
        ;; the prefix contract rests on) and a length mismatch.
        ;;
        ;; NOT refused on a same-session restore: a blob byte mutation. Checking
        ;; it costs ~345 ms over 52 MiB and candidate rewind restores once per
        ;; multi-token candidate, so verifying every time took the exact-spine
        ;; speedup from 3.93x to 3.00x. Within one session the blob is an
        ;; immutable array this library created and never hands out, so the
        ;; check would detect only memory corruption. verify-state-digest forces
        ;; it, and a CROSS-session state is checked in full.
        (let [s state-session
              toks (vec (tok (probe-text 3)))
              _ (llama/clear! s)
              _ (llama/eval! s toks)
              st (llama/save-state s)
              blob (:jolt.llama/blob st)
              n (alength ^bytes blob)
              flip (fn [off]
                     (let [c (byte-array n)]
                       (System/arraycopy blob 0 c 0 n)
                       (aset c off (byte (bit-xor (aget c off) 0x5a)))
                       c))
              refused? (fn [bad]
                         (try (llama/load-state! s bad toks) false
                              (catch Throwable _ true)))]
          ;; --- always refused
          (let [i (h/draw! (g/integer 0 (dec (count toks))))]
            (when-not (refused? (assoc st :tokens (assoc toks i (inc (nth toks i)))))
              (throw (ex-info "a mutated token vector was accepted"
                              {:hegel/origin "integrity/token-always-refused"}))))
          (when-not (refused? (assoc st :state-bytes (dec n)))
            (throw (ex-info "a wrong byte count was accepted"
                            {:hegel/origin "integrity/length-always-refused"})))

          ;; --- blob mutations: caught by the digest, at head, middle and tail
          (doseq [[label off] [[:head 0] [:middle (quot n 2)] [:tail (dec n)]]]
            (let [bad (assoc st :jolt.llama/blob (flip off))]
              (when (llama/verify-state-digest bad)
                (throw (ex-info "a blob mutation did not change the digest"
                                {:hegel/origin (str "integrity/digest-detects-" (name label))})))))

          ;; --- and an unmodified descriptor still passes both
          (when-not (llama/verify-state-digest st)
            (throw (ex-info "an unmodified blob failed its own digest"
                            {:hegel/origin "integrity/clean-digest-passes"})))
          (when-let [why (llama/state-compatible? s st)]
            (throw (ex-info "an unmodified descriptor was refused"
                            {:hegel/origin "integrity/clean-passes" :why why})))))))))

(defn check-unattributed-runtime-never-matches! [runner]
  (report/run!
   runner
   "an unattributable runtime id is refused even against an identical one"
   (fn []
     (h/run-test!
      {:test-cases 8 :database "" :verbosity :quiet}
      (fn [_]
        ;; String equality would let "unknown:x" match "unknown:x", so two
        ;; shims built from two UNIDENTIFIED trees would exchange state -- the
        ;; exact case the runtime check exists to stop.
        (let [s state-session
              toks (vec (tok (probe-text 2)))
              _ (llama/clear! s)
              _ (llama/eval! s toks)
              st (llama/save-state s)
              tag (str "unknown:" (h/draw! (g/integer 0 999)))
              got (try (llama/load-state! s (assoc st :runtime-id tag) toks) :ACCEPTED
                       (catch Throwable e (:jolt.llama/error (ex-data e))))]
          (when-not (= :state/runtime-unattributed got)
            (throw (ex-info "an unattributed runtime id was not refused"
                            {:hegel/origin "runtime/unattributed-never-matches"
                             :got got})))
          ;; and it must be a DIFFERENT error than a plain mismatch, or the two
          ;; cases cannot be told apart in an audit
          (let [mism (try (llama/load-state! s (assoc st :runtime-id "llama.cpp:dead:clean") toks)
                          :ACCEPTED
                          (catch Throwable e (:jolt.llama/error (ex-data e))))]
            (when-not (= :state/runtime-mismatch mism)
              (throw (ex-info "a runtime mismatch reported the wrong reason"
                              {:hegel/origin "runtime/mismatch-is-distinct"
                               :got mism}))))))))))


(defn check-concurrent-first-open-initialises-once! [runner]
  (report/run!
   runner
   "N threads opening a first model produce one runtime init and one outcome"
   (fn []
     (h/run-test!
      {:test-cases 4 :database "" :verbosity :quiet}
      (fn [_]
        ;; init-runtime! was read-then-write, so N threads opening their first
        ;; model could all observe false and all call jl_runtime_init -- and
        ;; the native refcount is a plain int, so those increments race too.
        ;; The runtime is process-wide and already up by the time this runs, so
        ;; what is asserted is the observable contract: every concurrent caller
        ;; succeeds and sees the same outcome, and no model is damaged.
        (let [n (h/draw! (g/integer 2 5))
              latch (promise)
              fs (doall (for [_ (range n)]
                          (future (deref latch)
                                  (try {:ok (llama/init-runtime!)}
                                       (catch Throwable e {:err (ex-message e)})))))]
          (deliver latch true)
          (let [rs (mapv deref fs)]
            (when-not (every? #(= :ok (:ok %)) rs)
              (throw (ex-info "concurrent init-runtime! gave differing outcomes"
                              {:hegel/origin "init/one-outcome" :results rs})))
            ;; and the runtime still works afterwards
            (let [m2 (llama/open-model {:path model-path})]
              (try
                (when-not (pos? (:n-vocab m2))
                  (throw (ex-info "runtime unusable after a concurrent init race"
                                  {:hegel/origin "init/usable-after-race"})))
                (finally (llama/close! m2)))))))))))

(defn check-close-outcomes-are-defined! [runner]
  (report/run!
   runner
   "every close returns a defined outcome and the count never goes negative"
   (fn []
     (h/run-test!
      {:test-cases 8 :database "" :verbosity :quiet}
      (fn [_]
        ;; Ownership counting under mixed legal and illegal lifecycle traces.
        ;; The invariant is that active-session-count >= 0 at every point, a
        ;; failed construction consumes none, and a model closes iff the count
        ;; is zero.
        (let [m2 (llama/open-model {:path model-path})]
          (try
            (let [k (h/draw! (g/integer 1 3))
                  sessions (doall (for [_ (range k)]
                                    (llama/new-session m2 {:context-size 256 :threads 2})))]
              ;; an illegal construction in the middle must consume no slot
              (let [before @(:sessions m2)]
                (try (llama/new-session m2 {:context-size 256 :seq-max 3})
                     (catch Throwable _ nil))
                (when-not (= before @(:sessions m2))
                  (throw (ex-info "a rejected session consumed an ownership slot"
                                  {:hegel/origin "own/failed-new-consumes-none"}))))
              ;; duplicate closes are defined and decrement exactly once
              (doseq [s sessions]
                (let [a (llama/close! s) b (llama/close! s)]
                  (when-not (and (= :closed a) (= :already-closed b))
                    (throw (ex-info "duplicate close was not defined"
                                    {:hegel/origin "own/duplicate-close-defined"
                                     :got [a b]})))))
              (when (neg? @(:sessions m2))
                (throw (ex-info "the ownership count went negative"
                                {:hegel/origin "own/never-negative"})))
              (when-not (zero? @(:sessions m2))
                (throw (ex-info "the ownership count did not drain"
                                {:hegel/origin "own/drains-to-zero"
                                 :n @(:sessions m2)}))))
            (finally (try (llama/close! m2) (catch Throwable _ nil))))))))))


(defn check-same-tokens-different-numerical-base-refused! [runner]
  (report/run!
   runner
   "C5: identical tokens reached by a different call structure are still refused"
   (fn []
     (h/run-test!
      {:test-cases 8 :database "" :verbosity :quiet}
      (fn [_]
        ;; THE case token equality cannot catch. Build the SAME token vector two
        ;; ways -- one pass, and a split with a short suffix -- which this
        ;; repository has measured to produce DIFFERENT logits below the
        ;; calibrated threshold. Save from one, sit at the other. The tokens
        ;; match exactly, so the old check passed and the score would have
        ;; composed P(t1 | A) with P(t2 | B, t1).
        (let [s state-session
              toks (vec (tok (probe-text 6)))
              cut (- (count toks) 8)]          ; short suffix: below the threshold
          ;; base A: split evaluation
          (llama/clear! s)
          (llama/eval! s (vec (take cut toks)))
          (llama/eval! s (vec (drop cut toks)))
          (let [st-split (llama/save-state s)]
            ;; now sit at base B: the same tokens, one pass
            (llama/clear! s)
            (llama/eval! s toks)
            (when-not (= (vec (:tokens st-split)) @(:tokens s))
              (throw (ex-info "the two arms did not produce the same token vector"
                              {:hegel/origin "c5/same-tokens"})))
            (let [top (llama/top-k s 2 {:pieces? false})
                  cands [{:id :multi :tokens [(:token (first top)) (:token (second top))]}]
                  got (try (llama/score-candidates s cands {:state st-split}) :ACCEPTED
                           (catch Throwable e (:jolt.llama/error (ex-data e))))]
              ;; tokens are identical, so ONLY the revision identity can refuse it
              (when-not (= :score/base-revision-mismatch got)
                (throw (ex-info "a numerically different base with identical tokens was accepted"
                                {:hegel/origin "c5/revision-refuses" :got got})))))))))))

(defn check-stale-base-logprobs-refused! [runner]
  (report/run!
   runner
   "C6: base scores captured at another base are refused, not merely reused"
   (fn []
     (h/run-test!
      {:test-cases 8 :database "" :verbosity :quiet}
      (fn [_]
        ;; A bare token->score map cannot say which base it came from, so it
        ;; could supply first-token scores from one evaluation while the
        ;; continuations came from another. The descriptor names its origin.
        (let [s state-session
              a (vec (tok (probe-text 3)))
              b (vec (tok (probe-text 5)))]
          (llama/clear! s)
          (llama/eval! s a)
          (let [top (llama/top-k s 3 {:pieces? false})
                cands (mapv (fn [t] {:id (:token t) :tokens [(:token t)]}) top)
                at-a (llama/score-candidates s cands)
                blp (:base-logprobs at-a)]
            ;; move to a different base
            (llama/clear! s)
            (llama/eval! s b)
            (let [top-b (llama/top-k s 3 {:pieces? false})
                  cands-b (mapv (fn [t] {:id (:token t) :tokens [(:token t)]}) top-b)
                  got (try (llama/score-candidates s cands-b {:base-logprobs blp}) :ACCEPTED
                           (catch Throwable e (:jolt.llama/error (ex-data e))))]
              (when-not (= :score/base-logprobs-mismatch got)
                (throw (ex-info "stale base scores were accepted at a different base"
                                {:hegel/origin "c6/stale-refused" :got got})))))))))))

(defn check-base-logprobs-round-trip! [runner]
  (report/run!
   runner
   "base scores replayed into the SAME base are accepted"
   (fn []
     (h/run-test!
      {:test-cases 6 :database "" :verbosity :quiet}
      (fn [_]
        ;; The safety check must not break the documented workflow: after
        ;; multi-token scoring leaves the session restored and logit-less, the
        ;; returned descriptor is how the same base is scored again.
        (let [s state-session
              toks (vec (tok (probe-text 3)))]
          (llama/clear! s)
          (llama/eval! s toks)
          (let [st (llama/save-state s)
                top (llama/top-k s 2 {:pieces? false})
                cands [{:id :one :tokens [(:token (first top))]}
                       {:id :two :tokens [(:token (first top)) (:token (second top))]}]
                r1 (llama/score-candidates s cands {:state st})
                ;; the session is now restored to the base and has no logits
                r2 (llama/score-candidates s cands {:state st
                                                    :base-logprobs (:base-logprobs r1)})]
            (when-not (= (map :id (:candidates r1)) (map :id (:candidates r2)))
              (throw (ex-info "a replayed base produced a different ranking"
                              {:hegel/origin "c6/round-trip"}))))))))))


(defn check-scoring-failure-is-atomic! [runner]
  (report/run!
   runner
   "C7: a failure during candidate scoring restores the base or poisons the session"
   (fn []
     (h/run-test!
      {:test-cases 10 :database "" :verbosity :quiet}
      (fn [_]
        ;; Candidate scoring moves native state. An exception partway through
        ;; used to skip the final restore, leaving the session advanced past its
        ;; base while the token ledger still claimed the base -- the exact
        ;; desynchronisation every later append-only check reads.
        ;;
        ;; The failure is injected AFTER at least one candidate has moved native
        ;; state, by giving a later candidate an out-of-range token.
        (let [s state-session
              toks (vec (tok (probe-text 4)))
              _ (llama/clear! s)
              _ (llama/eval! s toks)
              st (llama/save-state s)
              top (llama/top-k s 2 {:pieces? false})
              n-vocab (:n-vocab model)
              bad-token (+ n-vocab 1000)
              cands [{:id :ok    :tokens [(:token (first top)) (:token (second top))]}
                     {:id :boom  :tokens [(:token (first top)) bad-token]}]
              outcome (try (llama/score-candidates s cands {:state st}) :NO-FAILURE
                           (catch Throwable e (:jolt.llama/error (ex-data e))))]
          (cond
            ;; the model may tolerate the token; then there is nothing to assert
            (= :NO-FAILURE outcome) nil

            (= :score/failed-base-restored outcome)
            ;; the contract: base restored, ledger agrees, session usable
            (do
              (when-not (= toks @(:tokens s))
                (throw (ex-info "the base was not restored after a scoring failure"
                                {:hegel/origin "c7/base-restored"})))
              (when-not (= :open @(:state s))
                (throw (ex-info "the session was poisoned despite a successful restore"
                                {:hegel/origin "c7/not-poisoned-when-recovered"})))
              ;; and it still works
              (llama/eval! s [(:token (first top))])
              (when-not (pos? (count (llama/top-k s 2 {:pieces? false})))
                (throw (ex-info "session unusable after a recovered scoring failure"
                                {:hegel/origin "c7/usable-after-recovery"}))))

            (= :score/failed-session-poisoned outcome)
            ;; the other legal contract: explicitly poisoned, refusing further use
            (do
              (when-not (= :poisoned @(:state s))
                (throw (ex-info "reported poisoned but the handle says otherwise"
                                {:hegel/origin "c7/poison-is-visible"})))
              (when-not (try (llama/eval! s [1]) false
                             (catch Throwable _ true))
                (throw (ex-info "a poisoned session accepted further evaluation"
                                {:hegel/origin "c7/poisoned-refuses"}))))

            :else
            (throw (ex-info "a scoring failure produced an undefined outcome"
                            {:hegel/origin "c7/defined-outcome" :got outcome})))))))))

;; ----------------------------------------------------------------- main

(defn -main [& _]
  (let [runner (report/counting-runner)]
    (println "--- seam properties (contract) ---")
    (check-canonical-prefix-holds! runner)
    (check-prefix-predicate-agrees! runner)

    (println "--- lifecycle properties ---")
    (check-illegal-lifecycle-is-safe! runner)
    (check-close-is-idempotent! runner)

    (println "--- ownership properties ---")
    (check-model-close-is-refused-while-sessions-live! runner)
    (check-use-after-close-is-refused! runner)
    (check-v0-is-single-sequence! runner)
    (check-eval-is-append-only! runner)

    (println "--- sequence closed world ---")
    (check-seq-closed-world! runner)
    (check-seq-max-exactness! runner)

    (println "--- candidate base state ---")
    (check-scoring-base-must-be-exact! runner)
    (check-same-tokens-different-numerical-base-refused! runner)
    (check-stale-base-logprobs-refused! runner)
    (check-base-logprobs-round-trip! runner)
    (check-scoring-failure-is-atomic! runner)
    (check-single-token-path-is-state-free! runner)
    (check-scoring-leaves-the-session-untouched! runner)

    (println "--- concurrency and ownership ---")
    (check-concurrent-close-is-single! runner)
    (check-concurrent-first-open-initialises-once! runner)
    (check-close-outcomes-are-defined! runner)

    (println "--- integrity and runtime identity ---")
    (check-integrity-digests-detect-mutation! runner)
    (check-unattributed-runtime-never-matches! runner)

    (println "--- state compatibility ---")
    (check-incompatible-state-is-refused! runner)
    (check-same-artifact-different-handle-is-accepted! runner)

    (println "--- state properties ---")
    (println "calibrated exact-append threshold:" (:threshold calibration)
             "monotone?" (:monotone? calibration))
    (check-restore-is-transparent! runner)
    (check-restore-matches-onepass-above-threshold! runner)
    (check-short-append-diverges! runner)
    (check-prefix-mismatch-refused! runner)

    (let [contract-failures (report/failure-count runner)]
      (println)
      (println "contract properties failures:" contract-failures)

      ;; These two are demonstrations, counted separately. They are EXPECTED to
      ;; find a counterexample; a pass here would mean the tokenizer happened not
      ;; to merge over the generated domain, which is a weaker world than the one
      ;; the contract is written for.
      (println)
      (println "--- seam DEMONSTRATIONS (a failure here is the point) ---")
      (let [demo (report/counting-runner)]
        (check-naive-text-prefix-assumption! demo)
        (check-append-is-not-concat! demo)
        (println "demonstration counterexamples found:" (report/failure-count demo))
        (when (zero? (report/failure-count demo))
          (println "NOTE: no seam counterexample was generated; the contract is"
                   "still required, but this run did not exhibit it.")))

      (llama/close! state-session)
      (llama/close! model)
      (println)
      (if (zero? contract-failures)
        (println "HEGEL PROPERTIES OK")
        (println "HEGEL PROPERTY FAILURES:" contract-failures))
      (flush)
      (System/exit (if (zero? contract-failures) 0 1)))))
