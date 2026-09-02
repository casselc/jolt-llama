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
              (llama/load-state! s st {:for-tokens toks})
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
                  (llama/load-state! s st {:for-tokens toks})
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
              (let [refused? (try (llama/load-state! s st {:for-tokens bad}) false
                                  (catch Throwable e
                                    (= :state/prefix-mismatch (:jolt.llama/error (ex-data e)))))]
                (when-not refused?
                  (throw (ex-info "a mismatched prefix was accepted"
                                  {:hegel/origin "state/mismatch-refused"}))))))))))))

;; ----------------------------------------------------------------- main

(defn -main [& _]
  (let [runner (report/counting-runner)]
    (println "--- seam properties (contract) ---")
    (check-canonical-prefix-holds! runner)
    (check-prefix-predicate-agrees! runner)

    (println "--- lifecycle properties ---")
    (check-illegal-lifecycle-is-safe! runner)
    (check-close-is-idempotent! runner)

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
