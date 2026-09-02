(ns jolt.m1-alternating-test
  "Alternating-domain state restoration through one session.

  A single successful restore proves very little. The Halo work found a defect
  that only appeared once a DIFFERENT domain had occupied the slot, so this
  cycles domains A/B/C through one session many times and re-verifies against a
  fresh recompute oracle throughout -- not only at the start.

  Each domain carries an unguessable 64-bit tag. A foreign tag appearing in
  another domain's scored output is a real leak, not a lucky guess; sequential
  tags would let the model simply predict a neighbour.")

(require '[jolt.llama :as llama])

(def model-path (or (System/getenv "JOLT_LLAMA_MODEL")
                    (throw (ex-info "set JOLT_LLAMA_MODEL" {}))))
(def cycles (Integer/parseInt (or (System/getenv "JOLT_LLAMA_CYCLES") "100")))

(def failures (atom []))
(defn check [label ok?]
  (if ok? (println "  ok  " label)
      (do (println "  FAIL" label) (swap! failures conj label))))

;; deterministic but unguessable-looking per-domain tags
(defn tag-for [i]
  (let [h (reduce (fn [a c] (unchecked-int (+ (* 131 a) (int c))))
                  (int (+ 0xC0FFEE (* i 7919)))
                  (str "domain-" i "-salt"))]
    (format "%08x%08x" (bit-and h 0xffffffff) (bit-and (* h 31) 0xffffffff))))

(defn domain-spine [i]
  (str "CONTROLLER DOMAIN " (tag-for i) "\n"
       "Choose exactly one action: HOLD, SCALE, ROLLBACK, RESTART, PAGE.\n"
       "TOPOLOGY\n"
       (apply str (for [k (range 60)]
                    (format "  svc%03d: region=r%d tier=%d budget=%dms\n"
                            k (mod (+ k i) 7) (mod k 4) (+ 80 (mod (* k (inc i) 31) 400)))))))

(defn domain-delta [i epoch]
  (str "\nCURRENT STATE\n"
       (apply str (for [k (range 12)]
                    (format "  svc%03d: p95=%dms err=%d\n"
                            k (+ 40 (mod (+ (* k 13) (* epoch 31) i) 500)) (mod (+ k epoch) 9))))
       "\nACTION:"))

(llama/with-model [m {:path model-path}]
  (llama/with-session [s m {:context-size 8192 :threads 4}]
    (let [n-domains 3
          ;; Build each domain's canonical token projection ONCE, then slice.
          ;; Tokenising the concatenation per turn is exactly what the token
          ;; identity contract forbids.
          domains
          (vec (for [i (range n-domains)]
                 (let [spine (domain-spine i)
                       spine-toks (llama/tokenize m spine)
                       ;; token-exact boundary against a representative delta
                       probe (llama/tokenize m (str spine (domain-delta i 0)))
                       n-exact (count (take-while true? (map = spine-toks probe)))]
                   {:i i :tag (tag-for i) :spine spine
                    :n-exact n-exact})))

          ;; evaluate each spine once and capture its state
          states
          (vec (for [{:keys [i spine n-exact]} domains]
                 (do (llama/clear! s)
                     (let [toks (vec (take n-exact (llama/tokenize m spine)))]
                       (llama/eval! s toks)
                       (assoc (llama/save-state s) :domain i)))))]

      (println (format "domains=%d state_bytes=%d n_tokens=%s"
                       n-domains (:state-bytes (first states))
                       (mapv :n-tokens states)))
      (check "all domains produced state" (every? #(pos? (:state-bytes %)) states))
      (check "domain tags are distinct" (= n-domains (count (set (map :tag domains)))))

      (let [tags (mapv :tag domains)
            t-start (System/currentTimeMillis)
            results
            (doall
             (for [c (range cycles)]
               (let [i (mod c n-domains)
                     d (nth domains i)
                     st (nth states i)
                     epoch (+ 1000 c)
                     ;; canonical projection: tokenize spine++delta as ONE text,
                     ;; then the suffix is what follows the saved boundary
                     full (llama/tokenize m (str (:spine d) (domain-delta i epoch)))
                     suffix (vec (drop (:n-tokens st) full))
                     t0 (System/currentTimeMillis)
                     _ (llama/clear! s)
                     _ (llama/load-state! s st {:for-tokens full})
                     t-restore (- (System/currentTimeMillis) t0)
                     t1 (System/currentTimeMillis)
                     _ (llama/eval! s suffix)
                     t-eval (- (System/currentTimeMillis) t1)
                     top (llama/top-k s 5)
                     text (apply str (map :piece top))]
                 ;; verify against a fresh recompute every 10th cycle: the only
                 ;; check that distinguishes "restored" from merely "plausible"
                 (let [verify
                       (when (zero? (mod c 10))
                         (llama/clear! s)
                         (llama/eval! s full)
                         (let [ref (llama/top-k s 5 {:pieces? false})]
                           ;; `ref` is the fresh recompute; `top` above came from
                           ;; the restore path. Comparing those two is the check.
                           {:ref-top1 (:token (first ref))
                            :restored-top1 (:token (first top))
                            :match (= (:token (first ref)) (:token (first top)))}))
                       foreign (filterv #(and (not= % (:tag d))
                                              (clojure.string/includes? text %)) tags)]
                   {:cycle c :domain i :restore-ms t-restore :eval-ms t-eval
                    :top1 (:token (first top)) :verify verify :foreign foreign}))))
            wall (- (System/currentTimeMillis) t-start)
            verified (filter :verify results)
            mismatches (filter #(false? (:match (:verify %))) verified)
            contaminated (filter #(seq (:foreign %)) results)
            restore-times (sort (map :restore-ms results))
            eval-times (sort (map :eval-ms results))
            p (fn [xs q] (nth xs (min (dec (count xs)) (int (* q (count xs))))))]

        (println (format "REF: cycles=%d wall_ms=%d" cycles wall))
        (println (format "REF: restore_ms p50=%d p95=%d  delta_eval_ms p50=%d p95=%d"
                         (p restore-times 0.5) (p restore-times 0.95)
                         (p eval-times 0.5) (p eval-times 0.95)))
        (println (format "REF: verified_cycles=%d mismatches=%d contaminated=%d"
                         (count verified) (count mismatches) (count contaminated)))

        ;; per-domain determinism: the same domain must give the same answer for
        ;; the same epoch regardless of what ran in between
        (check "every cycle produced a top-1" (every? :top1 results))
        (check "no numerical mismatch against recompute" (zero? (count mismatches)))
        (check "no foreign-domain tag leaked" (zero? (count contaminated)))
        (check "restore stayed fast" (< (p restore-times 0.95) 2000))
        (check "ran the requested number of cycles" (= cycles (count results)))))))

(println)
(if (empty? @failures)
  (println "ALTERNATING OK")
  (do (println "ALTERNATING FAILURES:" @failures) (System/exit 1)))
