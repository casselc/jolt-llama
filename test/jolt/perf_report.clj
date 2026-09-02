(ns jolt.perf-report
  "Performance report for the operations a controller actually performs.

  Reported as median plus p95 over repeated trials, never a single sample, and
  with the token counts attached so the numbers can be compared against a
  different machine or model rather than taken on faith.

  Measured here: model load, cold prefill, state save, state load, delta append,
  and candidate scoring -- plus state bandwidth, which is what decides whether
  keeping N warm domains is affordable.")

(require '[jolt.llama :as llama])

(def model-path (System/getenv "JOLT_LLAMA_MODEL"))
(def trials (Integer/parseInt (or (System/getenv "JOLT_LLAMA_TRIALS") "7")))

(defn now [] (System/currentTimeMillis))
(defn stat [xs]
  (let [v (vec (sort xs)) n (count v)]
    {:p50 (nth v (quot n 2))
     :p95 (nth v (min (dec n) (int (* 0.95 n))))
     :min (first v) :max (peek v)}))
(defn row [label {:keys [p50 p95 min max]} extra]
  (println (format "  %-26s p50=%6d ms  p95=%6d ms  min=%6d  max=%6d   %s"
                   label p50 p95 min max extra)))

(defn spine-text [n]
  (str "CONTROLLER POLICY v1\n"
       "Choose exactly one action: HOLD, SCALE, ROLLBACK, RESTART, PAGE.\n"
       "TOPOLOGY\n"
       (apply str (for [i (range n)]
                    (format "  svc%03d: region=r%d tier=%d budget=%dms owner=team%d\n"
                            i (mod i 7) (mod i 4) (+ 80 (mod (* i 31) 400)) (mod i 11))))))
(defn delta-text [epoch n]
  (str "\nCURRENT STATE\n"
       (apply str (for [i (range n)]
                    (format "  svc%03d: p95=%dms err=%d cpu=%d%%\n"
                            i (+ 40 (mod (+ (* i 13) (* epoch 31)) 500))
                            (mod (+ i epoch) 9) (+ 20 (mod (+ (* i 7) epoch) 70)))))
       "\nACTION:"))

(println "jolt-llama performance report")
(println (format "trials=%d" trials))

;; ---- model load, measured across separate opens
(let [ts (doall (for [_ (range 3)]
                  (let [t (now) m (llama/open-model {:path model-path})]
                    (let [d (- (now) t)] (llama/close! m) d))))]
  (println)
  (println "MODEL")
  (row "open-model" (stat ts) "cold page cache not controlled"))

(llama/with-model [m {:path model-path}]
  (println (format "  vocab=%d n_ctx_train=%d desc=%s" (:n-vocab m) (:n-ctx-train m) (:desc m)))

  (let [ts (doall (for [_ (range 3)]
                    (let [t (now) s (llama/new-session m {:context-size 8192 :threads 8})]
                      (let [d (- (now) t)] (llama/close! s) d))))]
    (row "new-session (8192 ctx)" (stat ts) "allocates the compute buffer"))

  (llama/with-session [s m {:context-size 8192 :threads 8}]
    (let [spine (spine-text 120)
          delta (delta-text 7 30)
          all   (vec (llama/tokenize m (str spine delta)))
          sp    (vec (llama/tokenize m spine))
          n-exact (count (take-while true? (map = sp all)))
          stable (subvec all 0 n-exact)
          suffix (subvec all n-exact)]

      (println)
      (println (format "WORKLOAD  spine=%d tokens  delta=%d tokens  total=%d tokens"
                       n-exact (count suffix) (count all)))

      ;; ---- cold prefill
      (let [ts (doall (for [_ (range trials)]
                        (do (llama/clear! s)
                            (let [t (now)] (llama/eval! s all) (- (now) t)))))]
        (println)
        (println "PREFILL")
        (row "cold prefill (full)" (stat ts)
             (format "%.2f ms/token" (double (/ (:p50 (stat ts)) (count all))))))

      (let [ts (doall (for [_ (range trials)]
                        (do (llama/clear! s)
                            (let [t (now)] (llama/eval! s stable) (- (now) t)))))]
        (row "spine only" (stat ts)
             (format "%.2f ms/token" (double (/ (:p50 (stat ts)) (count stable))))))

      ;; ---- state save / load
      (llama/clear! s)
      (llama/eval! s stable)
      (let [save-ts (doall (for [_ (range trials)] (let [t (now)] (llama/save-state s) (- (now) t))))
            st (llama/save-state s)
            bytes (:state-bytes st)
            load-ts (doall (for [_ (range trials)]
                             (do (llama/clear! s)
                                 (let [t (now)] (llama/load-state! s st all) (- (now) t)))))
            mbps (fn [ms] (/ (/ bytes 1048576.0) (/ (max 1 ms) 1000.0)))]
        (println)
        (println (format "STATE  %d bytes (%.1f MiB) for %d tokens = %.0f bytes/token"
                         bytes (/ bytes 1048576.0) (:n-tokens st)
                         (double (/ bytes (:n-tokens st)))))
        (row "save-state" (stat save-ts) (format "%.0f MiB/s" (mbps (:p50 (stat save-ts)))))
        (row "load-state!" (stat load-ts) (format "%.0f MiB/s" (mbps (:p50 (stat load-ts)))))

        ;; ---- delta append after restore
        (let [ts (doall (for [_ (range trials)]
                          (do (llama/clear! s)
                              (llama/load-state! s st all)
                              (let [t (now)] (llama/eval! s suffix) (- (now) t)))))]
          (println)
          (println "WARM PATH")
          (row "delta append" (stat ts)
               (format "%d tokens, %.2f ms/token" (count suffix)
                       (double (/ (:p50 (stat ts)) (count suffix))))))

        (let [ts (doall (for [_ (range trials)]
                          (do (llama/clear! s)
                              (let [t (now)]
                                (llama/load-state! s st all)
                                (llama/eval! s suffix)
                                (- (now) t)))))
              cold (stat (doall (for [_ (range trials)]
                                  (do (llama/clear! s)
                                      (let [t (now)] (llama/eval! s all) (- (now) t))))))]
          (row "restore + delta (total)" (stat ts)
               (format "vs %d ms cold = %.2fx"
                       (:p50 cold) (double (/ (:p50 cold) (max 1 (:p50 (stat ts)))))))))

      ;; ---- candidate scoring
      (llama/clear! s)
      (llama/eval! s all)
      (let [st (llama/save-state s)
            ;; VERIFIED encodings, not truncated ones. (take 1 (tokenize " ROLLBACK"))
            ;; produced a one-token fixture out of a three-token label, so the
            ;; "single-token candidate scoring" row measured a fragment and was
            ;; not evidence about the controller ABI at all -- the same defect
            ;; issue #8 closed in the canary, left behind here.
            action-words ["hold" "scale" "rollback" "restart" "page"]
            enc (into {} (for [a action-words]
                           [a (vec (llama/tokenize m (str " " a) {:add-special? false}))]))
            bad (into {} (filter (fn [[_ t]] (not= 1 (count t))) enc))
            _ (when (seq bad)
                (println "  NOTE: these encodings are not single-token under this"
                         "model, so the single-token row below would be synthetic:")
                (println "       " (pr-str (into {} (for [[a t] bad] [a (count t)])))))
            single (vec (for [a action-words]
                          {:id (keyword a) :tokens (enc a)}))
            multi  (vec (for [a ["HOLD" "SCALE" "ROLLBACK" "RESTART" "PAGE"]]
                          {:id (keyword (.toLowerCase a))
                           :tokens (vec (llama/tokenize m (str " " a) {:add-special? false}))}))
            t1 (doall (for [_ (range trials)]
                        (let [t (now)] (llama/score-candidates s single {:state st}) (- (now) t))))
            ;; feed the base log-probabilities forward: after the first call
            ;; scored multi-token candidates, the session holds restored state
            ;; with no logits, exactly as load-state! promises
            base-lp (:base-logprobs (llama/score-candidates s multi {:state st}))
            t2 (doall (for [_ (range trials)]
                        (let [t (now)]
                          (llama/score-candidates s multi {:state st :base-logprobs base-lp})
                          (- (now) t))))]
        (println)
        (println "CANDIDATE SCORING (5 candidates)")
        (row "single-token candidates" (stat t1) "no evaluation: read from base logits")
        (row "multi-token candidates" (stat t2)
             (format "%d total tokens, restores between candidates"
                     (reduce + (map (comp count :tokens) multi))))))))

(println)
(println "PERF OK")
