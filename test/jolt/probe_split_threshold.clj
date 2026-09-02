(ns jolt.probe-split-threshold
  "Characterise the split-prefill divergence found by the hegel state property.

  Established by probe-suffix-len:
    * recompute vs recompute            = 0.0          (deterministic)
    * restore+suffix vs one-pass        = 0.1964101791
    * split-prefill vs one-pass         = 0.1964101791  <-- IDENTICAL

  So save/restore is exact. What diverges is evaluating a prompt as two
  llama_decode calls instead of one. M1 measured 0.00000000 for a 691-token
  suffix, so the effect is not universal. This finds the boundary.

  Two sweeps:
    A  vary SUFFIX length at a fixed split point
    B  vary the SPLIT POINT at a fixed total length

  If A shows a threshold and B does not, the cause is the small-ubatch kernel
  selection this model announces at load time (autoregressive / chunked fused
  Gated Delta Net). If B matters too, it is chunk alignment.")

(require '[jolt.llama :as llama])

(def model-path (System/getenv "JOLT_LLAMA_MODEL"))

(defn body [n]
  (apply str (for [k (range n)]
               (format "  svc%03d: region=r%d p95=%dms err=%d\n"
                       k (mod k 7) (+ 40 (mod (* k 13) 500)) (mod k 9)))))

(llama/with-model [m {:path model-path}]
  (llama/with-session [s m {:context-size 8192 :threads 4}]
    (let [toks (vec (llama/tokenize m (str "CONTROLLER\n" (body 400))))
          n    (count toks)
          topk (fn [] (into {} (map (juxt :token :logprob)
                                    (llama/top-k s 50 {:pieces? false}))))
          delta (fn [a b] (let [c (filter b (keys a))]
                            (if (seq c)
                              (apply max (map #(abs (- (double (a %)) (double (b %)))) c))
                              -1.0)))
          ;; one-pass reference for a given prompt length
          onepass (fn [len]
                    (llama/clear! s)
                    (llama/eval! s (vec (take len toks)))
                    (topk))
          ;; same prompt, evaluated as two decode calls at `at`
          split (fn [len at]
                  (llama/clear! s)
                  (llama/eval! s (vec (take at toks)))
                  (llama/eval! s (vec (subvec toks at len)))
                  (topk))]

      (println (format "total_tokens=%d" n))
      (println)
      (println "SWEEP A: split point fixed at 256, suffix length varies")
      (println "  suffix_len   max_dlogprob   exact?")
      (doseq [k [1 2 4 8 16 24 32 48 64 96 128 192 256 384]]
        (when (<= (+ 256 k) n)
          (let [a (onepass (+ 256 k))
                b (split (+ 256 k) 256)
                d (delta a b)]
            (println (format "  %10d   %12.10f   %s" k d (< d 1e-9))))))

      (println)
      (println "SWEEP B: total length fixed at 640, split point varies")
      (doseq [at [64 128 192 256 320 384 448 512 576 639]]
        (when (< at 640)
          (let [a (onepass 640)
                b (split 640 at)
                d (delta a b)]
            (println (format "  split_at=%4d suffix_len=%4d   %12.10f   %s"
                             at (- 640 at) d (< d 1e-9))))))

      (println)
      (println "SWEEP C: does a THIRD decode call change anything, at safe sizes?")
      (let [a (onepass 640)]
        (llama/clear! s)
        (llama/eval! s (vec (take 200 toks)))
        (llama/eval! s (vec (subvec toks 200 420)))
        (llama/eval! s (vec (subvec toks 420 640)))
        (let [d (delta a (topk))]
          (println (format "  three-way split (200/220/220)  %12.10f   %s" d (< d 1e-9))))))))
