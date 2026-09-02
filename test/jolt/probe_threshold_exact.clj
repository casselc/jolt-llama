(ns jolt.probe-threshold-exact
  "Pin the boundary found by probe-split-threshold, and test whether the shim's
  own n_batch chunking can trip it.

  Sweep A/B established: a decode call of >= 64 tokens agrees bit-for-bit with a
  one-pass evaluation; a decode call of < 64 tokens does not. 64 is the fused
  Gated Delta Net chunk size for this model.

  The shim chunks eval! at n_batch. If a prompt is n_batch*k + r with r < 64,
  the shim ITSELF emits a short trailing decode -- so a plain single eval! call
  would be on the short-kernel path without the caller doing anything wrong.")

(require '[jolt.llama :as llama])

(def model-path (System/getenv "JOLT_LLAMA_MODEL"))

(defn body [n]
  (apply str (for [k (range n)]
               (format "  svc%03d: region=r%d p95=%dms err=%d\n"
                       k (mod k 7) (+ 40 (mod (* k 13) 500)) (mod k 9)))))

(llama/with-model [m {:path model-path}]
  (llama/with-session [s m {:context-size 8192 :threads 4}]
    (let [toks (vec (llama/tokenize m (str "CONTROLLER\n" (body 400))))
          topk (fn [] (into {} (map (juxt :token :logprob)
                                    (llama/top-k s 50 {:pieces? false}))))
          delta (fn [a b] (let [c (filter b (keys a))]
                            (apply max (map #(abs (- (double (a %)) (double (b %)))) c))))
          onepass (fn [len] (llama/clear! s) (llama/eval! s (vec (take len toks))) (topk))
          split (fn [len at]
                  (llama/clear! s)
                  (llama/eval! s (vec (take at toks)))
                  (llama/eval! s (vec (subvec toks at len)))
                  (topk))]
      (println (format "total_tokens=%d n_batch=2048" (count toks)))
      (println)
      (println "EXACT BOUNDARY: split at 256, suffix length 56..72")
      (doseq [k (range 56 73)]
        (let [d (delta (onepass (+ 256 k)) (split (+ 256 k) 256))]
          (println (format "  suffix_len=%3d  %12.10f  %s" k d
                           (if (< d 1e-9) "EXACT" "diverges")))))

      (println)
      (println "SHIM CHUNKING: does eval! of n_batch*k + r (r<64) self-diverge?")
      ;; 2056 = 2048 + 8, so the shim emits decode(2048) then decode(8).
      ;; Compare against the SAME 2056 tokens split into two >=64 chunks.
      (doseq [len [2056 2100 2112 2560]]
        (let [a (onepass len)
              b (split len 1024)                 ; 1024 + (len-1024), both >= 64
              d (delta a b)
              r (mod len 2048)]
          (println (format "  len=%4d (n_batch remainder=%4d)  onepass_vs_balanced_split=%12.10f  %s"
                           len r d (if (< d 1e-9) "EXACT" "DIVERGES")))))
      (println)
      (println "  A DIVERGES row means the shim's own tail chunk took the short")
      (println "  kernel path: caller did nothing wrong, chunking alone did it."))))
