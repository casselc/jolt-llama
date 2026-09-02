(ns jolt.probe-which-path
  "Which evaluation path is the reference, and which is the approximation?

  The 64-token threshold is the fused Gated Delta Net chunk size. Below it the
  recurrence runs sequentially; at or above it llama.cpp uses the chunkwise
  reformulation (a matrix-parallel rewrite of the same recurrence, involving a
  triangular inverse -- mathematically equal, numerically not).

  The SEQUENTIAL recurrence is the definition. The chunked form is the
  optimisation. So the interesting question is not 'does restore diverge' -- it
  does not -- but whether PREFILL is the approximate path.

  Every strategy below evaluates the SAME token vector and is compared against
  fully sequential decode, one token per call. If the chunked strategies cluster
  together and away from sequential, the chunked kernel is a consistent
  approximation of the reference recurrence, and its error is what we measured.")

(require '[jolt.llama :as llama])

(def model-path (System/getenv "JOLT_LLAMA_MODEL"))

(defn body [n]
  (apply str (for [k (range n)]
               (format "  svc%03d: region=r%d p95=%dms err=%d\n"
                       k (mod k 7) (+ 40 (mod (* k 13) 500)) (mod k 9)))))

(llama/with-model [m {:path model-path}]
  (llama/with-session [s m {:context-size 8192 :threads 4}]
    (let [toks (vec (take 384 (llama/tokenize m (str "CONTROLLER\n" (body 200)))))
          topk (fn [] (into {} (map (juxt :token :logprob)
                                    (llama/top-k s 50 {:pieces? false}))))
          delta (fn [a b] (let [c (filter b (keys a))]
                            (apply max (map #(abs (- (double (a %)) (double (b %)))) c))))
          run (fn [chunk]
                (llama/clear! s)
                (doseq [part (partition-all chunk toks)]
                  (llama/eval! s (vec part)))
                (topk))]
      (println (format "tokens=%d" (count toks)))
      (println)
      (println "Reference = fully sequential decode (chunk size 1).")
      (let [seq-ref (run 1)
            results (for [c [1 2 4 8 16 32 63 64 65 96 128 192 384]]
                      [c (run c)])
            results (doall results)]
        (println)
        (println "chunk_size   vs_sequential      vs_onepass(384)")
        (let [onepass (second (last results))]
          (doseq [[c r] results]
            (println (format "%10d   %14.10f   %14.10f"
                             c (delta seq-ref r) (delta onepass r)))))
        (println)
        (println "Reading: rows matching sequential at 0 are on the reference")
        (println "recurrence; rows matching one-pass at 0 are on the chunked")
        (println "kernel. A clean split at 64 means two consistent")
        (println "implementations, not nondeterminism.")))))
