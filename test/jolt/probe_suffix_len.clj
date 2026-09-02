(ns jolt.probe-suffix-len
  "Why does a 1-token suffix after restore not match a full recompute exactly,
  when M1's многotoken suffix matched at 0.00000000?

  Hypothesis: llama.cpp selects a different Gated Delta Net kernel by ubatch
  size. The load log for this model announces both:

      sched_reserve: fused Gated Delta Net (autoregressive) enabled
      sched_reserve: fused Gated Delta Net (chunked) enabled

  A restore followed by ONE token decodes through the autoregressive path. The
  same token reached inside a full prefill decodes through the chunked path.
  Those are different kernels over the same math, so they are allowed to differ
  in the last float32 bits -- while a restore + LONG suffix uses the chunked
  path in both arms and agrees bit for bit.

  If that is what is happening, the delta should depend on SUFFIX LENGTH, not on
  whether a restore occurred. This sweeps suffix length to find out.")

(require '[jolt.llama :as llama])

(def model-path (System/getenv "JOLT_LLAMA_MODEL"))

(llama/with-model [m {:path model-path}]
  (llama/with-session [s m {:context-size 4096 :threads 4}]
    (let [text (str "CONTROLLER\n"
                    (apply str (for [k (range 24)]
                                 (format "  svc%03d: p95=%dms\n" k (+ 40 (* k 7))))))
          base (llama/tokenize m text)
          ;; a fixed, plausible continuation so every arm appends the same tokens
          cont (llama/tokenize m " svc024: p95=208ms err=0 cpu=41%\n" {:add-special? false})]
      (println (format "base_tokens=%d cont_tokens=%d" (count base) (count cont)))
      (println)
      (println "suffix_len  restore_vs_recompute_max_dlogprob   top1_match")
      (doseq [k (range 1 (inc (min 8 (count cont))))]
        (let [suffix (vec (take k cont))
              full   (vec (concat base suffix))]
          ;; arm A: full recompute in one prefill
          (llama/clear! s)
          (llama/eval! s full)
          (let [a (llama/top-k s 50 {:pieces? false})
                lpa (into {} (map (juxt :token :logprob) a))]
            ;; arm B: eval base, save, clear, restore, append suffix
            (llama/clear! s)
            (llama/eval! s base)
            (let [st (llama/save-state s)]
              (llama/clear! s)
              (llama/load-state! s st {:for-tokens full})
              (llama/eval! s suffix)
              (let [b (llama/top-k s 50 {:pieces? false})
                    lpb (into {} (map (juxt :token :logprob) b))
                    common (filter lpb (keys lpa))
                    d (if (seq common)
                        (apply max (map #(abs (- (double (lpa %)) (double (lpb %)))) common))
                        -1.0)]
                (println (format "%10d  %34.10f   %s"
                                 k d (= (:token (first a)) (:token (first b))))))))))

      (println)
      (println "CONTROL: no restore at all, just re-run the same full prompt twice")
      (let [full (vec (concat base (take 1 cont)))]
        (llama/clear! s)
        (llama/eval! s full)
        (let [a (llama/top-k s 50 {:pieces? false})
              lpa (into {} (map (juxt :token :logprob) a))]
          (llama/clear! s)
          (llama/eval! s full)
          (let [b (llama/top-k s 50 {:pieces? false})
                lpb (into {} (map (juxt :token :logprob) b))
                common (filter lpb (keys lpa))]
            (println (format "  recompute-vs-recompute max_dlogprob=%.10f"
                             (apply max (map #(abs (- (double (lpa %)) (double (lpb %)))) common)))))))

      (println)
      (println "CONTROL: split prefill WITHOUT any save/restore (base, then suffix)")
      (let [full (vec (concat base (take 1 cont)))]
        (llama/clear! s)
        (llama/eval! s full)
        (let [a (llama/top-k s 50 {:pieces? false})
              lpa (into {} (map (juxt :token :logprob) a))]
          (llama/clear! s)
          (llama/eval! s base)
          (llama/eval! s (vec (take 1 cont)))
          (let [b (llama/top-k s 50 {:pieces? false})
                lpb (into {} (map (juxt :token :logprob) b))
                common (filter lpb (keys lpa))]
            (println (format "  onepass-vs-split max_dlogprob=%.10f"
                             (apply max (map #(abs (- (double (lpa %)) (double (lpb %)))) common))))
            (println "  If this is ALSO nonzero, the divergence belongs to ubatch")
            (println "  splitting, NOT to state save/restore.")))))))
