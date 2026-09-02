(ns jolt.probe-magnitude
  "Is the three-path divergence ordinary float precision, or a real defect?

  probe-which-path found three self-consistent numerical paths separated by
  ~0.24-0.29 nats in max |delta logprob| over the top 50. That number alone
  cannot distinguish 'tail noise amplified at low probability' from 'the head of
  the distribution actually moved'. This prints the divergence BY RANK, plus the
  top-1 probability under each path, so the answer is visible.

  If rank-0 moves by ~1e-4 and only rank-40 moves by 0.28, this is precision.
  If rank-0 moves by 0.1+, the kernels disagree about the answer.")

(require '[jolt.llama :as llama])

(def model-path (System/getenv "JOLT_LLAMA_MODEL"))

(defn body [n]
  (apply str (for [k (range n)]
               (format "  svc%03d: region=r%d p95=%dms err=%d\n"
                       k (mod k 7) (+ 40 (mod (* k 13) 500)) (mod k 9)))))

(llama/with-model [m {:path model-path}]
  (llama/with-session [s m {:context-size 8192 :threads 4}]
    (let [toks (vec (take 384 (llama/tokenize m (str "CONTROLLER\n" (body 200)))))
          run (fn [chunk]
                (llama/clear! s)
                (doseq [part (partition-all chunk toks)]
                  (llama/eval! s (vec part)))
                (llama/top-k s 50 {:pieces? true}))
          seq1  (run 1)     ; autoregressive fused
          mid   (run 16)    ; intermediate path
          chunk (run 384)]  ; chunked fused
      (println "rank  token      p(seq1)     lp(seq1)   lp(mid)    lp(chunk)   |d mid|   |d chunk|")
      (doseq [i (range 12)]
        (let [a (nth seq1 i)
              lpa (double (:logprob a))
              ;; look up the SAME token under the other two paths
              f (fn [xs] (some #(when (= (:token %) (:token a)) (double (:logprob %))) xs))
              lpb (f mid) lpc (f chunk)]
          (println (format "%4d  %-9s  %.6f  %9.5f  %9s  %9s  %8s  %8s"
                           i (pr-str (:piece a)) (Math/exp lpa) lpa
                           (if lpb (format "%.5f" lpb) "-")
                           (if lpc (format "%.5f" lpc) "-")
                           (if lpb (format "%.5f" (abs (- lpa lpb))) "-")
                           (if lpc (format "%.5f" (abs (- lpa lpc))) "-")))))
      (println)
      (println (format "top-1 agrees across all three paths: %s"
                       (= (:token (first seq1)) (:token (first mid)) (:token (first chunk)))))
      (println (format "top-1 piece: seq1=%s mid=%s chunk=%s"
                       (pr-str (:piece (first seq1))) (pr-str (:piece (first mid)))
                       (pr-str (:piece (first chunk)))))
      (println (format "top-5 token order identical seq1 vs chunk: %s"
                       (= (mapv :token (take 5 seq1)) (mapv :token (take 5 chunk)))))
      ;; where in the ranking does the max delta actually occur?
      (let [ds (keep (fn [a] (let [lpc (some #(when (= (:token %) (:token a))
                                                (double (:logprob %))) chunk)]
                               (when lpc [(:token a) (abs (- (double (:logprob a)) lpc))])))
                     seq1)
            worst (apply max-key second ds)
            rank (first (keep-indexed #(when (= (:token %2) (first worst)) %1) seq1))]
        (println (format "worst |d| = %.6f at rank %d (p = %.6f)"
                         (second worst) rank
                         (Math/exp (double (:logprob (nth seq1 rank))))))))))
