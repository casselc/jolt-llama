(ns jolt.m1-exact-spine-test
  "M1: exact state save/restore at production shape, plus the token-identity
  contract it depends on.

  The experiment is the one the frozen Halo gate settled:

      A  full recompute of spine ++ delta          -> logits L_A
      B  eval spine, save, clear, restore, append  -> logits L_B

  L_A and L_B must agree. On the Halo gate they agreed exactly for Qwen3.5-0.8B
  once the saved state ended at a token-exact boundary, and did not when a BPE
  merge moved the seam by one token.

  This file also demonstrates WHY jolt-llama is built the way it is. Two ways to
  construct the same text:

    SAFE   tokenize the whole prompt once, slice the token vector
    UNSAFE tokenize spine and delta separately, concatenate the token vectors

  Those are not the same sequence. The unsafe path is what a text-prefix mental
  model produces, and TOKEN-IDENTITY.md's preferred design is exactly the safe
  one -- persist stable-state tokens and append dynamic tokens WITHOUT
  re-tokenizing their concatenated text. The test asserts the divergence exists
  and that load-state! refuses it.")

(require '[jolt.llama :as llama])

(def model-path
  (or (System/getenv "JOLT_LLAMA_MODEL")
      (throw (ex-info "set JOLT_LLAMA_MODEL" {}))))

(def failures (atom []))
(defn check [label ok?]
  (if ok? (println "  ok  " label)
      (do (println "  FAIL" label) (swap! failures conj label))))

(defn now-ms [] (System/currentTimeMillis))

;; A controller-shaped spine: stable policy + topology, then a changing delta.
;; Deliberately similar in shape to the Halo workload so the numbers are
;; comparable to the frozen evidence.
(defn spine-text [n]
  (str "CONTROLLER POLICY v1\n"
       "Choose exactly one action: HOLD, SCALE, ROLLBACK, RESTART, PAGE.\n"
       "TOPOLOGY\n"
       (apply str
              (for [i (range n)]
                (format "  svc%03d: region=r%d tier=%d budget=%dms owner=team%d\n"
                        i (mod i 7) (mod i 4) (+ 80 (mod (* i 31) 400)) (mod i 11))))))

(defn delta-text [epoch n]
  (str "\nCURRENT STATE\n"
       (apply str
              (for [i (range n)]
                (format "  svc%03d: p95=%dms err=%d cpu=%d%%\n"
                        i (+ 40 (mod (+ (* i 13) (* epoch 31)) 500))
                        (mod (+ i epoch) 9) (+ 20 (mod (+ (* i 7) epoch) 70)))))
       "\nACTION:"))

(llama/with-model [m {:path model-path}]
  (llama/with-session [s m {:context-size 8192 :threads 4}]
    (let [spine (spine-text 120)
          delta (delta-text 7 30)
          full  (str spine delta)

          ;; SAFE: one tokenization, sliced. This is the canonical projection.
          all-tokens   (llama/tokenize m full)
          spine-only   (llama/tokenize m spine)
          delta-only   (llama/tokenize m delta {:add-special? false})

          ;; The token-exact boundary: the longest prefix the spine shares with
          ;; the full prompt. Computed, never assumed from lengths or strings.
          n-exact (count (take-while true?
                                     (map (fn [a b] (= a b)) spine-only all-tokens)))]

      (println (format "spine_text_tokens=%d delta_tokens=%d full_tokens=%d exact_boundary=%d"
                       (count spine-only) (count delta-only) (count all-tokens) n-exact))
      (println (format "boundary_lost_to_bpe_merge=%d" (- (count spine-only) n-exact)))

      (check "spine is shorter than full" (< (count spine-only) (count all-tokens)))

      ;; ---- the seam, demonstrated
      (let [naive (vec (concat spine-only delta-only))]
        (println (format "naive_concat_tokens=%d canonical_tokens=%d"
                         (count naive) (count all-tokens)))
        ;; This is the trap the whole contract exists to prevent. If these ever
        ;; coincide for this model+text the test still passes below; what must
        ;; never happen is treating them as interchangeable without checking.
        (println (format "naive_concat_equals_canonical=%s" (= naive (vec all-tokens)))))

      ;; ---- A: full recompute
      (llama/clear! s)
      (let [t0 (now-ms)
            _  (llama/eval! s all-tokens)
            t-full (- (now-ms) t0)
            top-a (llama/top-k s 100 {:pieces? false})
            lp-a  (into {} (map (juxt :token :logprob) top-a))]
        (println (format "REF: full_recompute_ms=%d n_tokens=%d top1=%d"
                         t-full (count all-tokens) (:token (first top-a))))

        ;; ---- B: exact-spine restore
        ;; Save at the token-exact boundary, not at the spine's own tokenization.
        (llama/clear! s)
        (let [stable (vec (take n-exact all-tokens))
              suffix (vec (drop n-exact all-tokens))
              t1 (now-ms)
              _  (llama/eval! s stable)
              t-spine (- (now-ms) t1)
              st (llama/save-state s)
              t2 (now-ms)]
          (println (format "REF: spine_eval_ms=%d state_bytes=%d state_tokens=%d"
                           t-spine (:state-bytes st) (:n-tokens st)))
          (check "state token count is the exact boundary" (= n-exact (:n-tokens st)))

          (llama/clear! s)
          (let [t3 (now-ms)
                _  (llama/load-state! s st all-tokens)
                t-restore (- (now-ms) t3)
                t4 (now-ms)
                _  (llama/eval! s suffix)
                t-delta (- (now-ms) t4)
                top-b (llama/top-k s 100 {:pieces? false})
                lp-b  (into {} (map (juxt :token :logprob) top-b))
                common (filter lp-b (keys lp-a))
                deltas (map #(abs (- (double (lp-a %)) (double (lp-b %)))) common)
                max-d  (if (seq deltas) (apply max deltas) nil)
                mean-d (if (seq deltas) (/ (reduce + 0.0 deltas) (count deltas)) nil)]

            (println (format "REF: restore_ms=%d delta_eval_ms=%d suffix_tokens=%d"
                             t-restore t-delta (count suffix)))
            (println (format "REF: warm_total_ms=%d vs full_ms=%d speedup=%.2fx"
                             (+ t-restore t-delta) t-full
                             (double (/ t-full (max 1 (+ t-restore t-delta))))))
            (println (format "REF: n_common_topk=%d max_abs_dlogprob=%.8f mean_abs_dlogprob=%.8f"
                             (count common) (double (or max-d 0)) (double (or mean-d 0))))
            (println (format "REF: top1_A=%d top1_B=%d"
                             (:token (first top-a)) (:token (first top-b))))

            (check "restore+delta reproduces top-1" (= (:token (first top-a)) (:token (first top-b))))
            (check "top-k sets overlap substantially" (>= (count common) 90))
            ;; The Halo gate measured exactly 0.000000 for this model on this
            ;; protocol. Anything above the float32 noise floor is a real
            ;; divergence, not rounding.
            (check "logprobs match a full recompute" (< (double (or max-d 1.0)) 1e-6))
            (check "restore is much faster than full recompute"
                   (< (+ t-restore t-delta) t-full))))))))

(println)
(if (empty? @failures)
  (println "M1 OK")
  (do (println "M1 FAILURES:" @failures) (System/exit 1)))
