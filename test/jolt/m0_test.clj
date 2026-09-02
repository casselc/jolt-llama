(ns jolt.m0-test
  "M0: prove the whole path on STOCK Jolt.

      stock jolt -> jolt.ffi -> shim -> libllama -> real logits

  Every number printed with a REF: prefix is compared by the runner against
  native/smoke, which exercises the same model through the same shim with no
  Jolt involved. If the two disagree, the binding is wrong -- that is the whole
  point of keeping an independent reference.")

(require '[jolt.llama :as llama])

(def model-path
  (or (System/getenv "JOLT_LLAMA_MODEL")
      (throw (ex-info "set JOLT_LLAMA_MODEL to a .gguf path" {}))))

(def prompt (or (System/getenv "JOLT_LLAMA_PROMPT") "The capital of France is"))

(def failures (atom []))
(defn check [label ok?]
  (if ok?
    (println "  ok  " label)
    (do (println "  FAIL" label) (swap! failures conj label))))

(println "abi:" (llama/abi-version))
(check "abi is 1" (= 1 (llama/abi-version)))

(llama/with-model [m {:path model-path}]
  (println "model:" (:desc m))
  (println "n_vocab:" (:n-vocab m) "n_ctx_train:" (:n-ctx-train m))
  (check "vocab positive" (pos? (:n-vocab m)))

  (llama/with-session [s m {:context-size 4096 :threads 4}]
    (check "session n-ctx" (= 4096 (:n-ctx s)))

    (let [toks (llama/tokenize m prompt)]
      (println "REF: n_tokens=" (count toks))
      (println "REF: tokens=" (vec (take 16 toks)))
      (check "tokenized" (pos? (count toks)))

      ;; round-trip a token through the vocabulary
      (check "token->piece works" (string? (llama/token->piece m (first toks))))

      (llama/eval! s toks)

      (let [lg (llama/logits s)]
        (println "REF: n_logits=" (count lg))
        (check "logits sized to vocab" (= (count lg) (:n-vocab m))))

      (let [tk (llama/top-k s 10)]
        (println "REF: topk:")
        (doseq [{:keys [token logprob piece]} tk]
          (println (format "REF:   token=%d logprob=%.6f piece=%s" token logprob piece)))
        (check "topk returns 10" (= 10 (count tk)))
        (check "topk descending"
               (= (map :logprob tk) (reverse (sort (map :logprob tk)))))
        ;; log-softmax over the full vocabulary: every logprob must be <= 0
        (check "logprobs are log-probabilities" (every? #(<= (:logprob %) 0.0) tk))
        ;; the single-token accessor must agree with the top-k table
        (let [t0 (:token (first tk))
              direct (llama/token-logprob s t0)]
          (println (format "REF: token_logprob(%d)=%.6f" t0 direct))
          (check "token-logprob agrees with top-k"
                 (< (abs (- direct (:logprob (first tk)))) 1e-5))))

      ;; ---- state round trip
      (let [st (llama/save-state s)]
        (println "REF: state_bytes=" (:state-bytes st))
        (println "REF: state_n_tokens=" (:n-tokens st))
        (check "state has bytes" (pos? (:state-bytes st)))
        (check "state bound to token vector" (= (vec toks) (vec (:tokens st))))

        (let [before (llama/top-k s 5)]
          (llama/clear! s)
          ;; logits must be refused after a clear: nothing has been evaluated
          (check "logits refused after clear"
                 (try (llama/logits s) false
                      (catch Throwable e
                        (= :error/no-logits (:jolt.llama/error (ex-data e))))))

          (llama/load-state! s st)
          ;; a restore alone yields no logits either -- the restored state has
          ;; not produced an output position yet
          (check "logits refused after restore"
                 (try (llama/logits s) false
                      (catch Throwable e
                        (= :error/no-logits (:jolt.llama/error (ex-data e))))))

          ;; token-identity contract
          (check "prefix ok for identical tokens" (llama/token-prefix-ok? st toks))
          (check "prefix ok for a proper extension"
                 (llama/token-prefix-ok? st (concat toks [1 2 3])))
          (check "prefix NOT ok when a token differs"
                 (not (llama/token-prefix-ok? st (assoc (vec toks) 0 (inc (first toks))))))
          (check "prefix NOT ok when shorter"
                 (not (llama/token-prefix-ok? st (butlast toks))))
          (check "load-state! refuses a mismatched prefix"
                 (try (llama/load-state! s st {:for-tokens (assoc (vec toks) 0 (inc (first toks)))})
                      false
                      (catch Throwable e
                        (= :state/prefix-mismatch (:jolt.llama/error (ex-data e))))))
          (check "load-state! accepts an exact extension"
                 (map? (llama/load-state! s st {:for-tokens (concat toks [1])})))

          ;; and after re-evaluating one token the distribution must be usable
          (llama/eval! s [(:token (first before))])
          (check "logits available after restore+eval" (pos? (count (llama/top-k s 3)))))))

    ;; ---- candidate scoring
    (let [cands [{:id :continue :tokens (llama/tokenize m " Paris" {:add-special? false})}
                 {:id :verify   :tokens (llama/tokenize m " London" {:add-special? false})}
                 {:id :split    :tokens (llama/tokenize m " Berlin" {:add-special? false})}]]
      (llama/clear! s)
      (llama/eval! s (llama/tokenize m prompt))
      (let [st (llama/save-state s)
            res (llama/score-candidates s cands {:state st})]
        (println "REF: candidates:")
        (doseq [c (:candidates res)]
          (println (format "REF:   id=%s rank=%d n=%d sum=%.6f mean=%.6f"
                           (name (:id c)) (:rank c) (:n-tokens c)
                           (:logprob-sum c) (:logprob-mean c))))
        (check "all candidates scored" (= 3 (:n-candidates res)))
        (check "candidate identity preserved"
               (= #{:continue :verify :split} (set (map :id (:candidates res)))))
        (check "ranked by logprob-sum descending"
               (= (map :logprob-sum (:candidates res))
                  (reverse (sort (map :logprob-sum (:candidates res))))))
        (check "best is rank 0" (zero? (:rank (:best res))))
        ;; the model should prefer Paris here; a weak but real semantic check
        (check "Paris outranks London" (= :continue (:id (:best res))))
        (check "scoring left the spine intact"
               (= (:base-n-tokens res) (count @(:tokens s))))))))

(println)
(if (empty? @failures)
  (println "M0 OK")
  (do (println "M0 FAILURES:" @failures)
      (System/exit 1)))
