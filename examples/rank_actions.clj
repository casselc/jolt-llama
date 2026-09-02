;; Runnable minimal example: a controller choosing among a FIXED set of actions.
;;
;;   jolt run examples/rank_actions.clj
;;
;; Needs JOLT_LLAMA_LIB and JOLT_LLAMA_MODEL in the environment. No path in this
;; file is specific to any machine.
;;
;; The shape being demonstrated is the point: trusted code builds the legal
;; domain, the model only ranks it, and nothing the model produces can widen the
;; set of things that can happen. There is no sampler and no free-form text.

(require '[jolt.llama :as llama])

(def model-path
  (or (System/getenv "JOLT_LLAMA_MODEL")
      (throw (ex-info "set JOLT_LLAMA_MODEL to a .gguf file" {}))))

;; The legal domain. Finite, closed, and written down by us -- not by the model.
(def actions [:hold :scale :rollback :restart :page])

(def spine
  (str "CONTROLLER POLICY v1\n"
       "Choose exactly one action: HOLD, SCALE, ROLLBACK, RESTART, PAGE.\n"
       "TOPOLOGY\n"
       "  api: region=r1 tier=0 budget=120ms\n"
       "  db:  region=r1 tier=1 budget=400ms\n"
       "  cdn: region=r2 tier=2 budget=80ms\n"))

(defn delta [state-lines]
  (str "\nCURRENT STATE\n" state-lines "\nACTION:"))

(llama/with-model [m {:path model-path}]
  (llama/with-session [s m {:context-size 4096 :threads 4}]

    ;; Single-token candidates keep the scores exactly comparable. See
    ;; docs/EXACTNESS.md for why equal-length candidates matter.
    (let [candidates
          (vec (for [a actions]
                 (let [tks (llama/tokenize m (str " " (.toUpperCase (name a)))
                                           {:add-special? false})]
                   {:id a :tokens (vec (take 1 tks))})))]

      (doseq [[label lines]
              [["healthy"  "  api: p95=95ms err=0\n  db: p95=210ms err=0\n  cdn: p95=40ms err=0\n"]
               ["degraded" "  api: p95=780ms err=41\n  db: p95=1900ms err=88\n  cdn: p95=42ms err=0\n"]]]

        ;; ONE tokenization of the whole prompt. Building the token vector from
        ;; separately-tokenized pieces is exactly what the token-identity
        ;; contract forbids, because BPE merges across the seam.
        (let [tokens (llama/tokenize m (str spine (delta lines)))]
          (llama/clear! s)
          (llama/eval! s tokens)

          (let [st  (llama/save-state s)
                res (llama/score-candidates s candidates {:state st})]
            (println)
            (println "situation:" label)
            (println "  convention:" (:convention res)
                     " comparable:" (:homogeneous? res))
            (doseq [c (:candidates res)]
              (println (format "  %d. %-9s logprob=%9.5f  p=%.4f"
                               (inc (:rank c)) (name (:id c))
                               (:logprob-sum c) (Math/exp (double (:logprob-sum c))))))
            ;; The decision is ours. The model ranked; trusted policy selects.
            (println "  selected:" (name (:id (:best res))))))))))

(println)
(println "Note: the ranking above is what a 0.8B model with no controller")
(println "fine-tuning produces. The point of the example is the SHAPE -- a closed")
(println "domain, exact comparability, and a selection made by trusted code --")
(println "not the quality of the judgement.")
