(ns jolt.probe-state-cost
  "Where do the 7.6 seconds in save-state go?

  The perf report measured save-state at 7 MiB/s against load-state! at
  280 MiB/s -- a 41x asymmetry for what should be a symmetric memcpy. Three
  candidates:

    1. llama_state_seq_get_size performs a dry-run serialisation, and the
       two-call sizing convention therefore pays for it twice
    2. jl_state_save calls get_size AGAIN inside the write call, so a Jolt-side
       save costs three serialisations and one copy
    3. ffi/read-into! is slow

  These are separable: time the sizing call and the write call independently,
  then time the read-into! that follows.")

(require '[jolt.llama :as llama]
         '[jolt.ffi :as ffi])

(def model-path (System/getenv "JOLT_LLAMA_MODEL"))
(defn now [] (System/currentTimeMillis))

(llama/with-model [m {:path model-path}]
  (llama/with-session [s m {:context-size 8192 :threads 8}]
    (let [toks (vec (take 2792 (llama/tokenize m (apply str (repeat 400 "  svc: p95=40ms err=0\n")))))]
      (println (format "prefilling %d tokens" (count toks)))
      (llama/eval! s toks)
      (let [p (#'llama/session-ptr s)
            save (fn [buf cap np] (#'llama/c-state-save p 0 buf cap np))]
        (dotimes [trial 3]
          (ffi/with-out [np :size_t]
            (let [t0 (now)
                  _  (save ffi/null 0 np)
                  t1 (now)
                  n  (ffi/read np :size_t)]
              (ffi/with-alloc [buf n]
                (let [t2 (now)
                      _  (save buf n np)
                      t3 (now)
                      written (ffi/read np :size_t)
                      blob (byte-array written)
                      t4 (now)
                      _  (ffi/read-into! buf blob 0 written)
                      t5 (now)]
                  (println (format "trial %d  bytes=%d  sizing_call=%d ms  alloc=%d ms  write_call=%d ms  read_into=%d ms"
                                   trial written (- t1 t0) (- t2 t1) (- t3 t2) (- t5 t4)))))))))
        ;; the same thing through the public function, for comparison
        (println)
        (dotimes [trial 5]
          (let [t (now) st (llama/save-state s) d (- (now) t)]
            (println (format "public save-state trial %d = %d ms (%d bytes)"
                             trial d (:state-bytes st)))))
        ;; and with the blob retained, which is what a caller keeping N warm
        ;; domains actually does
        (println)
        (let [held (atom [])]
          (dotimes [trial 5]
            (let [t (now)
                  st (llama/save-state s)
                  d (- (now) t)]
              (swap! held conj st)
              (println (format "retained save-state trial %d = %d ms (holding %d states)"
                               trial d (count @held)))))))))
