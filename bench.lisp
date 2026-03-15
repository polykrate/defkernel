;;;; bench.lisp — NTT and STARK benchmarks
;;;;
;;;; Measures GPU vs CPU performance for NTT at various sizes,
;;;; and end-to-end STARK prover timing with breakdown.
;;;;
;;;; Usage: sbcl --load bench.lisp

(format t "~%Loading modules...~%")
(load (merge-pathnames "defkernel.lisp" *load-truename*))
(load (merge-pathnames "runtime.lisp" *load-truename*))
(load (merge-pathnames "ntt.lisp" *load-truename*))
(load (merge-pathnames "fri.lisp" *load-truename*))
(load (merge-pathnames "stark.lisp" *load-truename*))

(in-package #:defkernel)

;;; ═══════════════════════════════════════════════════════════════════
;;; Timing utilities
;;; ═══════════════════════════════════════════════════════════════════

(defun now-us ()
  "Return current time in microseconds (using gettimeofday)."
  (multiple-value-bind (sec usec) (sb-ext:get-time-of-day)
    (+ (* sec 1000000) usec)))

(defmacro bench (expr &key (warmup 1) (runs 5))
  "Run EXPR multiple times, return median time in milliseconds (microsecond resolution)."
  (let ((times (gensym)) (i (gensym)) (t0 (gensym)) (t1 (gensym)))
    `(let ((,times nil))
       (dotimes (,i ,warmup) ,expr)
       (dotimes (,i ,runs)
         (let ((,t0 (now-us)))
           ,expr
           (let ((,t1 (now-us)))
             (push (max 0.001 (/ (- ,t1 ,t0) 1000.0))
                   ,times))))
       (nth (floor (length ,times) 2) (sort ,times #'<)))))

(defun random-field-vec (n &optional (seed 42))
  "Generate a random vector of N field elements."
  (let ((result (make-array n :element-type '(unsigned-byte 64)))
        (state (sb-ext:seed-random-state seed)))
    (dotimes (i n result)
      (setf (aref result i) (random +goldilocks-p+ state)))))

;;; ═══════════════════════════════════════════════════════════════════
;;; NTT Benchmarks
;;; ═══════════════════════════════════════════════════════════════════

(defun bench-ntt ()
  (format t "~%╔══════════════════════════════════════════════════════════════╗~%")
  (format t "║              NTT Benchmark — Goldilocks Field               ║~%")
  (format t "╚══════════════════════════════════════════════════════════════╝~%")
  (format t "~%  ~8A  ~10A  ~10A  ~10A  ~10A~%"
          "N" "CPU (ms)" "GPU (ms)" "Speedup" "Throughput")
  (format t "  ~8A  ~10A  ~10A  ~10A  ~10A~%"
          "--------" "----------" "----------" "----------" "----------")

  (dolist (log-n '(10 12 14 16))
    (let* ((n (ash 1 log-n))
           (data (random-field-vec n)))

      (let ((cpu-ms (bench (cpu-ntt-forward data) :warmup 1 :runs 3))
            (gpu-ms (if *gpu*
                        (bench (ntt-forward data :gpu t) :warmup 1 :runs 3)
                        nil)))
        (if gpu-ms
            (let ((speedup (if (> gpu-ms 0.001) (/ cpu-ms gpu-ms) 0.0))
                  (mops (if (> gpu-ms 0.001) (/ (* n log-n) gpu-ms 1000.0) 0.0)))
              (format t "  2^~2D     ~8,1F    ~8,1F    ~8,2Fx   ~8,1F Mops/s~%"
                      log-n cpu-ms gpu-ms speedup mops))
            (format t "  2^~2D     ~8,1F    (no GPU)~%"
                    log-n cpu-ms)))))

  ;; Larger sizes (GPU only, CPU too slow)
  (when *gpu*
    (dolist (log-n '(18 20))
      (let* ((n (ash 1 log-n))
             (data (random-field-vec n)))
        (let ((gpu-ms (bench (ntt-forward data :gpu t) :warmup 1 :runs 3)))
          (format t "  2^~2D     (skip)     ~8,1F    —        ~8,1F Mops/s~%"
                  log-n gpu-ms
                  (/ (* n log-n) gpu-ms 1000.0)))))))

;;; ═══════════════════════════════════════════════════════════════════
;;; FRI Fold Benchmark
;;; ═══════════════════════════════════════════════════════════════════

(defun bench-fri-fold ()
  (format t "~%╔══════════════════════════════════════════════════════════════╗~%")
  (format t "║                FRI Fold Benchmark                           ║~%")
  (format t "╚══════════════════════════════════════════════════════════════╝~%")
  (format t "~%  ~8A  ~10A  ~10A  ~10A~%"
          "N" "CPU (ms)" "GPU (ms)" "Speedup")
  (format t "  ~8A  ~10A  ~10A  ~10A~%"
          "--------" "----------" "----------" "----------")

  (dolist (log-n '(10 12 14 16))
    (let* ((n (ash 1 log-n))
           (data (random-field-vec n))
           (alpha (random +goldilocks-p+)))
      (let ((cpu-ms (bench (fri-fold-cpu data alpha) :warmup 1 :runs 3))
            (gpu-ms (if *gpu*
                        (bench (fri-fold-gpu data alpha) :warmup 1 :runs 3)
                        nil)))
        (if gpu-ms
            (format t "  2^~2D     ~8,1F    ~8,1F    ~8,2Fx~%"
                    log-n cpu-ms gpu-ms (/ cpu-ms gpu-ms))
            (format t "  2^~2D     ~8,1F    (no GPU)~%"
                    log-n cpu-ms))))))

;;; ═══════════════════════════════════════════════════════════════════
;;; STARK Prover Benchmark
;;; ═══════════════════════════════════════════════════════════════════

(defun bench-stark ()
  (format t "~%╔══════════════════════════════════════════════════════════════╗~%")
  (format t "║           STARK Prover Benchmark — Fibonacci AIR            ║~%")
  (format t "╚══════════════════════════════════════════════════════════════╝~%")
  (format t "~%  ~8A  ~12A  ~12A  ~12A  ~8A~%"
          "Trace N" "CPU Prove" "GPU Prove" "Verify" "Valid?")
  (format t "  ~8A  ~12A  ~12A  ~12A  ~8A~%"
          "--------" "------------" "------------" "------------" "--------")

  (dolist (n '(64 256 1024 4096))
    (let ((trace (generate-fib-trace n)))
      ;; CPU prove
      (let* ((cpu-ms (bench (stark-prove trace :gpu nil) :warmup 0 :runs 1))
             ;; GPU prove
             (gpu-ms (if *gpu*
                         (bench (stark-prove trace :gpu t) :warmup 0 :runs 1)
                         nil))
             ;; Verify (use GPU proof if available)
             (proof (stark-prove trace :gpu (and *gpu* t)))
             (verify-ms (bench (stark-verify proof) :warmup 0 :runs 3)))
        (format t "  ~6D    ~8,1F ms   ~A  ~8,1F ms   ~A~%"
                n cpu-ms
                (if gpu-ms (format nil "~8,1F ms" gpu-ms) "  (no GPU) ")
                verify-ms
                (if (stark-verify proof) "OK" "FAIL"))))))

;;; ═══════════════════════════════════════════════════════════════════
;;; STARK Breakdown (detailed timing for one size)
;;; ═══════════════════════════════════════════════════════════════════

(defun bench-stark-breakdown (&key (trace-size 1024) (gpu t))
  (format t "~%╔══════════════════════════════════════════════════════════════╗~%")
  (format t "║        STARK Breakdown — N=~D, ~A               ║~%"
          trace-size (if gpu "GPU" "CPU"))
  (format t "╚══════════════════════════════════════════════════════════════╝~%")

  (let* ((trace (generate-fib-trace trace-size))
         (blowup 4)
         (n (length trace))
         (ext-n (* n blowup)))

    ;; Step 1: Inverse NTT (trace → coefficients)
    (let ((intt-ms (bench (ntt-inverse trace :gpu gpu) :warmup 1 :runs 3)))
      (format t "~%  1. Inverse NTT (trace→coeffs, N=~D):    ~8,1F ms~%" n intt-ms)

      ;; Step 2: LDE (coefficients → extended evaluations)
      (let* ((coeffs (ntt-inverse trace :gpu gpu))
             (lde-ms (bench (lde coeffs blowup :gpu gpu) :warmup 1 :runs 3)))
        (format t "  2. LDE (NTT N=~D, blowup=~D):            ~8,1F ms~%" ext-n blowup lde-ms)

        ;; Step 3: Constraint evaluation
        (let* ((trace-lde (lde coeffs blowup :gpu gpu))
               (constraint-ms (bench (compute-constraint-evals trace-lde) :warmup 1 :runs 3)))
          (format t "  3. Constraint evaluation (N=~D):         ~8,1F ms~%" ext-n constraint-ms)

          ;; Step 4: Merkle commitment
          (let ((merkle-ms (bench (merkle-commit trace-lde) :warmup 1 :runs 3)))
            (format t "  4. Merkle commit (N=~D):                 ~8,1F ms~%" ext-n merkle-ms)

            ;; Step 5: FRI commit
            (let* ((constraint-evals (compute-constraint-evals trace-lde))
                   (fri-ms (bench (fri-commit constraint-evals :gpu gpu) :warmup 0 :runs 1)))
              (format t "  5. FRI commit (~D rounds):                ~8,1F ms~%"
                      (max 1 (- (1- (integer-length ext-n)) 2)) fri-ms)

              ;; Total
              (format t "~%  Total pipeline:                          ~8,1F ms~%"
                      (+ intt-ms lde-ms constraint-ms merkle-ms fri-ms))
              (format t "  ─────────────────────────────────────────────~%")
              (format t "  NTT-dominated: ~,0F% of total~%"
                      (* 100.0 (/ (+ intt-ms lde-ms fri-ms)
                                  (+ intt-ms lde-ms constraint-ms merkle-ms fri-ms)))))))))))

;;; ═══════════════════════════════════════════════════════════════════
;;; Comparison with published numbers
;;; ═══════════════════════════════════════════════════════════════════

(defun print-comparison ()
  (format t "~%╔══════════════════════════════════════════════════════════════╗~%")
  (format t "║                 Comparison with Published Numbers           ║~%")
  (format t "╚══════════════════════════════════════════════════════════════╝~%")
  (format t "~%  System             Hardware           2^16 NTT    2^20 NTT~%")
  (format t "  ──────────────────  ─────────────────  ──────────  ──────────~%")
  (format t "  Plonky3 (CPU)      AMD EPYC (AVX-512)  ~~0.3 ms     ~~8 ms~%")
  (format t "  Icicle (CUDA GPU)  NVIDIA RTX 4090      <0.1 ms    ~~0.5 ms~%")
  (format t "  defkernel (OpenCL) Intel UHD iGPU       ← see above ←~%")
  (format t "~%  Notes:~%")
  (format t "  - Icicle Goldilocks NTT merged April 2025, not GPU-optimized yet~%")
  (format t "  - Plonky3 uses AVX-512 SIMD, not GPU~%")
  (format t "  - defkernel runs on ANY OpenCL device (Intel/AMD/NVIDIA)~%")
  (format t "  - Our overhead: Lisp AST->SSA->OpenCL C compilation at macro time~%"))

;;; ═══════════════════════════════════════════════════════════════════
;;; Main
;;; ═══════════════════════════════════════════════════════════════════

(defun run-benchmarks ()
  (format t "~%══════════════════════════════════════════════════════════════~%")
  (format t "  defkernel GPU-Accelerated STARK Prover — Benchmarks~%")
  (format t "══════════════════════════════════════════════════════════════~%")

  (when (gpu-available-p)
    (gpu-init))

  (bench-ntt)
  (bench-fri-fold)
  (bench-stark)

  (when *gpu*
    (bench-stark-breakdown :trace-size 1024 :gpu t))

  (bench-stark-breakdown :trace-size 1024 :gpu nil)

  (print-comparison)

  (when *gpu*
    (gpu-cleanup))

  (format t "~%══════════════════════════════════════════════════════════════~%")
  (format t "  Benchmark complete.~%")
  (format t "══════════════════════════════════════════════════════════════~%"))

(run-benchmarks)
(finish-output)
(sb-ext:exit :code 0)
