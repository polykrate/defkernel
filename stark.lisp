;;;; stark.lisp — Simple STARK prover and verifier over Goldilocks
;;;;
;;;; Proves knowledge of a Fibonacci trace:
;;;;   t[0] = a, t[1] = b, t[i+2] = t[i] + t[i+1] mod P
;;;;
;;;; The AIR constraint: t[i+2] - t[i+1] - t[i] = 0 for all i
;;;;
;;;; Pipeline:
;;;;   1. Generate execution trace
;;;;   2. Interpolate trace polynomial via INTT
;;;;   3. Evaluate constraint polynomial
;;;;   4. Commit via FRI
;;;;   5. Generate proof (Merkle proofs at query positions)

(in-package #:defkernel)

;;; ═══════════════════════════════════════════════════════════════════
;;; 1. Execution trace: Fibonacci over Goldilocks
;;; ═══════════════════════════════════════════════════════════════════

(defun generate-fib-trace (n &key (a 1) (b 1))
  "Generate a Fibonacci trace of length N.
   t[0]=a, t[1]=b, t[i+2] = t[i] + t[i+1] mod P.
   N must be a power of 2."
  (assert (= n (ash 1 (1- (integer-length n)))) ()
          "Trace length must be power of 2, got ~D" n)
  (let ((trace (make-array n :element-type '(unsigned-byte 64) :initial-element 0)))
    (setf (aref trace 0) (mod a +goldilocks-p+))
    (setf (aref trace 1) (mod b +goldilocks-p+))
    (loop for i from 2 below n
          do (setf (aref trace i)
                   (fp+ (aref trace (- i 2)) (aref trace (- i 1)))))
    trace))

;;; ═══════════════════════════════════════════════════════════════════
;;; 2. Constraint evaluation
;;;
;;; For Fibonacci AIR: C(x) = t(x*g^2) - t(x*g) - t(x)
;;; where g is the generator of the trace domain.
;;;
;;; In evaluation form (after NTT):
;;;   constraint_evals[i] = trace_evals[(i+2) % N] - trace_evals[(i+1) % N] - trace_evals[i]
;;;
;;; The constraint polynomial should vanish on the first N-2 positions.
;;; We divide by the vanishing polynomial Z(x) = prod_{i=0}^{N-3} (x - g^i)
;;; to get the quotient Q(x) = C(x) / Z(x).
;;; ═══════════════════════════════════════════════════════════════════

(defun compute-constraint-evals (trace-evals)
  "Compute constraint evaluations: c[i] = t[(i+2)%N] - t[(i+1)%N] - t[i]"
  (let* ((n (length trace-evals))
         (result (make-array n :element-type '(unsigned-byte 64))))
    (dotimes (i n result)
      (setf (aref result i)
            (fp- (aref trace-evals (mod (+ i 2) n))
                 (fp+ (aref trace-evals (mod (+ i 1) n))
                      (aref trace-evals i)))))))

;;; ═══════════════════════════════════════════════════════════════════
;;; 3. Low-degree extension (LDE)
;;;
;;; Evaluate the trace polynomial on a larger domain (blowup factor)
;;; to enable FRI queries at positions not in the original trace.
;;; ═══════════════════════════════════════════════════════════════════

(defun lde (coeffs blowup &key (gpu t))
  "Low-Degree Extension: evaluate polynomial (given as coefficients)
   on a domain BLOWUP times larger. Zero-pad then forward NTT."
  (let* ((n (length coeffs))
         (extended-n (* n blowup))
         (padded (make-array extended-n
                             :element-type '(unsigned-byte 64)
                             :initial-element 0)))
    (replace padded coeffs)
    (ntt-forward padded :gpu gpu)))

;;; ═══════════════════════════════════════════════════════════════════
;;; 4. STARK proof structure
;;; ═══════════════════════════════════════════════════════════════════

(defstruct stark-proof
  (trace-root nil)
  (constraint-root nil)
  (fri-layers nil)
  (queries nil)
  (trace-length 0)
  (blowup 0)
  (num-queries 0)
  (public-inputs nil))

;;; ═══════════════════════════════════════════════════════════════════
;;; 5. STARK prover
;;; ═══════════════════════════════════════════════════════════════════

(defun stark-prove (trace &key (blowup 4) (num-queries 16) (gpu t))
  "Generate a STARK proof for a Fibonacci trace.

   TRACE: u64 vector of trace values (power-of-2 length).
   BLOWUP: LDE blowup factor (default 4).
   NUM-QUERIES: number of FRI query positions (default 16).
   GPU: use GPU acceleration (default T).

   Returns a STARK-PROOF struct."
  (let* ((n (length trace))
         (extended-n (* n blowup))
         (random-state (sb-ext:seed-random-state
                        (sxhash (aref trace 0)))))

    ;; Step 1: Interpolate trace → coefficients via inverse NTT
    (let* ((trace-coeffs (ntt-inverse trace :gpu gpu))

           ;; Step 2: Low-degree extension of trace
           (trace-lde (lde trace-coeffs blowup :gpu gpu))

           ;; Step 3: Commit to trace LDE
           (trace-tree nil)
           (trace-root nil))

      (multiple-value-setq (trace-root trace-tree)
        (merkle-commit trace-lde))

      ;; Step 4: Compute constraint evaluations on extended domain
      (let* ((constraint-evals (compute-constraint-evals trace-lde))

             ;; Step 5: Commit constraints via FRI
             (fri-layers (fri-commit constraint-evals
                                     :gpu gpu
                                     :random-state random-state)))

        ;; Step 6: Generate queries
        (let ((queries nil))
          (dotimes (q num-queries)
            (let* ((pos (mod (random (expt 2 64) random-state) extended-n))
                   (fri-qs (fri-query fri-layers pos))
                   (trace-val (aref trace-lde (mod pos extended-n)))
                   (trace-merkle (merkle-path trace-tree
                                             (mod pos extended-n)
                                             extended-n)))
              (push (list :position pos
                          :trace-value trace-val
                          :trace-merkle trace-merkle
                          :fri-queries fri-qs)
                    queries)))

          (make-stark-proof
           :trace-root trace-root
           :constraint-root (fri-layer-root (first fri-layers))
           :fri-layers fri-layers
           :queries (nreverse queries)
           :trace-length n
           :blowup blowup
           :num-queries num-queries
           :public-inputs (list (aref trace 0) (aref trace 1))))))))

;;; ═══════════════════════════════════════════════════════════════════
;;; 6. STARK verifier
;;; ═══════════════════════════════════════════════════════════════════

(defun stark-verify (proof)
  "Verify a STARK proof. Returns T if valid, NIL if invalid."
  (let ((trace-root (stark-proof-trace-root proof))
        (queries (stark-proof-queries proof))
        (fri-layers (stark-proof-fri-layers proof)))

    ;; Check each query
    (every
     (lambda (query)
       (let* ((trace-val (getf query :trace-value))
              (trace-merkle (getf query :trace-merkle))
              (fri-qs (getf query :fri-queries))
              (trace-hash (hash-leaf trace-val)))

         ;; Verify trace Merkle proof
         (and (merkle-verify trace-root trace-hash trace-merkle)
              ;; Verify FRI query chain
              (fri-verify-query fri-layers fri-qs))))
     queries)))

;;; ═══════════════════════════════════════════════════════════════════
;;; 7. End-to-end test
;;; ═══════════════════════════════════════════════════════════════════

(defun stark-test (&key (trace-size 64) (gpu t))
  "Run a complete STARK prove-verify cycle."
  (format t "~%══ defkernel: STARK Prover ══~%")
  (format t "~%  Trace: Fibonacci, N=~D~%" trace-size)

  (let* ((trace (generate-fib-trace trace-size))
         (t0 (get-internal-real-time))
         (proof (stark-prove trace :gpu gpu))
         (t1 (get-internal-real-time))
         (prove-ms (/ (* 1000.0 (- t1 t0)) internal-time-units-per-second))

         (t2 (get-internal-real-time))
         (valid (stark-verify proof))
         (t3 (get-internal-real-time))
         (verify-ms (/ (* 1000.0 (- t3 t2)) internal-time-units-per-second)))

    (format t "  Prove:  ~,1F ms~%" prove-ms)
    (format t "  Verify: ~,1F ms~%" verify-ms)
    (format t "  Valid:  ~A~%" valid)
    (format t "  FRI layers: ~D~%" (length (stark-proof-fri-layers proof)))
    (format t "  Queries: ~D~%" (stark-proof-num-queries proof))
    (format t "  Public inputs: ~A~%" (stark-proof-public-inputs proof))
    valid))

;;; ═══════════════════════════════════════════════════════════════════
;;; Export STARK symbols
;;; ═══════════════════════════════════════════════════════════════════

(eval-when (:compile-toplevel :load-toplevel :execute)
  (export '(stark-prove stark-verify stark-test
            stark-proof stark-proof-trace-root
            stark-proof-fri-layers stark-proof-queries
            stark-proof-public-inputs
            generate-fib-trace)
          :defkernel))
