;;;; test-determinism.lisp — GPU determinism edge-case stress tests
;;;;
;;;; Verifies that the Goldilocks field arithmetic on GPU produces
;;;; bit-exact results matching the CPU reference, especially on
;;;; values that trigger overflow/underflow/reduction edge cases.
;;;;
;;;; Usage: sbcl --load test-determinism.lisp

(load (merge-pathnames "cl-gpu.lisp" *load-truename*))
(load (merge-pathnames "runtime.lisp" *load-truename*))

(in-package #:cl-gpu)

;;; Kernels

(defkernel *k-add* ((a :vec) (b :vec) (out :out)) (f+ a b))
(defkernel *k-sub* ((a :vec) (b :vec) (out :out)) (f- a b))
(defkernel *k-mul* ((a :vec) (b :vec) (out :out)) (f* a b))
(defkernel *k-sq*  ((x :vec) (out :out)) (f* x x))
(defkernel *k-chain* ((a :vec) (b :vec) (out :out))
  (f+ (f* a a) (f* b b)))

;;; CPU reference (bignum-exact)

(defconstant +P+ (+ (- (expt 2 64) (expt 2 32)) 1))
(defun ref-add (a b) (mod (+ a b) +P+))
(defun ref-sub (a b) (mod (- a b) +P+))
(defun ref-mul (a b) (mod (* a b) +P+))

;;; Test infrastructure

(defvar *pass* 0)
(defvar *fail* 0)
(defvar *details* nil)

(defun check (name gpu-vec cpu-vec)
  "Compare GPU result vector with CPU reference vector."
  (let ((ok (equalp gpu-vec cpu-vec)))
    (if ok
        (progn (incf *pass*) (format t "  ~48A PASS~%" name))
        (progn
          (incf *fail*)
          (format t "  ~48A FAIL~%" name)
          (let ((n (length gpu-vec)))
            (dotimes (i (min n 8))
              (unless (eql (aref gpu-vec i) (aref cpu-vec i))
                (format t "    [~D] GPU=~D  CPU=~D  diff=~D~%"
                        i (aref gpu-vec i) (aref cpu-vec i)
                        (- (aref gpu-vec i) (aref cpu-vec i))))))
          (push name *details*)))
    ok))

(defun to-u64-vec (list)
  (make-array (length list)
              :element-type '(unsigned-byte 64)
              :initial-contents list))

(defun cpu-map (fn a b)
  (map 'vector fn a b))

;;; Edge case vectors

(defparameter *epsilon* (1- (expt 2 32)))          ; 2^32 - 1 = 0xFFFFFFFF
(defparameter *half-p* (floor +P+ 2))

(defparameter *edge-values*
  (list
   0                                                ; zero
   1                                                ; unit
   2
   *epsilon*                                        ; ε = 2^32 - 1
   (expt 2 32)                                      ; ε + 1
   (1+ (expt 2 32))                                 ; ε + 2
   (1- (expt 2 48))                                 ; large sub-P
   (expt 2 48)                                      ; power of 2
   (1- (expt 2 63))                                 ; just below 2^63
   (expt 2 63)                                      ; 2^63 (half of u64 range)
   *half-p*                                         ; P/2
   (1+ *half-p*)                                    ; P/2 + 1
   (- +P+ *epsilon*)                                ; P - ε
   (- +P+ (expt 2 32))                              ; P - 2^32
   (- +P+ 2)                                        ; P - 2
   (- +P+ 1)                                        ; P - 1 (maximum field element)
   ))

(defun run-tests ()
  (setf *pass* 0 *fail* 0 *details* nil)

  (format t "~%╔══════════════════════════════════════════════════════╗~%")
  (format t   "║  CL-GPU: Determinism Edge-Case Tests (GPU)          ║~%")
  (format t   "╚══════════════════════════════════════════════════════╝~%")

  (unless (gpu-available-p)
    (format t "~%No OpenCL device. Skipping.~%")
    (finish-output)
    (sb-ext:exit :code 1))

  (gpu-init)
  (format t "~%")

  (let* ((vals *edge-values*)
         (n (length vals))
         (a (to-u64-vec vals))
         (b (to-u64-vec (reverse vals))))

    ;; ── 1. Addition edge cases ─────────────────────────────────
    (format t "=== fp_add edge cases ===~%")

    ;; a + b for all edge pairs
    (check "add: edge × edge(rev)"
           (gpu-map *k-add* :a a :b b)
           (cpu-map #'ref-add a b))

    ;; a + 0 = a (identity)
    (check "add: x + 0 = x"
           (gpu-map *k-add* :a a :b (to-u64-vec (make-list n :initial-element 0)))
           a)

    ;; a + (P-1-a) = P-1 for all a (complement)
    (let ((complements (to-u64-vec (mapcar (lambda (v) (- +P+ 1 v)) vals))))
      (check "add: a + (P-1-a) = P-1"
             (gpu-map *k-add* :a a :b complements)
             (to-u64-vec (make-list n :initial-element (- +P+ 1)))))

    ;; (P-1) + (P-1) = P-2 (double overflow into mod)
    (let ((pm1 (to-u64-vec (make-list n :initial-element (- +P+ 1)))))
      (check "add: (P-1) + (P-1) = P-2"
             (gpu-map *k-add* :a pm1 :b pm1)
             (to-u64-vec (make-list n :initial-element (- +P+ 2)))))

    ;; (P-1) + 1 = 0 (wrap to zero)
    (let ((pm1 (to-u64-vec (make-list n :initial-element (- +P+ 1))))
          (one (to-u64-vec (make-list n :initial-element 1))))
      (check "add: (P-1) + 1 = 0"
             (gpu-map *k-add* :a pm1 :b one)
             (to-u64-vec (make-list n :initial-element 0))))

    ;; ε + (P-ε) = P ≡ 0 (exact wrap to zero)
    (let ((eps-vec (to-u64-vec (make-list n :initial-element *epsilon*)))
          (comp   (to-u64-vec (make-list n :initial-element (- +P+ *epsilon*)))))
      (check "add: ε + (P-ε) = 0  (exact wrap)"
             (gpu-map *k-add* :a eps-vec :b comp)
             (to-u64-vec (make-list n :initial-element 0))))

    ;; (P-ε-1) + ε = P-1 (just below wrap)
    (let ((base (to-u64-vec (make-list n :initial-element (- +P+ *epsilon* 1))))
          (eps-vec (to-u64-vec (make-list n :initial-element *epsilon*))))
      (check "add: (P-ε-1) + ε = P-1  (just below wrap)"
             (gpu-map *k-add* :a base :b eps-vec)
             (to-u64-vec (make-list n :initial-element (- +P+ 1)))))

    ;; ── 2. Subtraction edge cases ──────────────────────────────
    (format t "~%=== fp_sub edge cases ===~%")

    ;; a - a = 0
    (check "sub: x - x = 0"
           (gpu-map *k-sub* :a a :b a)
           (to-u64-vec (make-list n :initial-element 0)))

    ;; 0 - a = P - a (additive inverse)
    (let ((zeros (to-u64-vec (make-list n :initial-element 0))))
      (check "sub: 0 - x = P - x  (inverse)"
             (gpu-map *k-sub* :a zeros :b a)
             (cpu-map #'ref-sub zeros a)))

    ;; 1 - 2 = P - 1 (underflow wrap)
    (let ((ones (to-u64-vec (make-list n :initial-element 1)))
          (twos (to-u64-vec (make-list n :initial-element 2))))
      (check "sub: 1 - 2 = P-1"
             (gpu-map *k-sub* :a ones :b twos)
             (to-u64-vec (make-list n :initial-element (- +P+ 1)))))

    ;; a - b for mixed edge values
    (check "sub: edge × edge(rev)"
           (gpu-map *k-sub* :a a :b b)
           (cpu-map #'ref-sub a b))

    ;; ── 3. Multiplication edge cases ───────────────────────────
    (format t "~%=== fp_mul edge cases ===~%")

    ;; a * 1 = a (identity)
    (let ((ones (to-u64-vec (make-list n :initial-element 1))))
      (check "mul: x * 1 = x"
             (gpu-map *k-mul* :a a :b ones)
             a))

    ;; a * 0 = 0
    (let ((zeros (to-u64-vec (make-list n :initial-element 0))))
      (check "mul: x * 0 = 0"
             (gpu-map *k-mul* :a a :b zeros)
             zeros))

    ;; a * b for all edge pairs
    (check "mul: edge × edge(rev)"
           (gpu-map *k-mul* :a a :b b)
           (cpu-map #'ref-mul a b))

    ;; x^2 for all edge values (triggers worst-case reduction)
    (check "mul: x^2 for edge values"
           (gpu-map *k-sq* :x a)
           (map 'vector (lambda (v) (ref-mul v v)) a))

    ;; (P-1)^2 = 1 (Fermat-adjacent)
    (let ((pm1 (to-u64-vec (make-list n :initial-element (- +P+ 1)))))
      (check "mul: (P-1)^2 = 1"
             (gpu-map *k-sq* :x pm1)
             (to-u64-vec (make-list n :initial-element 1))))

    ;; ε * ε (tests hi_l * EPSILON reduction)
    (let ((eps-vec (to-u64-vec (make-list n :initial-element *epsilon*))))
      (check "mul: ε × ε"
             (gpu-map *k-sq* :x eps-vec)
             (to-u64-vec (make-list n :initial-element (ref-mul *epsilon* *epsilon*)))))

    ;; 2^32 * 2^32 = 2^64 mod P = ε (triggers hi=1, lo=0)
    (let ((pow32 (to-u64-vec (make-list n :initial-element (expt 2 32)))))
      (check "mul: 2^32 × 2^32 = ε"
             (gpu-map *k-sq* :x pow32)
             (to-u64-vec (make-list n :initial-element *epsilon*))))

    ;; 2^48 * 2^48 = 2^96 mod P = P-1 (triggers hi_l=0, hi_h>0, underflow)
    (let ((pow48 (to-u64-vec (make-list n :initial-element (expt 2 48)))))
      (check "mul: 2^48 × 2^48 = P-1  (underflow path)"
             (gpu-map *k-sq* :x pow48)
             (to-u64-vec (make-list n :initial-element (ref-mul (expt 2 48) (expt 2 48))))))

    ;; P/2 * 2 = P-1 (large intermediate)
    (let ((halfp (to-u64-vec (make-list n :initial-element *half-p*)))
          (twos  (to-u64-vec (make-list n :initial-element 2))))
      (check "mul: (P/2) × 2"
             (gpu-map *k-mul* :a halfp :b twos)
             (to-u64-vec (make-list n :initial-element (ref-mul *half-p* 2)))))

    ;; ── 4. Chained operations: a^2 + b^2 ──────────────────────
    (format t "~%=== Chained ops (a^2 + b^2) ===~%")

    (check "chain: edge^2 + edge(rev)^2"
           (gpu-map *k-chain* :a a :b b)
           (map 'vector (lambda (x y) (ref-add (ref-mul x x) (ref-mul y y))) a b))

    ;; ── 5. Large-scale random determinism ──────────────────────
    (format t "~%=== Random determinism (N=4096, 50 runs) ===~%")

    (let* ((big-n 4096)
           (ra (to-u64-vec (loop repeat big-n collect (random +P+))))
           (rb (to-u64-vec (loop repeat big-n collect (random +P+)))))

      ;; Verify GPU = CPU on random data
      (check "mul: 4096 random elements vs CPU"
             (gpu-map *k-mul* :a ra :b rb)
             (cpu-map #'ref-mul ra rb))

      (check "add: 4096 random elements vs CPU"
             (gpu-map *k-add* :a ra :b rb)
             (cpu-map #'ref-add ra rb))

      ;; Run 50 times and verify all identical
      (let ((ref-mul-result (gpu-map *k-mul* :a ra :b rb))
            (all-same t))
        (dotimes (i 49)
          (let ((r (gpu-map *k-mul* :a ra :b rb)))
            (unless (equalp r ref-mul-result)
              (setf all-same nil)
              (format t "    Run ~D differs!~%" (+ i 2)))))
        (check "mul: 50 runs × 4096 elements → all identical" all-same t))

      (let ((ref-add-result (gpu-map *k-add* :a ra :b rb))
            (all-same t))
        (dotimes (i 49)
          (let ((r (gpu-map *k-add* :a ra :b rb)))
            (unless (equalp r ref-add-result)
              (setf all-same nil))))
        (check "add: 50 runs × 4096 elements → all identical" all-same t))

      ;; Chain: 50 runs of a^2 + b^2
      (let ((ref-chain-result (gpu-map *k-chain* :a ra :b rb))
            (all-same t))
        (dotimes (i 49)
          (let ((r (gpu-map *k-chain* :a ra :b rb)))
            (unless (equalp r ref-chain-result)
              (setf all-same nil))))
        (check "chain: 50 runs × 4096 elements → all identical" all-same t)))

    ;; ── 6. Adversarial patterns ────────────────────────────────
    (format t "~%=== Adversarial patterns ===~%")

    ;; All same value (tests uniform workgroup behavior)
    (let ((all-pm1 (to-u64-vec (make-list 256 :initial-element (- +P+ 1))))
          (all-eps (to-u64-vec (make-list 256 :initial-element *epsilon*))))
      (check "mul: 256×(P-1) * 256×ε (uniform)"
             (gpu-map *k-mul* :a all-pm1 :b all-eps)
             (to-u64-vec (make-list 256 :initial-element (ref-mul (- +P+ 1) *epsilon*)))))

    ;; Alternating extremes: 0,P-1,0,P-1,...
    (let* ((alt-n 256)
           (alt-a (to-u64-vec (loop for i below alt-n
                                    collect (if (evenp i) 0 (- +P+ 1)))))
           (alt-b (to-u64-vec (loop for i below alt-n
                                    collect (if (evenp i) (- +P+ 1) 0)))))
      (check "add: alternating 0/(P-1) pattern"
             (gpu-map *k-add* :a alt-a :b alt-b)
             (cpu-map #'ref-add alt-a alt-b)))

    ;; Progressive: 0, 1, 2, ..., 255 squared
    (let ((prog-vec (to-u64-vec (loop for i below 256 collect i))))
      (check "mul: progressive 0..255 squared"
             (gpu-map *k-sq* :x prog-vec)
             (map 'vector (lambda (v) (ref-mul v v)) prog-vec)))

    ;; Near-P boundary: P-256, P-255, ..., P-1
    (let ((near-p (to-u64-vec (loop for i from 256 downto 1 collect (- +P+ i)))))
      (check "mul: near-P values (P-256..P-1) squared"
             (gpu-map *k-sq* :x near-p)
             (map 'vector (lambda (v) (ref-mul v v)) near-p))))

  ;; ── Results ────────────────────────────────────────────────
  (format t "~%══════════════════════════════════════════════════════~%")
  (format t "Total: ~D passed, ~D failed~%" *pass* *fail*)
  (if (zerop *fail*)
      (format t "~%  *** ALL DETERMINISM TESTS PASSED ***~%")
      (progn
        (format t "~%  *** ~D TESTS FAILED ***~%" *fail*)
        (format t "  Failed: ~{~A~^, ~}~%" (reverse *details*))))
  (format t "══════════════════════════════════════════════════════~%")

  (gpu-cleanup)
  (zerop *fail*))

(let ((ok (run-tests)))
  (finish-output)
  (sb-ext:exit :code (if ok 0 1)))
