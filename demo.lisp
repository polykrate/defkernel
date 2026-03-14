;;;; demo.lisp — Run cl-gpu kernels on actual GPU hardware
;;;;
;;;; Usage: sbcl --load demo.lisp

(format t "~%Loading cl-gpu compiler...~%")
(load (merge-pathnames "cl-gpu.lisp" *load-truename*))
(format t "Loading OpenCL runtime...~%")
(load (merge-pathnames "runtime.lisp" *load-truename*))

(in-package #:cl-gpu)

;;; ═══════════════════════════════════════════════════════════════════
;;; Define kernels (compiled at load time, not at runtime)
;;; ═══════════════════════════════════════════════════════════════════

(defkernel *field-add-kernel*
    ((a :vec) (b :vec) (out :out))
  (f+ a b))

(defkernel *field-mul-kernel*
    ((a :vec) (b :vec) (out :out))
  (f* a b))

(defkernel *field-square-kernel*
    ((x :vec) (out :out))
  (f* x x))

(defkernel *poly-eval-kernel*
    ((x :vec) (c0 :scalar) (c1 :scalar) (c2 :scalar) (out :out))
  (f+ c0 (f+ (f* c1 x) (f* c2 (f* x x)))))

;;; ═══════════════════════════════════════════════════════════════════
;;; Reference CPU implementation for verification
;;; ═══════════════════════════════════════════════════════════════════

(defconstant +P+ (+ (- (expt 2 64) (expt 2 32)) 1)
  "Goldilocks prime for reference calculations.")

(defun fp-add (a b) (mod (+ a b) +P+))
(defun fp-mul (a b) (mod (* a b) +P+))

;;; ═══════════════════════════════════════════════════════════════════
;;; Demo
;;; ═══════════════════════════════════════════════════════════════════

(defun run-demo ()
  (format t "~%╔══════════════════════════════════════════════╗~%")
  (format t   "║  CL-GPU: Real GPU Execution Demo              ║~%")
  (format t   "╚══════════════════════════════════════════════╝~%")

  ;; Show generated kernels
  (format t "~%=== Generated OpenCL C Sources ===~%")
  (format t "~%--- field-add ---~%~A" (kernel-source *field-add-kernel*))
  (format t "~%--- field-mul ---~%~A" (kernel-source *field-mul-kernel*))
  (format t "~%--- poly-eval ---~%~A" (kernel-source *poly-eval-kernel*))

  ;; Check GPU availability
  (format t "~%=== GPU Detection ===~%")
  (unless (gpu-available-p)
    (format t "~%No OpenCL device available.~%")
    (format t "For Intel integrated GPU, install: sudo pacman -S intel-compute-runtime~%")
    (format t "~%Kernels compiled successfully — you just need a GPU driver to run them.~%")
    (finish-output)
    (sb-ext:exit :code 0))

  ;; Initialize GPU
  (gpu-init)

  ;; Test 1: Vector field addition
  (format t "~%=== Test 1: Field Addition (a + b mod P) ===~%")
  (let* ((n 8)
         (a (make-array n :element-type '(unsigned-byte 64)
                          :initial-contents '(1 2 3 4 5 6 7 8)))
         (b (make-array n :element-type '(unsigned-byte 64)
                          :initial-contents '(10 20 30 40 50 60 70 80)))
         (gpu-result (gpu-map *field-add-kernel* :a a :b b))
         (cpu-result (map 'vector #'fp-add a b)))
    (format t "  a:   ~A~%" a)
    (format t "  b:   ~A~%" b)
    (format t "  GPU: ~A~%" gpu-result)
    (format t "  CPU: ~A~%" cpu-result)
    (format t "  Match: ~A~%" (equalp gpu-result cpu-result)))

  ;; Test 2: Vector field multiplication
  (format t "~%=== Test 2: Field Multiplication (a * b mod P) ===~%")
  (let* ((n 8)
         (a (make-array n :element-type '(unsigned-byte 64)
                          :initial-contents '(100 200 300 400 500 600 700 800)))
         (b (make-array n :element-type '(unsigned-byte 64)
                          :initial-contents '(3 5 7 11 13 17 19 23)))
         (gpu-result (gpu-map *field-mul-kernel* :a a :b b))
         (cpu-result (map 'vector #'fp-mul a b)))
    (format t "  a:   ~A~%" a)
    (format t "  b:   ~A~%" b)
    (format t "  GPU: ~A~%" gpu-result)
    (format t "  CPU: ~A~%" cpu-result)
    (format t "  Match: ~A~%" (equalp gpu-result cpu-result)))

  ;; Test 3: Field squaring
  (format t "~%=== Test 3: Field Squaring (x^2 mod P) ===~%")
  (let* ((n 4)
         (x (make-array n :element-type '(unsigned-byte 64)
                          :initial-contents (list 2 (- +P+ 1) (ash 1 32) 42)))
         (gpu-result (gpu-map *field-square-kernel* :x x))
         (cpu-result (map 'vector (lambda (v) (fp-mul v v)) x)))
    (format t "  x:   ~A~%" x)
    (format t "  GPU: ~A~%" gpu-result)
    (format t "  CPU: ~A~%" cpu-result)
    (format t "  Match: ~A~%" (equalp gpu-result cpu-result)))

  ;; Test 4: Polynomial evaluation with scalars
  (format t "~%=== Test 4: Polynomial c0 + c1*x + c2*x^2 ===~%")
  (let* ((n 4)
         (x (make-array n :element-type '(unsigned-byte 64)
                          :initial-contents '(0 1 2 3)))
         (c0 7) (c1 5) (c2 3)
         (gpu-result (gpu-execute *poly-eval-kernel*
                                  :global-size n
                                  :inputs `((x . ,x))
                                  :scalars `((c0 . ,c0) (c1 . ,c1) (c2 . ,c2))
                                  :output-size n))
         (cpu-result (map 'vector
                          (lambda (xi)
                            (fp-add c0 (fp-add (fp-mul c1 xi)
                                               (fp-mul c2 (fp-mul xi xi)))))
                          x)))
    (format t "  x:   ~A~%" x)
    (format t "  c0=~D c1=~D c2=~D~%" c0 c1 c2)
    (format t "  GPU: ~A~%" gpu-result)
    (format t "  CPU: ~A~%" cpu-result)
    (format t "  Match: ~A~%" (equalp gpu-result cpu-result)))

  ;; Test 5: Determinism — run same kernel 10 times
  (format t "~%=== Test 5: Determinism (10 runs) ===~%")
  (let* ((n 1024)
         (a (make-array n :element-type '(unsigned-byte 64)))
         (b (make-array n :element-type '(unsigned-byte 64))))
    (dotimes (i n)
      (setf (aref a i) (random (expt 2 64))
            (aref b i) (random (expt 2 64))))
    (let ((results (loop repeat 10
                         collect (gpu-map *field-mul-kernel* :a a :b b))))
      (let ((all-same (every (lambda (r) (equalp r (first results)))
                             (rest results))))
        (format t "  1024-element field mul × 10 runs~%")
        (format t "  All identical: ~A~%" all-same))))

  ;; Test 6: Large-scale throughput
  (format t "~%=== Test 6: Throughput (64K elements) ===~%")
  (let* ((n 65536)
         (a (make-array n :element-type '(unsigned-byte 64)))
         (b (make-array n :element-type '(unsigned-byte 64))))
    (dotimes (i n)
      (setf (aref a i) (mod i (expt 2 63))
            (aref b i) (mod (* i 17) (expt 2 63))))
    (let* ((t0 (get-internal-real-time))
           (result (gpu-map *field-add-kernel* :a a :b b))
           (t1 (get-internal-real-time))
           (dt (- t1 t0))
           (ms (if (zerop dt) 0.01
                   (/ (* 1000.0 dt) internal-time-units-per-second))))
      (format t "  Elements: ~D~%" n)
      (format t "  Time:     ~,2F ms~%" ms)
      (format t "  Throughput: ~,1F M field-ops/sec~%"
              (/ n ms 1000.0))
      (format t "  First 8:  ~A~%" (subseq result 0 (min 8 n)))))

  ;; Summary
  (format t "~%═══════════════════════════════════════════════~%")
  (gpu-info)
  (gpu-cleanup)
  (format t "═══════════════════════════════════════════════~%"))

(run-demo)
(finish-output)
(sb-ext:exit :code 0)
