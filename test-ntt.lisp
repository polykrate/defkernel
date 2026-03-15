;;;; test-ntt.lisp — NTT correctness tests
;;;;
;;;; Verifies: round-trip (NTT then INTT = identity),
;;;; polynomial multiplication, GPU vs CPU consistency.
;;;;
;;;; Usage: sbcl --load test-ntt.lisp

(format t "~%Loading defkernel compiler...~%")
(load (merge-pathnames "defkernel.lisp" *load-truename*))
(format t "Loading OpenCL runtime...~%")
(load (merge-pathnames "runtime.lisp" *load-truename*))
(format t "Loading NTT...~%")
(load (merge-pathnames "ntt.lisp" *load-truename*))

(in-package #:defkernel)

(defvar *pass* 0)
(defvar *fail* 0)

(defun check-ntt (name ok &optional detail)
  (if ok
      (progn (incf *pass*) (format t "  ~48A PASS~%" name))
      (progn (incf *fail*)
             (format t "  ~48A FAIL~%" name)
             (when detail (format t "    ~A~%" detail)))))

(defun vec= (a b)
  (and (= (length a) (length b))
       (every #'eql (coerce a 'list) (coerce b 'list))))

(defun u64-vec (&rest elems)
  (make-array (length elems) :element-type '(unsigned-byte 64)
                             :initial-contents elems))

(defun run-ntt-tests ()
  (setf *pass* 0 *fail* 0)

  (format t "~%══ defkernel: NTT Tests ══~%")

  ;; ── 1. Root of unity sanity ─────────────────────────────
  (format t "~%=== Roots of Unity ===~%")

  (let ((omega (primitive-root-of-unity 8)))
    (check-ntt "omega^8 = 1"
               (= 1 (fp-pow omega 8)))
    (check-ntt "omega^4 = P-1 (= -1 mod P)"
               (= (- +goldilocks-p+ 1) (fp-pow omega 4)))
    (check-ntt "omega^2 /= 1 (primitive)"
               (/= 1 (fp-pow omega 2))))

  (let ((omega (primitive-root-of-unity 1024)))
    (check-ntt "omega-1024^1024 = 1"
               (= 1 (fp-pow omega 1024)))
    (check-ntt "omega-1024^512 = -1"
               (= (- +goldilocks-p+ 1) (fp-pow omega 512))))

  ;; ── 2. Bit-reversal ────────────────────────────────────
  (format t "~%=== Bit-Reversal ===~%")

  (let* ((data (u64-vec 0 1 2 3 4 5 6 7))
         (rev (bit-reverse-permutation data)))
    (check-ntt "bit-reverse [0..7]"
               (vec= rev (u64-vec 0 4 2 6 1 5 3 7))))

  (let* ((data (u64-vec 10 20 30 40))
         (rev (bit-reverse-permutation data)))
    (check-ntt "bit-reverse [10 20 30 40]"
               (vec= rev (u64-vec 10 30 20 40))))

  ;; ── 3. CPU NTT round-trip ──────────────────────────────
  (format t "~%=== CPU NTT Round-Trip ===~%")

  (dolist (n '(4 8 16 64 256))
    (let* ((data (make-array n :element-type '(unsigned-byte 64)))
           (state (sb-ext:seed-random-state 42)))
      (dotimes (i n) (setf (aref data i) (random +goldilocks-p+ state)))
      (let* ((fwd (cpu-ntt-forward data))
             (back (cpu-ntt-inverse fwd)))
        (check-ntt (format nil "CPU NTT round-trip N=~D" n)
                   (vec= data back)))))

  ;; ── 4. CPU polynomial multiplication ───────────────────
  (format t "~%=== CPU Polynomial Multiply ===~%")

  ;; (1 + 2x) * (3 + 4x) = 3 + 10x + 8x^2
  (let ((result (poly-mul-ntt (u64-vec 1 2) (u64-vec 3 4) :gpu nil)))
    (check-ntt "poly (1+2x)*(3+4x) = 3+10x+8x^2"
               (vec= result (u64-vec 3 10 8))))

  ;; (1 + x)^2 = 1 + 2x + x^2
  (let ((result (poly-mul-ntt (u64-vec 1 1) (u64-vec 1 1) :gpu nil)))
    (check-ntt "poly (1+x)^2 = 1+2x+x^2"
               (vec= result (u64-vec 1 2 1))))

  ;; ── 5. GPU NTT (if available) ──────────────────────────
  (format t "~%=== GPU NTT ===~%")

  (cond
    ((not (gpu-available-p))
     (format t "  No OpenCL device — skipping GPU tests.~%"))

    (t
     (gpu-init)

     ;; GPU NTT should match CPU NTT
     (dolist (n '(4 8 16 64 256 1024))
       (let* ((data (make-array n :element-type '(unsigned-byte 64)))
              (state (sb-ext:seed-random-state 42)))
         (dotimes (i n) (setf (aref data i) (random +goldilocks-p+ state)))
         (let ((cpu-result (cpu-ntt-forward data))
               (gpu-result (ntt-forward data :gpu t)))
           (check-ntt (format nil "GPU NTT = CPU NTT, N=~D" n)
                      (vec= cpu-result gpu-result)
                      (when (not (vec= cpu-result gpu-result))
                        (format nil "First diff at ~D"
                                (position nil (map 'vector #'eql
                                                   (coerce cpu-result 'list)
                                                   (coerce gpu-result 'list)))))))))

     ;; GPU round-trip
     (dolist (n '(4 8 16 256 1024))
       (let* ((data (make-array n :element-type '(unsigned-byte 64)))
              (state (sb-ext:seed-random-state 123)))
         (dotimes (i n) (setf (aref data i) (random +goldilocks-p+ state)))
         (let* ((fwd (ntt-forward data :gpu t))
                (back (ntt-inverse fwd :gpu t)))
           (check-ntt (format nil "GPU round-trip N=~D" n)
                      (vec= data back)))))

     ;; GPU polynomial multiplication
     (let ((result (poly-mul-ntt (u64-vec 1 2) (u64-vec 3 4) :gpu t)))
       (check-ntt "GPU poly (1+2x)*(3+4x) = 3+10x+8x^2"
                  (vec= result (u64-vec 3 10 8))))

     ;; Determinism: 10 GPU NTT runs produce identical output
     (let* ((n 256)
            (data (make-array n :element-type '(unsigned-byte 64)))
            (state (sb-ext:seed-random-state 77)))
       (dotimes (i n) (setf (aref data i) (random +goldilocks-p+ state)))
       (let ((ref (ntt-forward data :gpu t))
             (all-same t))
         (dotimes (i 9)
           (unless (vec= ref (ntt-forward data :gpu t))
             (setf all-same nil)))
         (check-ntt "GPU NTT determinism: 10 runs identical"
                    all-same)))

     (gpu-cleanup)))

  ;; ── Results ────────────────────────────────────────────
  (format t "~%════════════════════════════════════════════════════~%")
  (format t "Total: ~D passed, ~D failed~%" *pass* *fail*)
  (if (zerop *fail*)
      (format t "~%  *** ALL NTT TESTS PASSED ***~%")
      (format t "~%  *** ~D TESTS FAILED ***~%" *fail*))
  (format t "════════════════════════════════════════════════════~%")
  (zerop *fail*))

(let ((ok (run-ntt-tests)))
  (finish-output)
  (sb-ext:exit :code (if ok 0 1)))
