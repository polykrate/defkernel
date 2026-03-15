;;;; ntt.lisp — GPU-accelerated Number Theoretic Transform over Goldilocks
;;;;
;;;; NTT is the core primitive for STARK provers: it converts a polynomial
;;;; from coefficient form to evaluation form in O(N log N) field operations.
;;;;
;;;; The Goldilocks prime P = 2^64 - 2^32 + 1 has P-1 = 2^32 * (2^32 - 1),
;;;; so NTTs up to size 2^32 are possible.
;;;;
;;;; Usage:
;;;;   (load "defkernel.lisp")
;;;;   (load "runtime.lisp")
;;;;   (load "ntt.lisp")
;;;;   (gpu-init)
;;;;   (ntt-forward data)   ; → evaluations
;;;;   (ntt-inverse evals)  ; → coefficients

(in-package #:defkernel)

;;; ═══════════════════════════════════════════════════════════════════
;;; 1. Goldilocks field constants and CPU arithmetic
;;; ═══════════════════════════════════════════════════════════════════

(defconstant +goldilocks-p+ (+ (- (expt 2 64) (expt 2 32)) 1)
  "Goldilocks prime P = 2^64 - 2^32 + 1")

(defun fp+ (a b) (mod (+ a b) +goldilocks-p+))
(defun fp- (a b) (mod (- a b) +goldilocks-p+))
(defun fp* (a b) (mod (* a b) +goldilocks-p+))

(defun fp-pow (base exp)
  "Modular exponentiation: base^exp mod P. Uses square-and-multiply."
  (let ((result 1)
        (b (mod base +goldilocks-p+)))
    (loop while (plusp exp) do
      (when (oddp exp)
        (setf result (fp* result b)))
      (setf b (fp* b b))
      (setf exp (ash exp -1)))
    result))

(defun fp-inv (a)
  "Modular inverse: a^(-1) mod P = a^(P-2) mod P (Fermat's little theorem)."
  (fp-pow a (- +goldilocks-p+ 2)))

;;; ═══════════════════════════════════════════════════════════════════
;;; 2. Roots of unity for NTT
;;;
;;; For NTT of size N = 2^k, we need a primitive Nth root of unity omega
;;; such that omega^N = 1 and omega^(N/2) = -1 (mod P).
;;;
;;; Generator of the multiplicative group: g = 7
;;; Primitive Nth root: omega = g^((P-1)/N) mod P
;;; ═══════════════════════════════════════════════════════════════════

(defconstant +goldilocks-generator+ 7
  "Generator of the multiplicative group of GF(P).")

(defun primitive-root-of-unity (n)
  "Return a primitive Nth root of unity in GF(P).
   N must be a power of 2 and divide P-1."
  (assert (and (plusp n) (zerop (logand n (1- n))))
          () "N must be a power of 2, got ~D" n)
  (let ((max-order (ash 1 32)))
    (assert (<= n max-order)
            () "N=~D exceeds max NTT size 2^32" n))
  (fp-pow +goldilocks-generator+ (/ (- +goldilocks-p+ 1) n)))

(defun compute-twiddle-factors (n)
  "Precompute twiddle factors omega^0, omega^1, ..., omega^(N/2-1) for NTT of size N.
   Returns a u64 vector."
  (let* ((omega (primitive-root-of-unity n))
         (half (ash n -1))
         (twiddles (make-array half :element-type '(unsigned-byte 64))))
    (let ((w 1))
      (dotimes (i half twiddles)
        (setf (aref twiddles i) w)
        (setf w (fp* w omega))))))

;;; ═══════════════════════════════════════════════════════════════════
;;; 3. Bit-reversal permutation
;;; ═══════════════════════════════════════════════════════════════════

(defun bit-reverse (x bits)
  "Reverse the bottom BITS bits of integer X."
  (let ((result 0))
    (dotimes (i bits result)
      (setf result (logior (ash result 1) (logand x 1)))
      (setf x (ash x -1)))))

(defun bit-reverse-permutation (data)
  "Return a new vector with elements in bit-reversed order."
  (let* ((n (length data))
         (log-n (1- (integer-length n)))
         (result (make-array n :element-type '(unsigned-byte 64))))
    (dotimes (i n result)
      (setf (aref result (bit-reverse i log-n)) (aref data i)))))

;;; ═══════════════════════════════════════════════════════════════════
;;; 4. NTT butterfly kernel
;;;
;;; Each work-item handles one output element.
;;; For stride S at DIT stage:
;;;   - group  = gid / (2*S)
;;;   - j      = gid % S
;;;   - top    = group * 2S + j
;;;   - bot    = top + S
;;;   - twiddle = twiddles[group * S + j]
;;;
;;; If gid's position within its 2S-block is < S → top output
;;; Otherwise → bottom output
;;; ═══════════════════════════════════════════════════════════════════

(defkernel *ntt-butterfly*
    ((data-in :vec) (twiddles :vec) (out :out) (stride :scalar))
  (select (cmp< (imod gid (* 2 stride)) stride)
          ;; Top half: gid IS the top element, partner is gid+stride
          (f+ (aref data-in gid)
              (f* (aref twiddles (+ (* (idiv gid (* 2 stride)) stride)
                                    (imod gid stride)))
                  (aref data-in (+ gid stride))))
          ;; Bottom half: top element is gid-stride, gid IS the bottom
          (f- (aref data-in (- gid stride))
              (f* (aref twiddles (+ (* (idiv gid (* 2 stride)) stride)
                                    (imod gid stride)))
                  (aref data-in gid)))))

;;; Scaling kernel for inverse NTT: multiply each element by n_inv
(defkernel *ntt-scale*
    ((data :vec) (n-inv :scalar) (out :out))
  (f* data n-inv))

;;; ═══════════════════════════════════════════════════════════════════
;;; 5. GPU NTT orchestration
;;; ═══════════════════════════════════════════════════════════════════

(defun ntt-forward (data &key (gpu t))
  "Compute the forward NTT of DATA (u64 vector, length must be power of 2).
   Returns a new u64 vector with the evaluations.
   If GPU is NIL, falls back to CPU NTT."
  (let* ((n (length data))
         (log-n (1- (integer-length n))))
    (assert (= n (ash 1 log-n)) () "NTT size must be power of 2, got ~D" n)

    (if (or (not gpu) (not *gpu*))
        (cpu-ntt-forward data)
        (gpu-ntt data log-n :inverse nil))))

(defun ntt-inverse (data &key (gpu t))
  "Compute the inverse NTT of DATA.
   Returns a new u64 vector with the coefficients."
  (let* ((n (length data))
         (log-n (1- (integer-length n))))
    (assert (= n (ash 1 log-n)) () "NTT size must be power of 2, got ~D" n)

    (if (or (not gpu) (not *gpu*))
        (cpu-ntt-inverse data)
        (gpu-ntt data log-n :inverse t))))

(defun gpu-ntt (data log-n &key inverse)
  "Run NTT on GPU using multi-pass butterfly dispatch."
  (let* ((n (ash 1 log-n))
         (permuted (bit-reverse-permutation data))
         (buf-a (gpu-alloc n))
         (buf-b (gpu-alloc n))
         (tw-buf (gpu-alloc (ash n -1))))

    (unwind-protect
         (progn
           (gpu-upload buf-a permuted)

           ;; For each DIT stage
           (loop for stage below log-n
                 for stride = 1 then (* stride 2)
                 do
                    ;; Twiddle factors: omega_2S^j where j = i % stride
                    ;; Precompute one period then tile to N/2 entries
                    (let* ((omega-stage (if inverse
                                            (fp-inv (primitive-root-of-unity (* 2 stride)))
                                            (primitive-root-of-unity (* 2 stride))))
                           (period (make-array stride :element-type '(unsigned-byte 64)))
                           (twiddles (make-array (ash n -1)
                                                 :element-type '(unsigned-byte 64))))
                      ;; Build one period: omega^0, omega^1, ..., omega^(stride-1)
                      (let ((w 1))
                        (dotimes (j stride)
                          (setf (aref period j) w)
                          (setf w (fp* w omega-stage))))
                      ;; Tile: twiddles[i] = period[i % stride]
                      (dotimes (i (ash n -1))
                        (setf (aref twiddles i) (aref period (mod i stride))))
                      (gpu-upload tw-buf twiddles))

                    ;; Dispatch butterfly: buf-a → buf-b
                    (gpu-dispatch *ntt-butterfly*
                                  :global-size n
                                  :buffers `((data-in . ,buf-a)
                                             (twiddles . ,tw-buf)
                                             (out . ,buf-b))
                                  :scalars `((stride . ,stride)))

                    ;; Swap buffers for next stage
                    (rotatef buf-a buf-b))

           ;; For inverse NTT, scale by N^(-1)
           (when inverse
             (let ((n-inv (fp-inv n)))
               (gpu-dispatch *ntt-scale*
                             :global-size n
                             :buffers `((data . ,buf-a) (out . ,buf-b))
                             :scalars `((n-inv . ,n-inv)))
               (rotatef buf-a buf-b)))

           (gpu-download buf-a))

      ;; Cleanup
      (gpu-free buf-a)
      (gpu-free buf-b)
      (gpu-free tw-buf))))

;;; ═══════════════════════════════════════════════════════════════════
;;; 6. CPU NTT (reference implementation)
;;; ═══════════════════════════════════════════════════════════════════

(defun cpu-ntt-forward (data)
  "CPU reference NTT (Cooley-Tukey DIT)."
  (let* ((n (length data))
         (log-n (1- (integer-length n)))
         (result (bit-reverse-permutation data)))

    (loop for stage below log-n
          for stride = 1 then (* stride 2)
          do (let ((omega (primitive-root-of-unity (* 2 stride))))
               (dotimes (group (ash n (- (1+ stage))))
                 (let ((w 1))
                   (dotimes (j stride)
                     (let* ((top-idx (+ (* group 2 stride) j))
                            (bot-idx (+ top-idx stride))
                            (a (aref result top-idx))
                            (b (aref result bot-idx))
                            (wb (fp* w b)))
                       (setf (aref result top-idx) (fp+ a wb))
                       (setf (aref result bot-idx) (fp- a wb))
                       (setf w (fp* w omega))))))))
    result))

(defun cpu-ntt-inverse (data)
  "CPU reference inverse NTT."
  (let* ((n (length data))
         (log-n (1- (integer-length n)))
         (result (bit-reverse-permutation data))
         (n-inv (fp-inv n)))

    (loop for stage below log-n
          for stride = 1 then (* stride 2)
          do (let ((omega (fp-inv (primitive-root-of-unity (* 2 stride)))))
               (dotimes (group (ash n (- (1+ stage))))
                 (let ((w 1))
                   (dotimes (j stride)
                     (let* ((top-idx (+ (* group 2 stride) j))
                            (bot-idx (+ top-idx stride))
                            (a (aref result top-idx))
                            (b (aref result bot-idx))
                            (wb (fp* w b)))
                       (setf (aref result top-idx) (fp+ a wb))
                       (setf (aref result bot-idx) (fp- a wb))
                       (setf w (fp* w omega))))))))

    ;; Scale by N^(-1)
    (dotimes (i n result)
      (setf (aref result i) (fp* (aref result i) n-inv)))))

;;; ═══════════════════════════════════════════════════════════════════
;;; 7. Polynomial multiplication via NTT
;;; ═══════════════════════════════════════════════════════════════════

(defun poly-mul-ntt (a b &key (gpu t))
  "Multiply two polynomials using NTT. A and B are coefficient vectors.
   Returns the product polynomial coefficients."
  (let* ((na (length a))
         (nb (length b))
         (n-result (+ na nb -1))
         (n (ash 1 (ceiling (log n-result 2))))
         (pa (make-array n :element-type '(unsigned-byte 64) :initial-element 0))
         (pb (make-array n :element-type '(unsigned-byte 64) :initial-element 0)))
    (replace pa a)
    (replace pb b)
    (let* ((fa (ntt-forward pa :gpu gpu))
           (fb (ntt-forward pb :gpu gpu))
           (fc (map 'vector #'fp* fa fb))
           (result (ntt-inverse fc :gpu gpu)))
      (subseq result 0 n-result))))

;;; ═══════════════════════════════════════════════════════════════════
;;; Export NTT symbols
;;; ═══════════════════════════════════════════════════════════════════

(eval-when (:compile-toplevel :load-toplevel :execute)
  (export '(ntt-forward ntt-inverse poly-mul-ntt
            cpu-ntt-forward cpu-ntt-inverse
            fp+ fp- fp* fp-pow fp-inv
            +goldilocks-p+ primitive-root-of-unity
            compute-twiddle-factors bit-reverse-permutation)
          :defkernel))
