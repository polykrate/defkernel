;;;; fri.lisp — FRI polynomial commitment scheme over Goldilocks
;;;;
;;;; FRI (Fast Reed-Solomon IOP) is the polynomial commitment used by STARKs.
;;;; No elliptic curves — just NTT + hashing.
;;;;
;;;; The protocol:
;;;; 1. Prover evaluates polynomial p(x) at all roots of unity → evals
;;;; 2. Prover commits to evals via Merkle tree
;;;; 3. For each FRI round:
;;;;    a. Verifier sends random challenge alpha
;;;;    b. Prover folds: p'(x) = even(x) + alpha * odd(x), halving degree
;;;;    c. Prover commits to new evaluations
;;;; 4. When degree is small enough, prover sends final polynomial in the clear
;;;; 5. Verifier checks consistency at random query positions

(in-package #:defkernel)

;;; ═══════════════════════════════════════════════════════════════════
;;; 1. FRI fold kernel (GPU-accelerated)
;;;
;;; Folds a polynomial evaluation vector by combining even/odd pairs.
;;; p'[i] = evals[2i] + alpha * evals[2i+1]
;;; This halves the degree with each round.
;;; ═══════════════════════════════════════════════════════════════════

(defkernel *fri-fold-kernel*
    ((evals :vec) (alpha :scalar) (out :out))
  (f+ (aref evals (* 2 gid))
      (f* alpha (aref evals (+ (* 2 gid) 1)))))

;;; ═══════════════════════════════════════════════════════════════════
;;; 2. CPU Merkle tree (hash via SipHash-like mixing)
;;;
;;; Hashing isn't field arithmetic, so it stays on CPU.
;;; For production: use BLAKE3 or Poseidon. Here we use a simple
;;; collision-resistant mixing function sufficient for testing.
;;; ═══════════════════════════════════════════════════════════════════

(defun mix64 (a b)
  "Mix two u64 values into a u64 hash. NOT cryptographic."
  (let* ((h (logxor a (logxor (ash b -16) (logand (ash b 48) #xFFFFFFFFFFFFFFFF))))
         (h (logand (logxor h (logand (* h #xBF58476D1CE4E5B9) #xFFFFFFFFFFFFFFFF))
                    #xFFFFFFFFFFFFFFFF))
         (h (logand (logxor h (ash h -31)) #xFFFFFFFFFFFFFFFF))
         (h (logand (logxor h (logand (* h #x94D049BB133111EB) #xFFFFFFFFFFFFFFFF))
                    #xFFFFFFFFFFFFFFFF)))
    (logand (logxor h (ash h -32)) #xFFFFFFFFFFFFFFFF)))

(defun hash-leaf (value)
  "Hash a single field element for Merkle leaf."
  (mix64 value #xDEADBEEFCAFEBABE))

(defun hash-pair (left right)
  "Hash two node hashes together for Merkle internal node."
  (mix64 left right))

(defun merkle-commit (evals)
  "Build a Merkle tree over field element evaluations.
   Returns (values root tree) where tree is a vector of hash nodes.
   Tree layout: tree[1] = root, tree[2..3] = level 1, etc.
   Leaves start at tree[n..2n-1]."
  (let* ((n (length evals))
         (tree (make-array (* 2 n) :initial-element 0)))
    ;; Hash leaves
    (dotimes (i n)
      (setf (aref tree (+ n i)) (hash-leaf (aref evals i))))
    ;; Build up
    (loop for i from (1- n) downto 1
          do (setf (aref tree i)
                   (hash-pair (aref tree (* 2 i))
                              (aref tree (+ (* 2 i) 1)))))
    (values (aref tree 1) tree)))

(defun merkle-path (tree leaf-idx n)
  "Get the Merkle authentication path for a leaf.
   Returns a list of (sibling-hash . side) pairs from leaf to root."
  (let ((path nil)
        (idx (+ n leaf-idx)))
    (loop while (> idx 1) do
      (let ((sibling (if (evenp idx) (1+ idx) (1- idx)))
            (side (if (evenp idx) :right :left)))
        (push (cons (aref tree sibling) side) path))
      (setf idx (floor idx 2)))
    (nreverse path)))

(defun merkle-verify (root leaf-hash path)
  "Verify a Merkle proof. Returns T if the path is valid."
  (let ((current leaf-hash))
    (dolist (step path)
      (let ((sibling (car step))
            (side (cdr step)))
        (setf current (if (eq side :right)
                          (hash-pair current sibling)
                          (hash-pair sibling current)))))
    (eql current root)))

;;; ═══════════════════════════════════════════════════════════════════
;;; 3. FRI commit phase
;;; ═══════════════════════════════════════════════════════════════════

(defstruct fri-layer
  (evals nil)
  (root nil)
  (tree nil)
  (alpha 0 :type (unsigned-byte 64)))

(defun fri-fold-cpu (evals alpha)
  "CPU FRI fold: combine even/odd pairs."
  (let* ((n (length evals))
         (half (ash n -1))
         (result (make-array half :element-type '(unsigned-byte 64))))
    (dotimes (i half result)
      (setf (aref result i)
            (fp+ (aref evals (* 2 i))
                 (fp* alpha (aref evals (+ (* 2 i) 1))))))))

(defun fri-fold-gpu (evals alpha)
  "GPU FRI fold using the defkernel."
  (let* ((n (length evals))
         (half (ash n -1))
         (buf-in (gpu-alloc n))
         (buf-out (gpu-alloc half)))
    (unwind-protect
         (progn
           (gpu-upload buf-in evals)
           (gpu-dispatch *fri-fold-kernel*
                         :global-size half
                         :buffers `((evals . ,buf-in) (out . ,buf-out))
                         :scalars `((alpha . ,alpha)))
           (gpu-download buf-out))
      (gpu-free buf-in)
      (gpu-free buf-out))))

(defun fri-commit (evals &key (num-rounds nil) (gpu t) (random-state *random-state*))
  "Run the FRI commit phase.
   Takes polynomial evaluations, produces a list of FRI layers.
   Each layer commits to the folded evaluations with a random alpha.
   NUM-ROUNDS defaults to log2(len)-2 (stop when degree <= 4)."
  (let* ((n (length evals))
         (log-n (1- (integer-length n)))
         (rounds (or num-rounds (max 1 (- log-n 2))))
         (layers nil)
         (current evals))
    (dotimes (r rounds)
      (multiple-value-bind (root tree) (merkle-commit current)
        (let ((alpha (mod (random (expt 2 64) random-state) +goldilocks-p+)))
          (push (make-fri-layer :evals current :root root :tree tree :alpha alpha)
                layers)
          (setf current (if (and gpu *gpu*)
                            (fri-fold-gpu current alpha)
                            (fri-fold-cpu current alpha))))))
    ;; Final layer (no folding)
    (multiple-value-bind (root tree) (merkle-commit current)
      (push (make-fri-layer :evals current :root root :tree tree :alpha 0) layers))
    (nreverse layers)))

;;; ═══════════════════════════════════════════════════════════════════
;;; 4. FRI query phase
;;; ═══════════════════════════════════════════════════════════════════

(defstruct fri-query
  (layer-idx 0)
  (position 0)
  (value 0 :type (unsigned-byte 64))
  (sibling-value 0 :type (unsigned-byte 64))
  (merkle-path nil))

(defun fri-query (layers position)
  "Generate a FRI query at a given position.
   Returns a list of FRI-QUERY structs, one per layer."
  (let ((queries nil)
        (pos position))
    (loop for layer in layers
          for idx from 0
          do (let* ((evals (fri-layer-evals layer))
                    (n (length evals))
                    (tree (fri-layer-tree layer))
                    (p (mod pos n))
                    (sibling (logxor p 1)))
               (push (make-fri-query
                      :layer-idx idx
                      :position p
                      :value (aref evals p)
                      :sibling-value (aref evals sibling)
                      :merkle-path (merkle-path tree p n))
                     queries)
               (setf pos (ash pos -1))))
    (nreverse queries)))

(defun fri-verify-query (layers queries)
  "Verify a FRI query against the committed layers.
   Checks: Merkle proofs, FRI folding consistency."
  (loop for query in queries
        for layer in layers
        for next-query in (append (rest queries) (list nil))
        always
        (let* ((root (fri-layer-root layer))
               (val (fri-query-value query))
               (leaf-hash (hash-leaf val))
               (path (fri-query-merkle-path query)))
          (and
           (merkle-verify root leaf-hash path)
           (or (null next-query)
               (let* ((alpha (fri-layer-alpha layer))
                      (pos (fri-query-position query))
                      (even-val (if (evenp pos) val (fri-query-sibling-value query)))
                      (odd-val (if (evenp pos) (fri-query-sibling-value query) val))
                      (expected (fp+ even-val (fp* alpha odd-val)))
                      (actual (fri-query-value next-query)))
                 (= expected actual)))))))

;;; ═══════════════════════════════════════════════════════════════════
;;; Export FRI symbols
;;; ═══════════════════════════════════════════════════════════════════

(eval-when (:compile-toplevel :load-toplevel :execute)
  (export '(fri-commit fri-query fri-verify-query
            fri-layer fri-layer-evals fri-layer-root fri-layer-alpha
            fri-fold-cpu fri-fold-gpu
            merkle-commit merkle-path merkle-verify)
          :defkernel))
