;;;; defkernel.lisp — Lisp AST → Deterministic GPU Kernel Compiler
;;;;
;;;; The idea: Common Lisp macros operate on the AST before compilation.
;;;; We use this to REWRITE arithmetic expressions into a form with
;;;; a fixed evaluation order, then emit OpenCL C code.
;;;;
;;;; The GPU is non-deterministic because:
;;;;   - Thread execution order varies
;;;;   - Float addition is not associative: (a+b)+c ≠ a+(b+c)
;;;;   - Reduction order is hardware-dependent
;;;;
;;;; Our fix:
;;;;   1. Use integer field arithmetic (mod P) — exact, no floats
;;;;   2. Impose a fixed evaluation order via SSA form
;;;;   3. Generate balanced binary reduction trees (not left-folds)
;;;;   4. All of this happens at MACROEXPANSION TIME via Lisp macros
;;;;
;;;; The Lisp code never runs on the GPU. It's a COMPILER that
;;;; generates deterministic OpenCL C at compile time.
;;;;
;;;; Usage: sbcl --load defkernel.lisp

(defpackage #:defkernel
  (:use #:cl)
  (:export #:defkernel #:compile-kernel #:kernel-source
           #:kvec #:kscalar #:kreduce #:kmap))

(in-package #:defkernel)

;;; ═══════════════════════════════════════════════════════════════════
;;; 1. SSA IR — Static Single Assignment intermediate representation
;;;
;;; Every sub-expression gets a unique temporary variable.
;;; This freezes the evaluation order: t0 before t1 before t2 etc.
;;;
;;; An SSA node is: (name type op args)
;;;   name = :t0, :t1, ...
;;;   type = :u64 | :field
;;;   op   = :add | :mul | :sub | :load-global | :load-scalar | :const | :mod
;;;   args = list of SSA names or literals
;;; ═══════════════════════════════════════════════════════════════════

(defvar *ssa-counter* 0)
(defvar *ssa-nodes* nil)
(defvar *ssa-inputs* nil)

(defstruct ssa-node
  (name nil :type keyword)
  (ctype :u64 :type keyword)      ; :u64 or :field
  (op nil :type keyword)
  (args nil :type list))

(defun fresh-ssa ()
  (let ((name (intern (format nil "T~D" *ssa-counter*) :keyword)))
    (incf *ssa-counter*)
    name))

(defun emit-ssa (op args &key (ctype :u64))
  "Emit an SSA node. Returns its name."
  (let* ((name (fresh-ssa))
         (node (make-ssa-node :name name :ctype ctype :op op :args args)))
    (push node *ssa-nodes*)
    name))

;;; ═══════════════════════════════════════════════════════════════════
;;; 2. AST → SSA lowering
;;;
;;; Walks a Lisp s-expression and emits SSA nodes.
;;; The key: the WALK ORDER is the evaluation order.
;;; Since Lisp evaluates arguments left-to-right, and we walk
;;; the tree depth-first left-to-right, the SSA order is fixed.
;;; ═══════════════════════════════════════════════════════════════════

(declaim (ftype function lower-binop lower-reduce param-to-c))

(defun lower-expr (expr)
  "Lower a Lisp expression to SSA. Returns the SSA name of the result."
  (cond
    ;; Literal integer
    ((integerp expr)
     (emit-ssa :const (list expr)))

    ;; Symbol = variable reference (scalar or vector element)
    ((symbolp expr)
     (let ((info (assoc expr *ssa-inputs*)))
       (unless info
         (error "Unknown variable in kernel: ~A" expr))
       (ecase (second info)
         (:scalar (emit-ssa :load-scalar (list expr)))
         (:vec    (emit-ssa :load-vec (list expr))))))

    ;; S-expression = operation
    ((listp expr)
     (let ((op (first expr))
           (operands (rest expr)))
       (case op
         ;; Field arithmetic: fixed order, left-to-right
         ((f+ fp-add)
          (lower-binop :field-add operands))
         ((f* fp-mul)
          (lower-binop :field-mul operands))
         ((f- fp-sub)
          (lower-binop :field-sub operands))

         ;; Wrapping u64 arithmetic (for SIMD-like ops)
         (+  (lower-binop :add operands))
         (*  (lower-binop :mul operands))
         (-  (lower-binop :sub operands))

         ;; Bitwise
         (logand (lower-binop :and operands))
         (logxor (lower-binop :xor operands))
         (logior (lower-binop :or operands))
         (ash    (let ((val (lower-expr (first operands)))
                       (shift (lower-expr (second operands))))
                   (emit-ssa :shift (list val shift))))

         ;; Vector element access: (aref vec i)
         (aref
          (let ((vec (first operands))
                (idx (second operands)))
            (emit-ssa :load-element (list vec (lower-expr idx)))))

         ;; Deterministic reduction: (reduce-sum vec len)
         (reduce-sum
          (let ((vec (first operands))
                (len (second operands)))
            (lower-reduce :field-add vec len)))

         ;; Element-wise map: (map-kernel (x vec) body)
         ;; This doesn't lower inline — it marks a parallel region
         (map-kernel
          (error "map-kernel must be the top-level form in defkernel"))

         (otherwise
          (error "Unknown operation in kernel: ~A" op)))))

    (t (error "Cannot lower expression: ~S" expr))))

(defun lower-binop (op operands)
  "Lower a binary operation, chaining for >2 operands.
   (+ a b c) → t0=a+b, t1=t0+c — strict left-to-right."
  (let ((result (lower-expr (first operands))))
    (dolist (operand (rest operands) result)
      (let ((right (lower-expr operand)))
        (setf result (emit-ssa op (list result right)
                               :ctype (if (member op '(:field-add :field-mul :field-sub))
                                          :field :u64)))))))

;;; ═══════════════════════════════════════════════════════════════════
;;; 3. Deterministic reduction tree
;;;
;;; Instead of letting the GPU reduce in arbitrary order:
;;;   t = v[0]; for i=1..n: t += v[i]  ← order depends on thread scheduling
;;;
;;; We emit a balanced binary tree:
;;;   level 0:  s0=v[0]+v[1]  s1=v[2]+v[3]  s2=v[4]+v[5]  s3=v[6]+v[7]
;;;   level 1:  s4=s0+s1      s5=s2+s3
;;;   level 2:  s6=s4+s5
;;;
;;; The tree shape is fixed at compile time from the declared length.
;;; Every run produces the same binary tree, same evaluation order,
;;; same result — even on different GPUs.
;;; ═══════════════════════════════════════════════════════════════════

(defun lower-reduce (op vec len)
  "Emit a balanced binary reduction tree of LEN elements from VEC."
  (unless (and (integerp len) (plusp len))
    (error "reduce length must be a positive integer literal, got ~S" len))

  ;; Load all elements
  (let ((leaves (loop for i below len
                      collect (emit-ssa :load-element
                                        (list vec i)
                                        :ctype :field))))
    ;; Build balanced binary tree bottom-up
    (loop while (> (length leaves) 1)
          do (setf leaves
                   (loop for (a b) on leaves by #'cddr
                         collect (if b
                                     (emit-ssa op (list a b) :ctype :field)
                                     a))))
    (first leaves)))

;;; ═══════════════════════════════════════════════════════════════════
;;; 4. SSA → OpenCL C code generation
;;;
;;; Each SSA node becomes one line of C.
;;; Field arithmetic uses explicit mod operations.
;;; ═══════════════════════════════════════════════════════════════════

(defparameter *field-preamble*
  "
#define GP 0xFFFFFFFF00000001UL
#define EPSILON 0xFFFFFFFFUL

inline ulong fp_add(ulong a, ulong b) {
  ulong r = a + b;
  ulong c = (r < a);
  if (c) return r + EPSILON;
  return (r >= GP) ? r - GP : r;
}

inline ulong fp_sub(ulong a, ulong b) {
  if (a >= b) return a - b;
  return GP - (b - a);
}

inline ulong fp_mul(ulong a, ulong b) {
  ulong lo = a * b;
  ulong hi = mul_hi(a, b);
  ulong hi_l = hi & EPSILON;
  ulong hi_h = hi >> 32;
  ulong term = hi_l * EPSILON;
  ulong r = lo + term;
  ulong over = (r < lo) ? 1UL : 0UL;
  ulong under = (hi_h > r) ? 1UL : 0UL;
  r -= hi_h;
  if (over > under) {
    r += EPSILON;
    if (r < EPSILON) r += EPSILON;
  } else if (under > over) {
    if (r >= EPSILON) r -= EPSILON;
    else r += (GP - EPSILON);
  }
  if (r >= GP) r -= GP;
  return r;
}
"
  "OpenCL C helper functions for Goldilocks field arithmetic.
   Uses mul_hi() instead of __int128 for GPU compatibility.")

(defun ssa-name-to-c (name)
  "Convert SSA name :T0 to C variable name \"t0\"."
  (string-downcase (symbol-name name)))

(defun c-type-for (ctype)
  (ecase ctype (:u64 "ulong") (:field "ulong")))

(defun emit-c-node (node stream)
  "Emit one SSA node as a C statement."
  (let ((var (ssa-name-to-c (ssa-node-name node)))
        (ct (c-type-for (ssa-node-ctype node)))
        (args (ssa-node-args node)))
    (ecase (ssa-node-op node)
      (:const
       (format stream "  ~A ~A = ~DUL;~%" ct var (first args)))

      (:load-scalar
       (format stream "  ~A ~A = ~A;~%" ct var
               (sanitize-c-name (first args))))

      (:load-vec
       (format stream "  ~A ~A = ~A[gid];~%" ct var
               (sanitize-c-name (first args))))

      (:load-element
       (let ((vec (first args))
             (idx (second args)))
         (if (integerp idx)
             (format stream "  ~A ~A = ~A[~D];~%" ct var
                     (sanitize-c-name vec) idx)
             (format stream "  ~A ~A = ~A[~A];~%" ct var
                     (sanitize-c-name vec)
                     (ssa-name-to-c idx)))))

      ;; Wrapping u64 arithmetic
      (:add (format stream "  ~A ~A = ~A + ~A;~%" ct var
                    (ssa-name-to-c (first args))
                    (ssa-name-to-c (second args))))
      (:sub (format stream "  ~A ~A = ~A - ~A;~%" ct var
                    (ssa-name-to-c (first args))
                    (ssa-name-to-c (second args))))
      (:mul (format stream "  ~A ~A = ~A * ~A;~%" ct var
                    (ssa-name-to-c (first args))
                    (ssa-name-to-c (second args))))

      ;; Bitwise
      (:and (format stream "  ~A ~A = ~A & ~A;~%" ct var
                    (ssa-name-to-c (first args))
                    (ssa-name-to-c (second args))))
      (:xor (format stream "  ~A ~A = ~A ^ ~A;~%" ct var
                    (ssa-name-to-c (first args))
                    (ssa-name-to-c (second args))))
      (:or  (format stream "  ~A ~A = ~A | ~A;~%" ct var
                    (ssa-name-to-c (first args))
                    (ssa-name-to-c (second args))))
      (:shift (format stream "  ~A ~A = ~A << ~A;~%" ct var
                      (ssa-name-to-c (first args))
                      (ssa-name-to-c (second args))))

      ;; Field arithmetic: call helper functions (no __int128 needed)
      (:field-add
       (format stream "  ~A ~A = fp_add(~A, ~A);~%"
               ct var
               (ssa-name-to-c (first args))
               (ssa-name-to-c (second args))))
      (:field-sub
       (format stream "  ~A ~A = fp_sub(~A, ~A);~%"
               ct var
               (ssa-name-to-c (first args))
               (ssa-name-to-c (second args))))
      (:field-mul
       (format stream "  ~A ~A = fp_mul(~A, ~A);~%"
               ct var
               (ssa-name-to-c (first args))
               (ssa-name-to-c (second args)))))))

;;; ═══════════════════════════════════════════════════════════════════
;;; 5. Kernel compiler: AST → SSA → OpenCL C
;;; ═══════════════════════════════════════════════════════════════════

(defstruct kernel
  (name nil :type symbol)
  (params nil :type list)           ; ((name type) ...) where type = :scalar | :vec | :out
  (source "" :type string)          ; generated OpenCL C source
  (ssa nil :type list))             ; SSA nodes for inspection

(defun sanitize-c-name (name)
  "Turn a Lisp symbol name into a valid C identifier.
   *foo-bar* → foo_bar, MY-KERNEL → my_kernel"
  (let* ((s (string-downcase (symbol-name name)))
         (s (remove #\* s)))
    (substitute #\_ #\- s)))

(defun compile-kernel (name params body)
  "Compile a kernel body to OpenCL C.
   PARAMS: list of (name type) where type is :scalar, :vec, or :out.
   BODY: a Lisp expression using field arithmetic.
   Returns a KERNEL struct."
  (let ((*ssa-counter* 0)
        (*ssa-nodes* nil)
        (*ssa-inputs* (mapcar (lambda (p) (list (first p) (second p))) params)))

    ;; Lower body to SSA
    (let ((result-name (lower-expr body)))

      ;; Reverse SSA nodes (they were pushed in reverse order)
      (let ((nodes (nreverse *ssa-nodes*)))

        ;; Generate OpenCL C
        (let* ((has-field (some (lambda (n)
                                  (member (ssa-node-op n)
                                          '(:field-add :field-sub :field-mul)))
                                nodes))
               (source (with-output-to-string (s)
                        ;; Preamble for field arithmetic helpers
                        (when has-field
                          (write-string *field-preamble* s))
                        ;; Kernel signature
                        (format s "__kernel void ~A(~{~A~^, ~}) {~%"
                                (sanitize-c-name name)
                                (mapcar #'param-to-c params))
                        (format s "  size_t gid = get_global_id(0);~%")

                        ;; SSA statements
                        (dolist (node nodes)
                          (emit-c-node node s))

                        ;; Write result to output
                        (let ((out-param (find :out params :key #'second)))
                          (when out-param
                            (format s "  ~A[gid] = ~A;~%"
                                    (sanitize-c-name (first out-param))
                                    (ssa-name-to-c result-name))))

                        (format s "}~%"))))

          (make-kernel :name name
                       :params params
                       :source source
                       :ssa nodes))))))

(defun param-to-c (param)
  "Convert a kernel parameter spec to C syntax."
  (let ((name (sanitize-c-name (first param)))
        (type (second param)))
    (ecase type
      (:scalar (format nil "ulong ~A" name))
      (:vec    (format nil "__global const ulong* ~A" name))
      (:out    (format nil "__global ulong* ~A" name)))))

;;; ═══════════════════════════════════════════════════════════════════
;;; 6. defkernel macro — the user-facing API
;;;
;;; (defkernel vec-dot ((a :vec) (b :vec) (out :out) (n :scalar))
;;;   (reduce-sum ...))
;;;
;;; Expands to a variable holding the compiled KERNEL struct,
;;; with the OpenCL source ready to dispatch.
;;; ═══════════════════════════════════════════════════════════════════

(defmacro defkernel (name params &body body)
  "Define a GPU kernel. Compiles Lisp arithmetic to OpenCL C at macroexpansion time.

   PARAMS: list of (name type) — :scalar for single values,
           :vec for input arrays, :out for the output array.

   BODY: Lisp expression using:
     (f+ a b)       — field addition mod P
     (f* a b)       — field multiplication mod P
     (f- a b)       — field subtraction mod P
     (reduce-sum v n) — deterministic reduction (balanced binary tree)
     (aref v i)     — element access

   The body is compiled to OpenCL C with a fixed evaluation order.
   The generated code is deterministic across all GPU hardware."
  (let ((kernel (compile-kernel name params (first body))))
    `(defparameter ,name
       (make-kernel :name ',name
                    :params ',params
                    :source ,(kernel-source kernel)
                    :ssa ',(kernel-ssa kernel)))))

;; kernel-source accessor is auto-generated by defstruct

;;; ═══════════════════════════════════════════════════════════════════
;;; 7. Tests
;;; ═══════════════════════════════════════════════════════════════════

(defvar *test-pass* 0)
(defvar *test-fail* 0)

(defun check (name ok &optional detail)
  (if ok
      (progn (incf *test-pass*) (format t "  ~36A PASS~%" name))
      (progn (incf *test-fail*)
             (format t "  ~36A FAIL~%" name)
             (when detail (format t "    ~A~%" detail)))))

(defun run-tests ()
  (setf *test-pass* 0 *test-fail* 0)

  (format t "~%══ defkernel: Deterministic Kernel Compiler ══~%")

  ;; ── Test 1: Simple field multiply ──────────────────
  (format t "~%=== Scalar Kernels ===~%")

  (let ((k (compile-kernel 'field-square
                           '((x :vec) (out :out))
                           '(f* x x))))
    (check "field-square compiles" (kernel-p k))
    (check "source contains kernel"
           (search "__kernel void" (kernel-source k)))
    (check "source uses fp_mul helper"
           (search "fp_mul" (kernel-source k)))
    (check "source has mod P"
           (search "0xFFFFFFFF00000001" (kernel-source k)))
    (check "source writes to out[gid]"
           (search "out[gid]" (kernel-source k)))
    (format t "~%  Generated:~%~A~%" (kernel-source k)))

  ;; ── Test 2: Multi-operand chains ─────────────────
  (let ((k (compile-kernel 'field-triple-add
                           '((a :vec) (b :vec) (c :vec) (out :out))
                           '(f+ a b c))))
    (check "triple-add compiles" (kernel-p k))
    ;; Should produce t0=a, t1=b, t2=a+b, t3=c, t4=t2+c  (left-to-right)
    (let* ((nodes (kernel-ssa k))
           (ops (mapcar #'ssa-node-op nodes)))
      (check "triple-add: 2 field-add ops"
             (= (count :field-add ops) 2))
      (check "triple-add: left-to-right chain"
             ;; The second add must reference the first add's result
             (let ((adds (remove-if-not (lambda (n) (eq (ssa-node-op n) :field-add)) nodes)))
               (and (= (length adds) 2)
                    (member (ssa-node-name (first adds))
                            (ssa-node-args (second adds)))))))
    (format t "~%  Generated:~%~A~%" (kernel-source k)))

  ;; ── Test 3: Nested expression ──────────────────────
  (let ((k (compile-kernel 'dot2
                           '((a :vec) (b :vec) (c :vec) (d :vec) (out :out))
                           '(f+ (f* a b) (f* c d)))))
    (check "dot2 compiles" (kernel-p k))
    (let* ((nodes (kernel-ssa k))
           (ops (mapcar #'ssa-node-op nodes)))
      (check "dot2: 2 field-mul + 1 field-add"
             (and (= (count :field-mul ops) 2)
                  (= (count :field-add ops) 1))))
    (format t "~%  Generated:~%~A~%" (kernel-source k)))

  ;; ── Test 4: Deterministic reduction ────────────────
  (format t "~%=== Reduction Kernels ===~%")

  (let ((k (compile-kernel 'sum-8
                           '((v :vec) (out :out))
                           '(reduce-sum v 8))))
    (check "sum-8 compiles" (kernel-p k))
    (let* ((nodes (kernel-ssa k))
           (ops (mapcar #'ssa-node-op nodes)))
      ;; 8 loads + 7 adds (balanced binary tree: 4+2+1)
      (check "sum-8: 8 element loads"
             (= (count :load-element ops) 8))
      (check "sum-8: 7 field-adds (balanced tree)"
             (= (count :field-add ops) 7)))
    (format t "~%  Generated:~%~A~%" (kernel-source k)))

  ;; Verify tree structure: level 0 has 4 adds, level 1 has 2, level 2 has 1
  (let ((k (compile-kernel 'sum-4
                           '((v :vec) (out :out))
                           '(reduce-sum v 4))))
    (let* ((nodes (kernel-ssa k))
           (adds (remove-if-not (lambda (n) (eq (ssa-node-op n) :field-add)) nodes)))
      ;; 4 loads → 2 adds → 1 add = 3 total adds
      (check "sum-4: 3 adds (balanced 4→2→1)"
             (= (length adds) 3))
      ;; The last add must reference the two previous adds
      (let ((last-add (car (last adds))))
        (check "sum-4: root references both subtrees"
               (every (lambda (arg)
                        (some (lambda (a) (eq (ssa-node-name a) arg))
                              (butlast adds)))
                      (ssa-node-args last-add))))))

  ;; ── Test 5: Odd-length reduction ────────────────────
  (let ((k (compile-kernel 'sum-5
                           '((v :vec) (out :out))
                           '(reduce-sum v 5))))
    (let* ((nodes (kernel-ssa k))
           (adds (remove-if-not (lambda (n) (eq (ssa-node-op n) :field-add)) nodes)))
      ;; 5 elements: pairs (0,1)(2,3) + lone 4 → 3 results → pair + lone → final
      (check "sum-5: 4 adds (handles odd count)"
             (= (length adds) 4)))
    (format t "~%  Generated:~%~A~%" (kernel-source k)))

  ;; ── Test 6: Determinism — same AST always gives same SSA ───
  (format t "~%=== Determinism ===~%")

  (let ((sources (loop repeat 100
                       collect (kernel-source
                                (compile-kernel 'det-test
                                                '((a :vec) (b :vec) (out :out))
                                                '(f+ (f* a a) (f* b b)))))))
    (check "100 compilations → identical source"
           (every (lambda (s) (string= s (first sources))) (rest sources))))

  (let ((sources (loop repeat 50
                       collect (kernel-source
                                (compile-kernel 'det-reduce
                                                '((v :vec) (out :out))
                                                '(reduce-sum v 16))))))
    (check "50 reduce-16 compilations → identical"
           (every (lambda (s) (string= s (first sources))) (rest sources))))

  ;; ── Test 7: Mixed expressions ───────────────────────
  (format t "~%=== Mixed Expressions ===~%")

  ;; Polynomial: c0 + c1*x + c2*x*x
  (let ((k (compile-kernel 'poly-eval
                           '((x :vec) (c0 :scalar) (c1 :scalar) (c2 :scalar) (out :out))
                           '(f+ c0 (f+ (f* c1 x) (f* c2 (f* x x)))))))
    (check "poly-eval compiles" (kernel-p k))
    (let* ((nodes (kernel-ssa k))
           (ops (mapcar #'ssa-node-op nodes)))
      (check "poly-eval: 3 muls + 2 adds"
             (and (= (count :field-mul ops) 3)
                  (= (count :field-add ops) 2))))
    (format t "~%  Generated:~%~A~%" (kernel-source k)))

  ;; ── Results ──────────────────────────────────────────
  (format t "═══════════════════════════════════════════~%")
  (format t "Total: ~D passed, ~D failed~%" *test-pass* *test-fail*)
  (if (zerop *test-fail*)
      (format t "~%  *** ALL TESTS PASSED ***~%")
      (format t "~%  *** ~D TESTS FAILED ***~%" *test-fail*))
  (format t "═══════════════════════════════════════════~%")
  (zerop *test-fail*))

;;; To run tests: sbcl --load tests.lisp
