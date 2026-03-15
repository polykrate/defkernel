(defsystem "defkernel"
  :description "Deterministic GPU kernels from Lisp — AST → SSA → OpenCL C compiler with STARK prover"
  :version "0.2.0"
  :author "polykrate"
  :license "MIT"
  :depends-on ("cffi")
  :serial t
  :components ((:file "defkernel")
               (:file "runtime")
               (:file "ntt")
               (:file "fri")
               (:file "stark")))
