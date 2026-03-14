;;;; tests.lisp — Run cl-gpu compiler tests
;;;; Usage: sbcl --load tests.lisp

(load (merge-pathnames "cl-gpu.lisp" *load-truename*))

(let ((ok (cl-gpu::run-tests)))
  (finish-output)
  (sb-ext:exit :code (if ok 0 1)))
