;;;; tests.lisp — Run defkernel compiler tests
;;;; Usage: sbcl --load tests.lisp

(load (merge-pathnames "defkernel.lisp" *load-truename*))

(let ((ok (defkernel::run-tests)))
  (finish-output)
  (sb-ext:exit :code (if ok 0 1)))
