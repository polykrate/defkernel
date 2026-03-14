;;;; runtime.lisp — OpenCL runtime via CFFI for cl-gpu kernels
;;;;
;;;; Provides: gpu-init, gpu-execute, gpu-cleanup, gpu-available-p
;;;;
;;;; Usage:
;;;;   (ql:quickload :cffi)            ; if needed
;;;;   (load "cl-gpu.lisp")
;;;;   (load "runtime.lisp")
;;;;   (cl-gpu:gpu-init)
;;;;   (cl-gpu:gpu-execute *my-kernel* :global-size 1024
;;;;     :inputs '((:a #(1 2 3 ...)) (:b #(4 5 6 ...)))
;;;;     :scalars '((:n . 1024))
;;;;     :output-size 1024)
;;;;   (cl-gpu:gpu-cleanup)

(in-package #:cl-gpu)

;;; ═══════════════════════════════════════════════════════════════════
;;; 0. Mark runtime as loaded (prevents auto-test on cl-gpu.lisp load)
;;; ═══════════════════════════════════════════════════════════════════

(eval-when (:compile-toplevel :load-toplevel :execute)
  (unless (find-package :cl-gpu/runtime)
    (defpackage #:cl-gpu/runtime)))

;;; ═══════════════════════════════════════════════════════════════════
;;; 1. Load CFFI and define OpenCL foreign library
;;; ═══════════════════════════════════════════════════════════════════

(eval-when (:compile-toplevel :load-toplevel :execute)
  (require :asdf)
  (asdf:load-system :cffi))

(cffi:define-foreign-library libopencl
  (:unix "libOpenCL.so")
  (:darwin (:framework "OpenCL"))
  (t "libOpenCL"))

;;; ═══════════════════════════════════════════════════════════════════
;;; 2. OpenCL constants
;;; ═══════════════════════════════════════════════════════════════════

(defconstant +cl-success+ 0)
(defconstant +cl-device-type-gpu+ (ash 1 2))
(defconstant +cl-device-type-cpu+ (ash 1 1))
(defconstant +cl-device-type-all+ #xFFFFFFFF)

(defconstant +cl-mem-read-write+  (ash 1 0))
(defconstant +cl-mem-write-only+  (ash 1 1))
(defconstant +cl-mem-read-only+   (ash 1 2))

(defconstant +cl-device-name+     #x102B)
(defconstant +cl-platform-name+   #x0902)
(defconstant +cl-device-vendor+   #x102C)
(defconstant +cl-device-type+     #x1000)
(defconstant +cl-device-max-compute-units+ #x1002)
(defconstant +cl-device-global-mem-size+   #x101F)

(defconstant +cl-program-build-log+ #x1183)

;;; ═══════════════════════════════════════════════════════════════════
;;; 3. CFFI bindings for OpenCL C API
;;; ═══════════════════════════════════════════════════════════════════

(cffi:defcfun ("clGetPlatformIDs" %cl-get-platform-ids) :int32
  (num-entries :uint32)
  (platforms :pointer)
  (num-platforms :pointer))

(cffi:defcfun ("clGetPlatformInfo" %cl-get-platform-info) :int32
  (platform :pointer)
  (param-name :uint32)
  (param-value-size :size)
  (param-value :pointer)
  (param-value-size-ret :pointer))

(cffi:defcfun ("clGetDeviceIDs" %cl-get-device-ids) :int32
  (platform :pointer)
  (device-type :uint64)
  (num-entries :uint32)
  (devices :pointer)
  (num-devices :pointer))

(cffi:defcfun ("clGetDeviceInfo" %cl-get-device-info) :int32
  (device :pointer)
  (param-name :uint32)
  (param-value-size :size)
  (param-value :pointer)
  (param-value-size-ret :pointer))

(cffi:defcfun ("clCreateContext" %cl-create-context) :pointer
  (properties :pointer)
  (num-devices :uint32)
  (devices :pointer)
  (pfn-notify :pointer)
  (user-data :pointer)
  (errcode-ret :pointer))

(cffi:defcfun ("clCreateCommandQueue" %cl-create-command-queue) :pointer
  (context :pointer)
  (device :pointer)
  (properties :uint64)
  (errcode-ret :pointer))

(cffi:defcfun ("clCreateProgramWithSource" %cl-create-program-with-source) :pointer
  (context :pointer)
  (count :uint32)
  (strings :pointer)
  (lengths :pointer)
  (errcode-ret :pointer))

(cffi:defcfun ("clBuildProgram" %cl-build-program) :int32
  (program :pointer)
  (num-devices :uint32)
  (device-list :pointer)
  (options :string)
  (pfn-notify :pointer)
  (user-data :pointer))

(cffi:defcfun ("clGetProgramBuildInfo" %cl-get-program-build-info) :int32
  (program :pointer)
  (device :pointer)
  (param-name :uint32)
  (param-value-size :size)
  (param-value :pointer)
  (param-value-size-ret :pointer))

(cffi:defcfun ("clCreateKernel" %cl-create-kernel) :pointer
  (program :pointer)
  (kernel-name :string)
  (errcode-ret :pointer))

(cffi:defcfun ("clCreateBuffer" %cl-create-buffer) :pointer
  (context :pointer)
  (flags :uint64)
  (size :size)
  (host-ptr :pointer)
  (errcode-ret :pointer))

(cffi:defcfun ("clSetKernelArg" %cl-set-kernel-arg) :int32
  (kernel :pointer)
  (arg-index :uint32)
  (arg-size :size)
  (arg-value :pointer))

(cffi:defcfun ("clEnqueueWriteBuffer" %cl-enqueue-write-buffer) :int32
  (command-queue :pointer)
  (buffer :pointer)
  (blocking-write :uint32)
  (offset :size)
  (size :size)
  (ptr :pointer)
  (num-events :uint32)
  (event-wait-list :pointer)
  (event :pointer))

(cffi:defcfun ("clEnqueueNDRangeKernel" %cl-enqueue-nd-range-kernel) :int32
  (command-queue :pointer)
  (kernel :pointer)
  (work-dim :uint32)
  (global-work-offset :pointer)
  (global-work-size :pointer)
  (local-work-size :pointer)
  (num-events :uint32)
  (event-wait-list :pointer)
  (event :pointer))

(cffi:defcfun ("clEnqueueReadBuffer" %cl-enqueue-read-buffer) :int32
  (command-queue :pointer)
  (buffer :pointer)
  (blocking-read :uint32)
  (offset :size)
  (size :size)
  (ptr :pointer)
  (num-events :uint32)
  (event-wait-list :pointer)
  (event :pointer))

(cffi:defcfun ("clFinish" %cl-finish) :int32
  (command-queue :pointer))

(cffi:defcfun ("clReleaseMemObject" %cl-release-mem-object) :int32
  (memobj :pointer))

(cffi:defcfun ("clReleaseKernel" %cl-release-kernel) :int32
  (kernel :pointer))

(cffi:defcfun ("clReleaseProgram" %cl-release-program) :int32
  (program :pointer))

(cffi:defcfun ("clReleaseCommandQueue" %cl-release-command-queue) :int32
  (command-queue :pointer))

(cffi:defcfun ("clReleaseContext" %cl-release-context) :int32
  (context :pointer))

;;; ═══════════════════════════════════════════════════════════════════
;;; 4. GPU state
;;; ═══════════════════════════════════════════════════════════════════

(defstruct gpu-state
  (platform (cffi:null-pointer) :type cffi:foreign-pointer)
  (device   (cffi:null-pointer) :type cffi:foreign-pointer)
  (context  (cffi:null-pointer) :type cffi:foreign-pointer)
  (queue    (cffi:null-pointer) :type cffi:foreign-pointer)
  (device-name "" :type string)
  (program-cache (make-hash-table :test #'equal) :type hash-table))

(defvar *gpu* nil "Current GPU state, or NIL if not initialized.")

;;; ═══════════════════════════════════════════════════════════════════
;;; 5. Error handling
;;; ═══════════════════════════════════════════════════════════════════

(define-condition opencl-error (error)
  ((code :initarg :code :reader opencl-error-code)
   (context :initarg :context :reader opencl-error-context))
  (:report (lambda (c s)
             (format s "OpenCL error ~D in ~A"
                     (opencl-error-code c) (opencl-error-context c)))))

(defmacro cl-check (form context)
  "Check OpenCL return code. Signal error if not CL_SUCCESS."
  (let ((rc (gensym "RC")))
    `(let ((,rc ,form))
       (unless (= ,rc +cl-success+)
         (error 'opencl-error :code ,rc :context ,context))
       ,rc)))

;;; ═══════════════════════════════════════════════════════════════════
;;; 6. Query helpers
;;; ═══════════════════════════════════════════════════════════════════

(defun query-platform-string (platform param)
  (cffi:with-foreign-object (size :size)
    (%cl-get-platform-info platform param 0 (cffi:null-pointer) size)
    (let ((n (cffi:mem-ref size :size)))
      (cffi:with-foreign-pointer-as-string (buf n)
        (%cl-get-platform-info platform param n buf (cffi:null-pointer))))))

(defun query-device-string (device param)
  (cffi:with-foreign-object (size :size)
    (%cl-get-device-info device param 0 (cffi:null-pointer) size)
    (let ((n (cffi:mem-ref size :size)))
      (cffi:with-foreign-pointer-as-string (buf n)
        (%cl-get-device-info device param n buf (cffi:null-pointer))))))

(defun query-device-ulong (device param)
  (cffi:with-foreign-object (val :uint64)
    (%cl-get-device-info device param 8 val (cffi:null-pointer))
    (cffi:mem-ref val :uint64)))

;;; ═══════════════════════════════════════════════════════════════════
;;; 7. Initialization
;;; ═══════════════════════════════════════════════════════════════════

(declaim (ftype function gpu-cleanup))

(defun gpu-available-p ()
  "Return T if OpenCL is available and at least one device exists."
  (handler-case
      (progn
        (cffi:load-foreign-library 'libopencl)
        (cffi:with-foreign-object (count :uint32)
          (let ((rc (%cl-get-platform-ids 0 (cffi:null-pointer) count)))
            (and (= rc +cl-success+)
                 (plusp (cffi:mem-ref count :uint32))))))
    (error () nil)))

(defun gpu-init (&key (prefer-gpu t))
  "Initialize OpenCL. Find a device, create context and command queue.
   Returns the GPU-STATE. Prefers GPU device; falls back to CPU if PREFER-GPU is NIL
   or no GPU is found."
  (when *gpu*
    (gpu-cleanup))

  (cffi:load-foreign-library 'libopencl)

  ;; Find platform
  (cffi:with-foreign-objects ((num-platforms :uint32)
                              (platform-buf :pointer 8))
    (cl-check (%cl-get-platform-ids 8 platform-buf num-platforms)
              "clGetPlatformIDs")
    (let ((n-plat (cffi:mem-ref num-platforms :uint32)))
      (when (zerop n-plat)
        (error "No OpenCL platforms found. Install an OpenCL ICD driver."))

      (format t "~%Found ~D OpenCL platform~:P:~%" n-plat)

      ;; Try each platform for a GPU, then fall back to CPU
      (let ((chosen-platform nil)
            (chosen-device nil))

        (loop for plat-idx below n-plat
              for plat = (cffi:mem-aref platform-buf :pointer plat-idx)
              for pname = (query-platform-string plat +cl-platform-name+)
              do (format t "  Platform ~D: ~A~%" plat-idx pname)
              until chosen-device
              do
                 ;; Try GPU first
                 (when prefer-gpu
                   (cffi:with-foreign-objects ((nd :uint32)
                                              (dev :pointer))
                     (let ((rc (%cl-get-device-ids plat +cl-device-type-gpu+
                                                   1 dev nd)))
                       (when (= rc +cl-success+)
                         (setf chosen-platform plat
                               chosen-device (cffi:mem-ref dev :pointer))))))
                 ;; Fall back to CPU
                 (unless chosen-device
                   (cffi:with-foreign-objects ((nd :uint32)
                                              (dev :pointer))
                     (let ((rc (%cl-get-device-ids plat +cl-device-type-cpu+
                                                   1 dev nd)))
                       (when (= rc +cl-success+)
                         (setf chosen-platform plat
                               chosen-device (cffi:mem-ref dev :pointer)))))))

        (unless chosen-device
          (error "No OpenCL devices found. Install a GPU/CPU OpenCL driver.~%~
                  For Intel integrated GPU: sudo pacman -S intel-compute-runtime"))

        (let* ((dev-name (query-device-string chosen-device +cl-device-name+))
               (dev-vendor (query-device-string chosen-device +cl-device-vendor+))
               (dev-mem (query-device-ulong chosen-device +cl-device-global-mem-size+)))

          (format t "~%  Device: ~A (~A)~%" dev-name dev-vendor)
          (format t "  Memory: ~,1F MB~%" (/ dev-mem 1048576.0))

          ;; Create context
          (cffi:with-foreign-objects ((err :int32)
                                     (dev-ptr :pointer))
            (setf (cffi:mem-ref dev-ptr :pointer) chosen-device)

            (let ((ctx (%cl-create-context (cffi:null-pointer)
                                           1 dev-ptr
                                           (cffi:null-pointer)
                                           (cffi:null-pointer)
                                           err)))
              (cl-check (cffi:mem-ref err :int32) "clCreateContext")

              ;; Create command queue
              (let ((queue (%cl-create-command-queue ctx chosen-device 0 err)))
                (cl-check (cffi:mem-ref err :int32) "clCreateCommandQueue")

                (setf *gpu*
                      (make-gpu-state :platform chosen-platform
                                      :device chosen-device
                                      :context ctx
                                      :queue queue
                                      :device-name dev-name))

                (format t "~%  OpenCL ready.~%")
                *gpu*))))))))

;;; ═══════════════════════════════════════════════════════════════════
;;; 8. Program compilation (with caching)
;;; ═══════════════════════════════════════════════════════════════════

(defun get-build-log (program device)
  (cffi:with-foreign-object (log-size :size)
    (%cl-get-program-build-info program device +cl-program-build-log+
                                0 (cffi:null-pointer) log-size)
    (let ((n (cffi:mem-ref log-size :size)))
      (if (zerop n) ""
          (cffi:with-foreign-pointer-as-string (buf n)
            (%cl-get-program-build-info program device +cl-program-build-log+
                                        n buf (cffi:null-pointer)))))))

(defun compile-opencl-source (source)
  "Compile OpenCL C source string. Returns CL program handle.
   Caches by source string."
  (unless *gpu* (error "GPU not initialized. Call (gpu-init) first."))

  (let ((cached (gethash source (gpu-state-program-cache *gpu*))))
    (when cached (return-from compile-opencl-source cached)))

  (cffi:with-foreign-objects ((err :int32)
                              (src-ptr :pointer)
                              (len-ptr :size))
    (let ((c-source (cffi:foreign-string-alloc source)))
      (unwind-protect
           (progn
             (setf (cffi:mem-ref src-ptr :pointer) c-source)
             (setf (cffi:mem-ref len-ptr :size) (length source))

             (let ((program (%cl-create-program-with-source
                             (gpu-state-context *gpu*)
                             1 src-ptr len-ptr err)))
               (cl-check (cffi:mem-ref err :int32) "clCreateProgramWithSource")

               (let ((rc (%cl-build-program program 0 (cffi:null-pointer)
                                            "-cl-std=CL1.2"
                                            (cffi:null-pointer)
                                            (cffi:null-pointer))))
                 (unless (= rc +cl-success+)
                   (let ((log (get-build-log program (gpu-state-device *gpu*))))
                     (%cl-release-program program)
                     (error "OpenCL build failed (code ~D):~%~A~%~%Source:~%~A"
                            rc log source)))

                 (setf (gethash source (gpu-state-program-cache *gpu*)) program)
                 program)))
        (cffi:foreign-string-free c-source)))))

;;; ═══════════════════════════════════════════════════════════════════
;;; 9. Kernel execution
;;; ═══════════════════════════════════════════════════════════════════

(defun gpu-execute (kernel &key (global-size 1) inputs scalars output-size)
  "Execute a compiled CL-GPU kernel on the GPU.

   KERNEL: a KERNEL struct from defkernel or compile-kernel.
   GLOBAL-SIZE: number of work items (parallel threads).
   INPUTS: alist of (param-name . #(u64-values...)) for :vec params.
   SCALARS: alist of (param-name . u64-value) for :scalar params.
   OUTPUT-SIZE: number of u64 elements in the output buffer.

   Returns a vector of (unsigned-byte 64) with OUTPUT-SIZE elements."
  (unless *gpu* (error "GPU not initialized. Call (gpu-init) first."))

  (let* ((source (kernel-source kernel))
         (program (compile-opencl-source source))
         (k-name (sanitize-c-name (kernel-name kernel)))
         (params (kernel-params kernel))
         (buffers nil))

    (cffi:with-foreign-object (err :int32)
      (let ((cl-kernel (%cl-create-kernel program k-name err)))
        (cl-check (cffi:mem-ref err :int32) "clCreateKernel")

        (unwind-protect
             (progn
               ;; Set arguments in parameter order
               (loop for (pname ptype) in params
                     for arg-index from 0
                     do (ecase ptype
                          (:vec
                           (let* ((data (cdr (assoc pname inputs)))
                                  (n (length data))
                                  (byte-size (* n 8)))
                             (unless data
                               (error "Missing input data for param ~A" pname))
                             (let ((buf (%cl-create-buffer
                                         (gpu-state-context *gpu*)
                                         +cl-mem-read-only+
                                         byte-size
                                         (cffi:null-pointer) err)))
                               (cl-check (cffi:mem-ref err :int32) "clCreateBuffer(vec)")
                               (push buf buffers)

                               ;; Upload data
                               (cffi:with-foreign-object (host :uint64 n)
                                 (loop for i below n
                                       do (setf (cffi:mem-aref host :uint64 i)
                                                (aref data i)))
                                 (cl-check
                                  (%cl-enqueue-write-buffer
                                   (gpu-state-queue *gpu*)
                                   buf 1 0 byte-size host 0
                                   (cffi:null-pointer) (cffi:null-pointer))
                                  "clEnqueueWriteBuffer"))

                               ;; Set kernel arg (pointer to cl_mem)
                               (cffi:with-foreign-object (buf-ptr :pointer)
                                 (setf (cffi:mem-ref buf-ptr :pointer) buf)
                                 (cl-check
                                  (%cl-set-kernel-arg
                                   cl-kernel arg-index
                                   (cffi:foreign-type-size :pointer) buf-ptr)
                                  "clSetKernelArg(vec)")))))

                          (:out
                           (let* ((n (or output-size global-size))
                                  (byte-size (* n 8)))
                             (let ((buf (%cl-create-buffer
                                         (gpu-state-context *gpu*)
                                         +cl-mem-write-only+
                                         byte-size
                                         (cffi:null-pointer) err)))
                               (cl-check (cffi:mem-ref err :int32) "clCreateBuffer(out)")
                               (push (cons :output buf) buffers)

                               (cffi:with-foreign-object (buf-ptr :pointer)
                                 (setf (cffi:mem-ref buf-ptr :pointer) buf)
                                 (cl-check
                                  (%cl-set-kernel-arg
                                   cl-kernel arg-index
                                   (cffi:foreign-type-size :pointer) buf-ptr)
                                  "clSetKernelArg(out)")))))

                          (:scalar
                           (let ((val (cdr (assoc pname scalars))))
                             (unless val
                               (error "Missing scalar value for param ~A" pname))
                             (cffi:with-foreign-object (val-ptr :uint64)
                               (setf (cffi:mem-ref val-ptr :uint64) val)
                               (cl-check
                                (%cl-set-kernel-arg
                                 cl-kernel arg-index 8 val-ptr)
                                "clSetKernelArg(scalar)"))))))

               ;; Enqueue kernel
               (cffi:with-foreign-object (gws :size)
                 (setf (cffi:mem-ref gws :size) global-size)
                 (cl-check
                  (%cl-enqueue-nd-range-kernel
                   (gpu-state-queue *gpu*)
                   cl-kernel 1
                   (cffi:null-pointer) gws (cffi:null-pointer)
                   0 (cffi:null-pointer) (cffi:null-pointer))
                  "clEnqueueNDRangeKernel"))

               ;; Wait for completion
               (cl-check (%cl-finish (gpu-state-queue *gpu*)) "clFinish")

               ;; Read output
               (let* ((out-entry (find :output buffers :key
                                       (lambda (b) (if (consp b) (car b) nil))))
                      (out-buf (if out-entry (cdr out-entry) nil))
                      (n (or output-size global-size))
                      (result (make-array n :element-type '(unsigned-byte 64))))
                 (when out-buf
                   (cffi:with-foreign-object (host :uint64 n)
                     (cl-check
                      (%cl-enqueue-read-buffer
                       (gpu-state-queue *gpu*)
                       out-buf 1 0 (* n 8) host 0
                       (cffi:null-pointer) (cffi:null-pointer))
                      "clEnqueueReadBuffer")
                     (loop for i below n
                           do (setf (aref result i)
                                    (cffi:mem-aref host :uint64 i)))))
                 result))

          ;; Cleanup: release buffers and kernel
          (dolist (b buffers)
            (let ((mem (if (consp b) (cdr b) b)))
              (%cl-release-mem-object mem)))
          (%cl-release-kernel cl-kernel))))))

;;; ═══════════════════════════════════════════════════════════════════
;;; 10. GPU info and cleanup
;;; ═══════════════════════════════════════════════════════════════════

(defun gpu-info ()
  "Print info about the current GPU state."
  (if *gpu*
      (format t "GPU: ~A~%Programs cached: ~D~%"
              (gpu-state-device-name *gpu*)
              (hash-table-count (gpu-state-program-cache *gpu*)))
      (format t "GPU not initialized.~%")))

(defun gpu-cleanup ()
  "Release all OpenCL resources."
  (when *gpu*
    ;; Release cached programs
    (maphash (lambda (k v)
               (declare (ignore k))
               (%cl-release-program v))
             (gpu-state-program-cache *gpu*))

    (unless (cffi:null-pointer-p (gpu-state-queue *gpu*))
      (%cl-release-command-queue (gpu-state-queue *gpu*)))
    (unless (cffi:null-pointer-p (gpu-state-context *gpu*))
      (%cl-release-context (gpu-state-context *gpu*)))

    (setf *gpu* nil)
    (format t "GPU resources released.~%")))

;;; ═══════════════════════════════════════════════════════════════════
;;; 11. Convenience: run a defkernel on vectors
;;; ═══════════════════════════════════════════════════════════════════

(defun gpu-map (kernel &rest args)
  "Convenience function to map a kernel over input vectors.
   Automatically determines global-size from the first vector argument.

   Example:
     (defkernel *field-add* ((a :vec) (b :vec) (out :out))
       (f+ a b))
     (gpu-map *field-add* :a #(1 2 3) :b #(4 5 6))
     => #(5 7 9)  ; mod P"
  (unless *gpu* (error "GPU not initialized."))

  (let ((params (kernel-params kernel))
        (inputs nil)
        (scalars nil)
        (global-size nil))

    ;; Parse keyword args
    (loop for (key val) on args by #'cddr
          for pname = (intern (symbol-name key) :cl-gpu)
          for param = (find pname params :key #'first)
          do (unless param
               (error "Unknown parameter ~A for kernel ~A" key (kernel-name kernel)))
             (ecase (second param)
               (:vec
                (push (cons pname (coerce val 'vector)) inputs)
                (unless global-size
                  (setf global-size (length val))))
               (:scalar
                (push (cons pname val) scalars))
               (:out nil)))

    (gpu-execute kernel
                 :global-size global-size
                 :inputs inputs
                 :scalars scalars
                 :output-size global-size)))

;;; ═══════════════════════════════════════════════════════════════════
;;; Export runtime symbols
;;; ═══════════════════════════════════════════════════════════════════

(eval-when (:compile-toplevel :load-toplevel :execute)
  (export '(gpu-init gpu-cleanup gpu-info gpu-execute gpu-map
            gpu-available-p gpu-state gpu-state-device-name
            *gpu*)
          :cl-gpu))
