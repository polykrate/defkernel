;;;; test-throttle.lisp — Determinism under GPU thermal throttling
;;;;
;;;; Runs a heavy kernel in a tight loop for ~30 seconds.
;;;; As the GPU heats up and throttles (drops clock speed),
;;;; we verify every single result is bit-identical.
;;;;
;;;; If determinism breaks under throttling, it means the hardware
;;;; is reordering operations or dropping precision — a real problem.
;;;;
;;;; Usage: sbcl --load test-throttle.lisp

(load (merge-pathnames "defkernel.lisp" *load-truename*))
(load (merge-pathnames "runtime.lisp" *load-truename*))

(in-package #:defkernel)

;;; Heavy kernel: chained field multiplications (expensive per element)
;;; out = ((a * b) * (a + b)) * ((a * a) + (b * b))

(defkernel *heavy-kernel*
    ((a :vec) (b :vec) (out :out))
  (f* (f* (f* a b) (f+ a b))
      (f+ (f* a a) (f* b b))))

;;; CPU reference

(defconstant +P+ (+ (- (expt 2 64) (expt 2 32)) 1))
(defun fp-op (a b) (mod (* (* (* a b) (mod (+ a b) +P+))
                           (mod (+ (* a a) (* b b)) +P+)) +P+))

;;; GPU frequency monitoring (Intel i915)

(defun read-gpu-freq ()
  "Read current GPU frequency in MHz, or NIL if unavailable."
  (handler-case
      (with-open-file (f "/sys/class/drm/card1/gt_cur_freq_mhz"
                         :direction :input :if-does-not-exist nil)
        (when f (parse-integer (read-line f) :junk-allowed t)))
    (error () nil)))

(defun read-gpu-temp ()
  "Read GPU temperature if available via hwmon."
  (handler-case
      (let ((paths '("/sys/class/drm/card1/device/hwmon/hwmon4/temp1_input"
                     "/sys/class/drm/card1/device/hwmon/hwmon3/temp1_input"
                     "/sys/class/drm/card1/device/hwmon/hwmon2/temp1_input"
                     "/sys/class/drm/card1/device/hwmon/hwmon1/temp1_input")))
        (dolist (p paths)
          (handler-case
              (with-open-file (f p :direction :input :if-does-not-exist nil)
                (when f
                  (let ((val (parse-integer (read-line f) :junk-allowed t)))
                    (when val (return (/ val 1000.0))))))
            (error () nil))))
    (error () nil)))

;;; Main test

(defun run-throttle-test (&key (duration-sec 30) (n 16384))
  (format t "~%══ defkernel: Determinism Under Thermal Throttling ══~%")

  (unless (gpu-available-p)
    (format t "~%No OpenCL device.~%")
    (finish-output)
    (sb-ext:exit :code 1))

  (gpu-init)

  ;; Generate random test data (fixed seed for reproducibility)
  (let* ((state (sb-ext:seed-random-state 42))
         (a (make-array n :element-type '(unsigned-byte 64)))
         (b (make-array n :element-type '(unsigned-byte 64))))
    (dotimes (i n)
      (setf (aref a i) (random +P+ state)
            (aref b i) (random +P+ state)))

    ;; Compute reference result (first GPU run)
    (format t "~%  Elements:  ~D~%" n)
    (format t "  Duration:  ~D seconds~%" duration-sec)
    (format t "  Kernel:    ((a*b)*(a+b)) * (a²+b²) mod P~%")

    (let ((reference (gpu-map *heavy-kernel* :a a :b b)))

      ;; Verify reference matches CPU
      (let ((cpu-ref (map 'vector #'fp-op a b)))
        (format t "  CPU match: ~A~%" (if (equalp reference cpu-ref) "YES" "NO — BUG!"))
        (unless (equalp reference cpu-ref)
          (format t "  ABORTING: GPU doesn't match CPU even before throttling.~%")
          (gpu-cleanup)
          (finish-output)
          (sb-ext:exit :code 1)))

      ;; Now hammer the GPU and check every result
      (format t "~%  Starting throttle stress test...~%")
      (format t "  ~6A  ~6A  ~5A  ~8A  ~A~%"
              "Run" "Freq" "Temp" "Time" "Match")
      (format t "  ~6A  ~6A  ~5A  ~8A  ~A~%"
              "---" "----" "----" "------" "-----")

      (let* ((start-time (get-internal-real-time))
             (end-time (+ start-time (* duration-sec internal-time-units-per-second)))
             (run-count 0)
             (mismatch-count 0)
             (min-freq most-positive-fixnum)
             (max-freq 0)
             (freq-at-start (or (read-gpu-freq) 0)))

        (loop
          (when (>= (get-internal-real-time) end-time) (return))

          (let* ((t0 (get-internal-real-time))
                 (result (gpu-map *heavy-kernel* :a a :b b))
                 (t1 (get-internal-real-time))
                 (dt (- t1 t0))
                 (ms (if (zerop dt) 0.01
                         (/ (* 1000.0 dt) internal-time-units-per-second)))
                 (freq (or (read-gpu-freq) 0))
                 (temp (read-gpu-temp))
                 (match (equalp result reference)))

            (incf run-count)
            (unless match (incf mismatch-count))
            (when (plusp freq)
              (setf min-freq (min min-freq freq))
              (setf max-freq (max max-freq freq)))

            ;; Print every 10th run + first 5 + any mismatch
            (when (or (< run-count 6)
                      (zerop (mod run-count 10))
                      (not match))
              (format t "  ~6D  ~4DMHz ~@[~4,1F°~] ~6,1Fms  ~A~%"
                      run-count freq temp ms
                      (if match "OK" "*** MISMATCH ***")))))

        ;; Summary
        (format t "~%══════════════════════════════════════════════════════════~%")
        (format t "  Runs:       ~D~%" run-count)
        (format t "  Mismatches: ~D~%" mismatch-count)
        (format t "  GPU freq:   ~D → ~D MHz (range: ~D-~D)~%"
                freq-at-start (or (read-gpu-freq) 0) min-freq max-freq)
        (let ((temp (read-gpu-temp)))
          (when temp (format t "  GPU temp:   ~,1F°C~%" temp)))
        (format t "  Elements:   ~D per run × ~D runs = ~,1F M field ops~%"
                n run-count (* n run-count 7.0 1e-6))

        (if (zerop mismatch-count)
            (format t "~%  *** DETERMINISTIC UNDER THROTTLING ***~%")
            (format t "~%  *** DETERMINISM BROKEN: ~D MISMATCHES ***~%" mismatch-count))
        (format t "══════════════════════════════════════════════════════════~%")

        (gpu-cleanup)
        (zerop mismatch-count)))))

(let ((ok (run-throttle-test)))
  (finish-output)
  (sb-ext:exit :code (if ok 0 1)))
