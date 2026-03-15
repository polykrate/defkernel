# defkernel

**Deterministic GPU kernels from Lisp — with a GPU-accelerated STARK prover.**

Common Lisp macros rewrite the AST at compile time, producing OpenCL C
with a fixed evaluation order. No floats, no thread-dependent reductions
— just exact integer field arithmetic (mod Goldilocks prime P = 2^64 - 2^32 + 1).

Includes a complete **NTT → FRI → STARK** pipeline that runs on any OpenCL device.

## Why

GPUs are non-deterministic for three reasons:
1. Thread execution order varies between runs
2. Floating-point addition is not associative: `(a+b)+c ≠ a+(b+c)`
3. Reduction order depends on hardware scheduling

`defkernel` kills all three:
- **No floats** — integer field arithmetic mod P is exact
- **SSA form** — every sub-expression gets a unique temporary, fixing eval order
- **Balanced binary reduction trees** — `(reduce-sum v 8)` compiles to a fixed tree shape, not a runtime loop

The Lisp code never runs on the GPU. It's a **compiler** that generates
deterministic OpenCL C at macroexpansion time.

## Architecture

```
Lisp AST → SSA IR → OpenCL C → GPU
                                 ↓
                    NTT → FRI → STARK proof
```

## Quick start

```bash
# Compiler tests (no GPU needed)
sbcl --load tests.lisp

# NTT correctness tests (GPU)
sbcl --load test-ntt.lisp

# Full STARK prover tests (GPU)
sbcl --load test-stark.lisp

# Benchmarks
sbcl --load bench.lisp

# GPU demo
sbcl --load demo.lisp
```

Or load as an ASDF system:

```lisp
(asdf:load-system "defkernel")
(defkernel:gpu-init)
(defkernel:stark-test :trace-size 1024 :gpu t)
(defkernel:gpu-cleanup)
```

## Kernel API

```lisp
(defkernel *my-kernel* ((a :vec) (b :vec) (out :out))
  (f+ (f* a b) (f* a b)))
```

Compiles at macroexpansion time to OpenCL C with deterministic evaluation order.

### Expressions

| Form | Description |
|------|-------------|
| `(f+ a b)` | Field addition mod P |
| `(f* a b)` | Field multiplication mod P |
| `(f- a b)` | Field subtraction mod P |
| `(+ a b)` | Wrapping u64 addition |
| `(* a b)` | Wrapping u64 multiplication |
| `(idiv a b)` | Integer division |
| `(imod a b)` | Integer modulo |
| `(cmp< a b)` | Comparison (returns 0 or 1) |
| `(select c a b)` | Conditional: `c ? a : b` |
| `gid` | Work-item index (`get_global_id(0)`) |
| `(aref v i)` | Element access |
| `(reduce-sum v n)` | Deterministic sum (balanced binary tree) |
| `(logand a b)` | Bitwise AND |
| `(logxor a b)` | Bitwise XOR |

### Runtime

```lisp
(gpu-init)                                ; detect GPU, create context
(gpu-map *kernel* :a #(1 2 3) :b #(4 5 6)) ; single-pass execution
(gpu-alloc n)                             ; allocate GPU buffer
(gpu-upload buf data)                     ; host → GPU
(gpu-dispatch kernel :global-size n       ; multi-pass (no download)
             :buffers '((a . buf-a) (out . buf-b))
             :scalars '((stride . 4)))
(gpu-download buf)                        ; GPU → host
(gpu-free buf)                            ; release buffer
(gpu-cleanup)                             ; release all resources
```

## NTT (Number Theoretic Transform)

GPU-accelerated NTT over the Goldilocks field via a multi-pass butterfly kernel.

```lisp
(ntt-forward data :gpu t)     ; coefficient → evaluation form
(ntt-inverse evals :gpu t)    ; evaluation → coefficient form
(poly-mul-ntt a b :gpu t)     ; polynomial multiplication via NTT
```

### NTT Benchmarks (Intel UHD iGPU)

| N | CPU (ms) | GPU (ms) | Speedup |
|---|----------|----------|---------|
| 2^10 | 2.3 | 1.2 | 1.9x |
| 2^14 | 48.8 | 10.2 | 4.8x |
| 2^16 | 211.3 | 40.4 | 5.2x |
| 2^20 | — | 691.1 | — |

## STARK Prover

Complete STARK proof system: trace → INTT → LDE → constraints → FRI → proof.

```lisp
(let* ((trace (generate-fib-trace 1024))
       (proof (stark-prove trace :gpu t)))
  (stark-verify proof))  ; → T
```

### STARK Benchmarks

| Trace N | CPU Prove | GPU Prove | Speedup |
|---------|-----------|-----------|---------|
| 256 | 3.6 ms | 3.5 ms | 1.0x |
| 1024 | 15.0 ms | 12.0 ms | 1.3x |
| 4096 | 75.0 ms | 31.0 ms | 2.4x |

### Prover breakdown (N=1024, GPU)

| Step | Time |
|------|------|
| Inverse NTT (trace → coeffs) | 1.3 ms |
| LDE (NTT, blowup=4) | 2.5 ms |
| Constraint evaluation | 0.8 ms |
| Merkle commit | 0.7 ms |
| FRI commit (10 rounds) | 2.9 ms |
| **Total** | **8.2 ms** |

NTT-dominated: 82% of total prover time.

## Files

| File | Purpose |
|------|---------|
| `defkernel.lisp` | Compiler: AST → SSA → OpenCL C |
| `runtime.lisp` | OpenCL CFFI bindings, GPU buffer management |
| `ntt.lisp` | NTT: twiddles, butterfly kernel, forward/inverse |
| `fri.lisp` | FRI: fold kernel, Merkle commitment, query/verify |
| `stark.lisp` | STARK prover + verifier (Fibonacci AIR) |
| `bench.lisp` | NTT + STARK benchmarks |
| `tests.lisp` | Compiler unit tests |
| `test-ntt.lisp` | NTT correctness tests |
| `test-stark.lisp` | STARK prover tests |
| `test-determinism.lisp` | GPU vs CPU edge-case tests |
| `test-throttle.lisp` | Determinism under thermal throttling |
| `demo.lisp` | GPU demo with CPU reference comparison |
| `defkernel.asd` | ASDF system definition |

## Requirements

- **SBCL** (tested with 2.6.1)
- **Quicklisp** (for CFFI)
- **OpenCL ICD driver**:
  - Intel integrated GPU: `intel-compute-runtime`
  - CPU fallback: `pocl`
  - NVIDIA: proprietary driver
  - AMD: `rocm-opencl-runtime`

## Field arithmetic

Uses the Goldilocks prime `P = 2^64 - 2^32 + 1`. Multiplication uses
`mul_hi()` (OpenCL built-in for upper 64 bits of 64×64 multiply) instead
of `__int128`, which isn't supported by all GPU OpenCL compilers.

## License

Part of the [JOTL](https://github.com/music-karma/jotl) project (JAM On The Lisp).
