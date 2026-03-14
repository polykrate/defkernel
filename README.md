# defkernel

**Deterministic GPU kernels from Lisp arithmetic expressions.**

Common Lisp macros rewrite the AST at compile time, producing OpenCL C
with a fixed evaluation order. No floats, no thread-dependent reductions
— just exact integer field arithmetic (mod Goldilocks prime P = 2^64 - 2^32 + 1).

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

## How it works

```
Lisp AST → SSA IR → OpenCL C
```

```lisp
(defkernel *dot-product*
    ((a :vec) (b :vec) (out :out))
  (f+ (f* a b) (f* a b)))
```

Expands at compile time to a `kernel` struct containing this OpenCL C:

```c
#define GP 0xFFFFFFFF00000001UL
#define EPSILON 0xFFFFFFFFUL

inline ulong fp_add(ulong a, ulong b) { /* ... */ }
inline ulong fp_mul(ulong a, ulong b) { /* ... */ }

__kernel void dot_product(__global const ulong* a,
                          __global const ulong* b,
                          __global ulong* out) {
  size_t gid = get_global_id(0);
  ulong t0 = a[gid];
  ulong t1 = b[gid];
  ulong t2 = fp_mul(t0, t1);
  ulong t3 = a[gid];
  ulong t4 = b[gid];
  ulong t5 = fp_mul(t3, t4);
  ulong t6 = fp_add(t2, t5);
  out[gid] = t6;
}
```

Every temporary is numbered. The evaluation order is the same on every
GPU, every run, every driver version.

## API

### Kernel definition

```lisp
(defkernel name ((param type) ...) body)
```

Parameter types: `:vec` (input array), `:scalar` (single value), `:out` (output array).

Body expressions:
| Form | Description |
|------|-------------|
| `(f+ a b)` | Field addition mod P |
| `(f* a b)` | Field multiplication mod P |
| `(f- a b)` | Field subtraction mod P |
| `(+ a b)` | Wrapping u64 addition |
| `(* a b)` | Wrapping u64 multiplication |
| `(reduce-sum v n)` | Deterministic sum of n elements (balanced binary tree) |
| `(aref v i)` | Element access |
| `(logand a b)` | Bitwise AND |
| `(logxor a b)` | Bitwise XOR |

### Runtime (OpenCL)

```lisp
(load "defkernel.lisp")
(load "runtime.lisp")

(defkernel:gpu-init)                          ; detect GPU, create context
(defkernel:gpu-map *my-kernel* :a #(1 2 3)    ; run kernel
                               :b #(4 5 6))
(defkernel:gpu-cleanup)                       ; release resources
```

## Files

| File | Purpose |
|------|---------|
| `defkernel.lisp` | Compiler: AST → SSA → OpenCL C |
| `runtime.lisp` | OpenCL CFFI bindings, GPU init/execute/cleanup |
| `demo.lisp` | Run kernels on real GPU, verify against CPU reference |
| `tests.lisp` | Compiler unit tests |
| `test-determinism.lisp` | 30 edge-case tests: GPU vs CPU on overflow/underflow/boundary values |
| `test-throttle.lisp` | Determinism under thermal throttling (sustained GPU load) |

## Requirements

- SBCL
- Quicklisp (for CFFI)
- OpenCL ICD driver:
  - Intel integrated GPU: `intel-compute-runtime` (or `intel-compute-runtime-legacy-bin` for pre-Gen11)
  - CPU fallback: `pocl`

## Run

```bash
# Compiler tests (no GPU needed)
sbcl --load tests.lisp

# GPU demo
sbcl --load demo.lisp

# Determinism edge cases (GPU required)
sbcl --load test-determinism.lisp

# Throttling stress test (GPU required, runs 30s)
sbcl --load test-throttle.lisp
```

## Field arithmetic

Uses the Goldilocks prime `P = 2^64 - 2^32 + 1`. Multiplication uses
`mul_hi()` (OpenCL built-in for upper 64 bits of 64x64 multiply) instead
of `__int128`, which isn't supported by all GPU OpenCL compilers.

## License

Part of the [JOTL](https://github.com/music-karma/jotl) project (JAM On The Lisp).
