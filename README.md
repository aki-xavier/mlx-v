# mlx-v

V bindings for the [MLX](https://github.com/ml-explore/mlx) C API
([mlx-c](https://ml-explore.github.io/mlx-c/build/html/index.html)), so you can
drive Apple's MLX array library — including the **Metal GPU backend** — directly
from V.

```v
import mlx

fn main() {
  a := mlx.array_f32([f32(1), 2, 3, 4], [2, 2])
  b := mlx.array_f32([f32(10), 20, 30, 40], [2, 2])
  defer {
    a.free()
    b.free()
  }
  c := a + b           // runs on the GPU when available
  println(c.data_f32()) // [11.0, 22.0, 33.0, 44.0]
}
```

## Features

- **Full raw C API surface** — 593 `mlx_*` functions are declared in
  [`cdefs.v`](cdefs.v) (auto-generated from the installed mlx-c headers), so
  anything mlx-c can do is reachable.
- **Ergonomic V wrappers** — an `Array` type with operator overloading
  (`+ - * / %`) plus methods for elementwise math, reductions, shape
  manipulation, matmul, softmax, sorting, indexing and more.
- **Autograd** — `mlx.value_and_grad`, `mlx.jvp`, `mlx.vjp`, `mlx.compile` and
  `mlx.checkpoint` work from plain V `fn`s, so you can train models on the GPU.
- **Device / Stream / memory** helpers, RNG, linear algebra, FFT, fast fused
  kernels (layer-norm, RMS-norm, scaled-dot-product attention), custom Metal
  kernels, and `.npy` / `.safetensors` / `.gguf` I/O.
- **All dtypes read back** — float32/64, int, bool, and also `complex64`,
  `float16` and `bfloat16` accessors.
- **Automatic memory management** — MLX handles are attached to V's Boehm GC and
  reclaimed once the last `Array` referencing them goes away, so `free()` is
  optional.  `mlx.gc_collect()` forces a collection cycle and
  `mlx.live_arrays()` reports the number of live handles.
  **Requires V's Boehm collector (`v -gc boehm`, the default mode)** — the
  conservative scan must be able to see the `Array` boxes inside V's heap
  allocations, which only happens when V's own heap is Boehm-managed.
- **Proper error handling** — MLX errors become V panics with the original
  message (instead of MLX's default `abort()`).

## Requirements

- macOS with Apple Silicon (MLX's Metal backend), or Linux with CUDA.
- [mlx-c](https://github.com/ml-explore/mlx) installed (Homebrew:
  `brew install mlx-c` — this pulls in `mlx` too). Headers are expected under
  `/opt/homebrew/include/mlx/c` and the library under
  `/opt/homebrew/opt/mlx-c/lib`.
- [V](https://vlang.io) 0.5.x (`brew install vlang`).

The module searches `/opt/homebrew/include` and `/opt/homebrew/opt/mlx-c/lib`
by default (Homebrew on Apple Silicon). For Intel Homebrew (`/usr/local`),
Linux/CUDA, or a custom build, set the `MLX_INCLUDE_DIR` and `MLX_LIB_DIR`
environment variables before compiling — e.g.

```sh
export MLX_INCLUDE_DIR=/usr/local/include
export MLX_LIB_DIR=/usr/local/opt/mlx-c/lib
```

The binding generator (`gen/gen_cdefs.py`) honours the same `MLX_INCLUDE_DIR`
variable.

## Install

The module is a standard V module (`v.mod` declares `name: 'mlx'`). Put it on
your `VMODULES` path so `import mlx` resolves. For local development, symlink
it into `~/.vmodules`:

```sh
ln -s "$PWD" ~/.vmodules/mlx
```

(Once published, the usual `v install --git <repo-url>` also works.)

Then compile/run any example:

```sh
v run examples/hello.v
v run examples/demo.v
v run examples/matmul.v
v run examples/training.v    # autograd: gradient descent on the GPU
```

Run the test suite:

```sh
v test .
```

> **Note on the GC:** the automatic handle reclamation relies on V's Boehm
> collector, so consumers must compile with `-gc boehm` (which is V's default
> mode — no flag is normally needed).  V's bundled bdw-gc supplies the
> `GC_MALLOC` / `GC_register_finalizer` symbols used by `gc.v`, so the module
> does not link a separate `-lgc`.

## API overview

```v
// Construction
a := mlx.array_f32([f32(1), 2, 3, 4], [2, 2])   // and array_f64 / _i32 / _i64 / _bool / _u32 / _u64
b := mlx.zeros([2, 2], .float32)
c := mlx.ones([3], .float32)
d := mlx.arange(0, 10, 2, .float32)
e := mlx.full_value([2, 2], 3.14, .float32)
s := mlx.f32_scalar(2.5)
k := mlx.random_key(42)
n := mlx.random_normal([4, 4], .float32, 0.0, 0.02, k)

// Elementwise + operators
x := a + b            // .add(b)
x2 := a * b           // .multiply(b)
x3 := a.sqrt().exp().log()

// Reductions
t := a.sum()          // .sum_axis(0, false) / .sum_axes([0,1], false)
m := a.mean()         // .mean_axis / .mean_axes ; also .max() .min() .prod()
sm := a.softmax(false)

// Linear algebra & shapes
mm := a.matmul(b)     // matrix multiply
r := a.reshape([4])
tr := a.transpose()   // also .swapaxes / .expand_dims / .squeeze / .flatten
q, r2 := a.qr()       // .svd / .lu / .eig / .inv / .solve / .cholesky ...

// Reading results back
vals := mm.data_f32()          // []f32 (row-major copy)
scalar := t.item_f32()         // for 0-d arrays
println(a)                     // calls .str()

// Devices / streams
println(mlx.gpu_available())   // Metal/CUDA
mlx.use_cpu()                  // force CPU (some linalg ops are CPU-only)
mlx.use_gpu()

// Autograd (f must be a top-level fn, not a capturing closure)
fn loss(xs []mlx.Array) []mlx.Array {
  return [xs[0].square().sum()]
}
vag := mlx.value_and_grad(loss, [0])
values, grads := vag.apply([a])   // grads[0] == 2*a
jvps := mlx.jvp(loss, [a], [b])   // also mlx.vjp / mlx.compile / mlx.checkpoint

// Other dtypes read back into V
f16 := a.astype(.float16).data_f16()          // []f32
bf16 := a.astype(.bfloat16).data_bf16()       // []f32
cx := a.rfft(4, 0, .backward).data_complex64() // []mlx.Complex64

// Memory: arrays are garbage-collected; free() is optional
mlx.gc_collect()               // force a collection cycle
println(mlx.live_arrays())     // number of live handles
```

All MLX handles (`Array`, `Device`, `Stream`, `Closure`, maps, GGUF, kernels,
vectors) are attached to V's Boehm GC and reclaimed automatically once the last
copy of a wrapper goes away, so `free()` is optional (call it for deterministic
release in hot loops).  Copies of a wrapper share the same handle.

## Layout

| File | Purpose |
|------|---------|
| `mlx.v` | module entry, flags, enums, error handling |
| `cdefs.v` | raw `C.mlx_*` bindings (generated by `gen/gen_cdefs.py`) |
| `array.v` | `Array` type, constructors, data access |
| `gc.v` | Boehm GC wiring (`gc_collect`, `live_arrays`) |
| `handle.v` | generic GC-managed shared box for non-array handles |
| `ops.v` | elementwise / reductions / shape / matmul ops + operators |
| `closure.v` | `Func`/`Closure`, the V-function → MLX-closure bridge |
| `transforms.v` | `value_and_grad` / `jvp` / `vjp` / `compile` / `checkpoint` |
| `device.v`, `stream.v`, `vector.v`, `map.v` | supporting types |
| `random.v`, `linalg.v`, `fft.v`, `fast.v`, `io.v`, `memory.v`, `compile.v` | domain wrappers |
| `mlx.c`, `mlx_v.h` | tiny C helpers (thread-local error buffer, CPU flag, stream cache + override, half→float, GC) |

## Regenerating the raw bindings

`cdefs.v` is generated from the installed mlx-c headers:

```sh
python3 gen/gen_cdefs.py
```

## Notes

- **Autograd functions must be top-level `fn`s** — they are passed to MLX as C
  function pointers, so V closures that capture variables are not supported
  there.
- `custom_vjp` / `custom_function` (custom backward rules) are bound at the raw
  `C.` level but do not yet have ergonomic V wrappers.
- float16/bfloat16 arrays are read back as `f32` (via the bit-exact
  `mlx_v_f16_to_f32` / `mlx_v_bf16_to_f32` helpers); writing raw half data from
  V is not exposed (use `astype(.float16)` from a float32 array instead).
- **`free()` is idempotent** on every handle type (`Array`, `Device`, `Stream`,
  `Closure`, maps, GGUF, kernels, vectors); calling it twice is a no-op.  All of
  them are also GC-managed (see above), so forgetting `free()` does not leak.
- **Thread safety** — the error buffer is thread-local and the internal
  counters/stream cache are synchronised, so concurrent ops on different
  threads get their own error messages.  However, MLX's error handler, default
  device/stream and `use_cpu()`/`use_gpu()` are process-wide, so those
  controls affect every thread.
- **Panics in autograd functions** — an MLX error inside a `fn` passed to
  `value_and_grad`/`jvp`/`vjp`/`compile`/`checkpoint` panics, and that panic
  escapes MLX's C callback boundary (no C++ stack unwinding), aborting the
  process.  Keep those functions free of failing ops.
