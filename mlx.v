module mlx

// mlx.v — V bindings for the MLX C API (mlx-c).
//
// This module lets V programs drive Apple's MLX array library (CPU + GPU/Metal)
// through the official C API.  All raw `C.mlx_*` symbols are declared in
// `cdefs.v`; this file adds the V-flavoured types, error handling and small
// helper functions.
//
// All mutable state (error buffer, force-CPU flag, stream cache) lives in the
// tiny C helpers in `mlx.c`/`mlx_v.h`, so no V module-level globals are needed
// and consumers do not need `-enable-globals`.

// --- C toolchain wiring -----------------------------------------------------

// The GC-backed `Array` (see gc.v) relies on V's Boehm collector: V's own heap
// allocations (slices holding `Array` values) must live on the Boehm heap so
// the conservative scan can find the `ArrayBox` pointers stored in them.
// Without `-gc boehm` those pointers live on the malloc heap, Boehm never sees
// them, and the finalizers run too early (use-after-free).  `-gc boehm` is V's
// default mode and provides the bundled bdw-gc + GC_THREADS defines, so no
// extra -lgc / -L flags are needed here.

// mlx-c include/library search paths.  Homebrew (Apple Silicon) is the
// default.  For Intel Homebrew, Linux/CUDA, or a custom build, set
// `MLX_INCLUDE_DIR` and `MLX_LIB_DIR` (the extra -I/-L flags are harmless when
// unset).
#flag -I/opt/homebrew/include
#flag -L/opt/homebrew/opt/mlx-c/lib
#flag -I$MLX_INCLUDE_DIR
#flag -L$MLX_LIB_DIR
#flag -lmlxc
#include "mlx/c/mlx.h"

// Tiny C helpers (thread-local error buffer, force-CPU flag, stream cache, GC).
#include "@VMODROOT/mlx_v.h"
#flag @VMODROOT/mlx.c

// --- enums (values must match the C `mlx_*` enums exactly) -------------------

// Dtype enumerates the MLX array element types.
pub enum Dtype {
	bool_
	uint8
	uint16
	uint32
	uint64
	int8
	int16
	int32
	int64
	float16
	float32
	float64
	bfloat16
	complex64
}

// DeviceType enumerates the MLX device backends.
pub enum DeviceType {
	cpu
	gpu
}

// CompileMode selects how `mlx.compile` behaves.
pub enum CompileMode {
	disabled
	no_simplify
	no_fuse
	enabled
}

// FftNorm selects the FFT normalisation convention.
pub enum FftNorm {
	backward
	ortho
	forward
}

// Complex64 is a float32 complex number (real + imaginary), layout-compatible
// with MLX's `complex64` dtype.
pub struct Complex64 {
pub:
	real f32
	imag f32
}

// --- error handling ----------------------------------------------------------

// MlxErrorHandlerFunc is the C error-handler callback signature.
type MlxErrorHandlerFunc = fn (msg &char, data voidptr)

// C function declared manually (it takes a function pointer, so the
// auto-generated bindings skip it).
fn C.mlx_set_error_handler(handler MlxErrorHandlerFunc, data voidptr, dtor voidptr)

// Helpers in mlx.c.
fn C.mlx_v_clear_error()
fn C.mlx_v_set_error(msg &char)
fn C.mlx_v_get_error() &char
fn C.mlx_v_set_force_cpu(v int)
fn C.mlx_v_get_force_cpu() int
fn C.mlx_v_f16_to_f32(h u16) f32
fn C.mlx_v_bf16_to_f32(h u16) f32
fn C.mlx_v_note_box_alloc()
fn C.mlx_v_note_box_free()
fn C.mlx_v_get_live_boxes() int
fn C.mlx_v_cached_cpu_stream() C.mlx_stream
fn C.mlx_v_cached_gpu_stream() C.mlx_stream
fn C.mlx_v_set_stream_override(s C.mlx_stream)
fn C.mlx_v_clear_stream_override()
fn C.mlx_v_stream_for_ops() C.mlx_stream
fn C.mlx_v_ensure_error_handler(handler MlxErrorHandlerFunc)

// Accessors for complex64 / float16 / bfloat16 (declared manually: the C
// pointer types use `voidptr` to avoid `_Complex`/`__fp16` ABI type clashes).
fn C.mlx_array_item_complex64(res voidptr, arr C.mlx_array) int
fn C.mlx_array_data_complex64(arr C.mlx_array) voidptr
fn C.mlx_array_item_float16(res voidptr, arr C.mlx_array) int
fn C.mlx_array_data_float16(arr C.mlx_array) voidptr
fn C.mlx_array_item_bfloat16(res voidptr, arr C.mlx_array) int
fn C.mlx_array_data_bfloat16(arr C.mlx_array) voidptr

// c_error_handler records the message passed by the C runtime.  MLX's default
// handler would `abort()` the process; installing this one lets us turn errors
// into ordinary V panics instead.
fn c_error_handler(msg &char, _data voidptr) {
	C.mlx_v_set_error(msg)
}

// setup installs the V error handler.  mlx.c guards the process-global
// mlx_set_error_handler call with pthread_once, so this is cheap to repeat.
@[inline]
fn setup() {
	C.mlx_v_ensure_error_handler(c_error_handler)
}

// begin_op resets the recorded error before an op runs.
@[inline]
fn begin_op() {
	C.mlx_v_clear_error()
}

// check panics with the recorded MLX message when an op returns non-zero.
@[inline]
fn check(rc int) {
	if rc != 0 {
		panic('MLX error: ${cstr(C.mlx_v_get_error())}')
	}
}

// check_res is check() for ops that write into a pre-allocated result handle:
// on error the handle is released before panicking, so the error path does
// not leak it.
@[inline]
fn check_res(rc int, res C.mlx_array) {
	if rc != 0 {
		C.mlx_array_free(res)
		check(rc)
	}
}

// check_res2 is check_res() for two-output ops.
@[inline]
fn check_res2(rc int, r0 C.mlx_array, r1 C.mlx_array) {
	if rc != 0 {
		C.mlx_array_free(r0)
		C.mlx_array_free(r1)
		check(rc)
	}
}

// check_vec is check_res() for vector-of-array results.
@[inline]
fn check_vec(rc int, vec C.mlx_vector_array) {
	if rc != 0 {
		C.mlx_vector_array_free(vec)
		check(rc)
	}
}

// check_vec2 is check_res() for two-vector-output ops.
@[inline]
fn check_vec2(rc int, v0 C.mlx_vector_array, v1 C.mlx_vector_array) {
	if rc != 0 {
		C.mlx_vector_array_free(v0)
		C.mlx_vector_array_free(v1)
		check(rc)
	}
}

// --- C string helpers --------------------------------------------------------

// cstr copies a C string into a V string.
@[inline]
fn cstr(p &char) string {
	return unsafe { cstring_to_vstring(p) }
}

// --- version & device queries ------------------------------------------------

// version returns the MLX library version string.
pub fn version() string {
	s := C.mlx_string_new()
	C.mlx_version(&s)
	res := cstr(C.mlx_string_data(s))
	C.mlx_string_free(s)
	return res
}

// metal_available reports whether the Metal (GPU) backend is available.
pub fn metal_available() bool {
	mut res := false
	C.mlx_metal_is_available(&res)
	return res
}

// cuda_available reports whether the CUDA backend is available.
pub fn cuda_available() bool {
	mut res := false
	C.mlx_cuda_is_available(&res)
	return res
}

// gpu_available reports whether any GPU backend is available.
pub fn gpu_available() bool {
	return metal_available() || cuda_available()
}

// use_cpu makes subsequent ops run on the CPU (some linalg ops are CPU-only).
// This flips process-wide MLX state and is not thread-safe.
pub fn use_cpu() {
	C.mlx_v_set_force_cpu(1)
	C.mlx_v_clear_stream_override()
	d := device(.cpu, 0)
	d.set_default()
	d.free()
}

// use_gpu makes subsequent ops run on the GPU when one is available.
// This flips process-wide MLX state and is not thread-safe.
pub fn use_gpu() {
	C.mlx_v_set_force_cpu(0)
	C.mlx_v_clear_stream_override()
	if gpu_available() {
		d := device(.gpu, 0)
		d.set_default()
		d.free()
	}
}
