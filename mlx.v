module mlx

// mlx.v — V bindings for the MLX C API (mlx-c).
//
// This module lets V programs drive Apple's MLX array library (CPU + GPU/Metal)
// through the official C API.  All raw `C.mlx_*` symbols are declared in
// `cdefs.v`; this file adds the V-flavoured types, error handling and small
// helper functions.
//
// Build note: the module needs mutable module-level state for the MLX error
// message, so compile consumers with `v -enable-globals ...`.

// --- C toolchain wiring -----------------------------------------------------

// Homebrew keeps bdw-gc (Boehm GC, which V itself links) keg-only; give the
// linker the path so any V program linking this module resolves `-lgc`.
#flag darwin -L/opt/homebrew/opt/bdw-gc/lib

#flag -I/opt/homebrew/include
#flag -L/opt/homebrew/opt/mlx-c/lib
#flag -lmlxc
#include "mlx/c/mlx.h"

// Tiny C helpers (error buffer + cpu flag).
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

// setup installs the V error handler.  Setting it repeatedly is cheap and
// idempotent, so every op can just call it.
@[inline]
fn setup() {
	C.mlx_set_error_handler(c_error_handler, 0, 0)
}

// begin_op resets the recorded error before an op runs.
@[inline]
fn begin_op() {
	C.mlx_set_error_handler(c_error_handler, 0, 0)
	C.mlx_v_clear_error()
}

// check panics with the recorded MLX message when an op returns non-zero.
@[inline]
fn check(rc int) {
	if rc != 0 {
		panic('MLX error: ${cstr(C.mlx_v_get_error())}')
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
pub fn use_cpu() {
	C.mlx_v_set_force_cpu(1)
	d := device(.cpu, 0)
	d.set_default()
	d.free()
}

// use_gpu makes subsequent ops run on the GPU when one is available.
pub fn use_gpu() {
	C.mlx_v_set_force_cpu(0)
	if gpu_available() {
		d := device(.gpu, 0)
		d.set_default()
		d.free()
	}
}
