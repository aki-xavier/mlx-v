module mlx

// closure.v — MLX closures built from V functions, the basis for autograd
// transforms (value_and_grad / jvp / vjp / compile / checkpoint).

// Func is a differentiable function from a slice of arrays to a slice of
// arrays.  It must be a top-level `fn` (not a V closure that captures
// variables), because it is passed to MLX as a plain C function pointer.
pub type Func = fn (xs []Array) []Array

// ClosureFuncPayload is the C callback signature MLX invokes.
type ClosureFuncPayload = fn (vres &C.mlx_vector_array, input C.mlx_vector_array, payload voidptr) int

// C function declared manually (takes a function pointer).
fn C.mlx_closure_new_func_payload(fun ClosureFuncPayload, payload voidptr, dtor voidptr) C.mlx_closure

// Closure wraps an MLX closure (a callable).
pub struct Closure {
mut:
	ctx   C.mlx_closure
	freed bool
}

// new_closure builds a closure from `f`.
pub fn new_closure(f Func) Closure {
	setup()
	return Closure{
		ctx: C.mlx_closure_new_func_payload(closure_thunk, voidptr(f), 0)
	}
}

// free releases the closure (idempotent).
pub fn (mut c Closure) free() {
	if !c.freed {
		c.freed = true
		C.mlx_closure_free(c.ctx)
	}
}

// apply runs the closure on `inputs` and returns its outputs.
pub fn (c Closure) apply(inputs []Array) []Array {
	setup()
	begin_op()
	vin := arrays_to_vector(inputs)
	defer {
		C.mlx_vector_array_free(vin)
	}
	out := C.mlx_vector_array_new()
	check(C.mlx_closure_apply(&out, c.ctx, vin))
	return array_vector_to_slice(out)
}

// closure_thunk bridges MLX's vector-of-arrays ABI to the V `Func` type.
//
// NOTE: `f` is user V code and may `panic` (MLX errors become panics).  That
// panic unwinds out of this callback and through MLX's C++ frames without C++
// stack unwinding, so an error inside an autograd function aborts the process
// rather than returning a recoverable error.  Keep `f` panic-free in practice.
fn closure_thunk(vres &C.mlx_vector_array, input C.mlx_vector_array, payload voidptr) int {
	f := Func(payload)
	xs := vector_to_arrays(input)
	ys := f(xs)
	for x in xs {
		x.free()
	}
	out := C.mlx_vector_array_new()
	for y in ys {
		C.mlx_vector_array_append_value(out, y.raw())
		y.free()
	}
	C.mlx_vector_array_set(vres, out)
	C.mlx_vector_array_free(out)
	return 0
}

// vector_to_arrays copies the arrays out of a raw vector (caller owns the copies).
fn vector_to_arrays(input C.mlx_vector_array) []Array {
	n := int(C.mlx_vector_array_size(input))
	mut xs := []Array{cap: n}
	for i in 0 .. n {
		a := C.mlx_array_new()
		C.mlx_vector_array_get(&a, input, usize(i))
		xs << wrap_array(a)
	}
	return xs
}

// arrays_to_vector builds a raw vector from a V slice of arrays.
fn arrays_to_vector(xs []Array) C.mlx_vector_array {
	v := C.mlx_vector_array_new()
	for x in xs {
		C.mlx_vector_array_append_value(v, x.raw())
	}
	return v
}
