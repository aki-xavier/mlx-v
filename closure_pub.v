module mlx

// closure_pub.v — public payload-closure primitives for consumers that must
// attach a pointer to a library class to a compiled closure (the mlx Func
// type cannot capture values; the payload slot is the escape hatch).

// FuncPayload is the raw callback signature with a payload slot.
pub type FuncPayload = fn (vres &C.mlx_vector_array, input C.mlx_vector_array, payload voidptr) int

// new_closure_payload builds a closure that calls `fun` with `payload`.
pub fn new_closure_payload(fun FuncPayload, payload voidptr) Closure {
	return Closure{
		box:    wrap_handle(C.mlx_closure_new_func_payload(ClosureFuncPayload(fun), payload, 0).ctx,
			free_closure_handle, true)
	}
}

// wrap_closure_pub wraps an owned raw closure handle created outside (e.g. by
// mlx_compile) so callers can use the standard Closure API.
pub fn wrap_closure_pub(ctx C.mlx_closure) Closure {
	return Closure{
		box: wrap_handle(ctx, free_closure_handle, true)
	}
}

// compile_closure compiles an existing closure (payload preserved).
pub fn (c Closure) compile_closure(shapeless bool) Closure {
	setup()
	begin_op()
	res := C.mlx_closure_new()
	rc := C.mlx_compile(&res, c.raw(), shapeless)
	if rc != 0 {
		C.mlx_closure_free(res)
		check(rc)
	}
	return wrap_closure_pub(res)
}

// vector_to_arrays_pub mirrors vector_to_arrays() for consumer trampolines.
pub fn vector_to_arrays_pub(input C.mlx_vector_array) []Array {
	return vector_to_arrays(input)
}

// arrays_to_vector_pub mirrors arrays_to_vector() for consumer trampolines.
pub fn arrays_to_vector_pub(xs []Array) C.mlx_vector_array {
	return arrays_to_vector(xs)
}
