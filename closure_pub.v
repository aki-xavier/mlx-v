module mlx

// closure_pub.v — public payload-closure primitives for consumers that must
// attach a pointer to a library class to a compiled closure (the mlx Func
// type cannot capture values; the payload slot is the escape hatch).
//
// PayloadPair carries both the consumer dispatcher (a plain V fn taking
// arrays + a voidptr) and the instance pointer, so consumer modules never
// touch the C vector types.

// PayloadPair is the user-supplied callback plus its instance pointer.
pub struct PayloadPair {
pub:
	f    fn (xs []Array, data voidptr) []Array
	data voidptr
}

// payload_thunk converts vectors to arrays, calls the consumer fn and writes
// the result vector back.
fn payload_thunk(vres &C.mlx_vector_array, input C.mlx_vector_array, payload voidptr) int {
	p := unsafe { &PayloadPair(payload) }
	xs := vector_to_arrays(input)
	defer {
		for x in xs {
			x.free()
		}
	}
	ys := p.f(xs, p.data)
	out := C.mlx_vector_array_new()
	for y in ys {
		C.mlx_vector_array_append_value(out, y.raw())
		y.free()
	}
	C.mlx_vector_array_set(vres, out)
	C.mlx_vector_array_free(out)
	return 0
}

// new_payload_closure builds a closure bound to `pair`.  The caller must keep
// the pair alive as long as the closure (the closure only stores the pointer).
pub fn new_payload_closure(pair &PayloadPair) Closure {
	return Closure{
		box: wrap_handle(C.mlx_closure_new_func_payload(ClosureFuncPayload(payload_thunk),
			voidptr(pair), 0).ctx, free_closure_handle, true)
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
