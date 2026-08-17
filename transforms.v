module mlx

// transforms.v — autograd transforms and evaluation helpers.

// ValueAndGrad evaluates a function and its gradient w.r.t. selected arguments.
pub struct ValueAndGrad {
mut:
	ctx     C.mlx_closure_value_and_grad
	closure Closure // kept alive while `ctx` may still reference it
	freed   bool
}

// value_and_grad wraps `f`, differentiating with respect to `argnums`.
pub fn value_and_grad(f Func, argnums []int) ValueAndGrad {
	c := new_closure(f)
	ctx := C.mlx_closure_value_and_grad_new()
	setup()
	begin_op()
	check(C.mlx_value_and_grad(&ctx, c.ctx, argnums.data, argnums.len))
	return ValueAndGrad{
		ctx:     ctx
		closure: c
	}
}

// free releases the value-and-grad closure (idempotent).
pub fn (mut vg ValueAndGrad) free() {
	if !vg.freed {
		vg.freed = true
		C.mlx_closure_value_and_grad_free(vg.ctx)
		vg.closure.free()
	}
}

// apply returns (values, grads) for the given inputs.
pub fn (vg ValueAndGrad) apply(inputs []Array) ([]Array, []Array) {
	setup()
	begin_op()
	vin := arrays_to_vector(inputs)
	defer {
		C.mlx_vector_array_free(vin)
	}
	vout := C.mlx_vector_array_new()
	vdout := C.mlx_vector_array_new()
	check(C.mlx_closure_value_and_grad_apply(&vout, &vdout, vg.ctx, vin))
	return array_vector_to_slice(vout), array_vector_to_slice(vdout)
}

// jvp computes the Jacobian-vector product of `f` at `primals` along `tangents`.
// Returns (values, jvps).
pub fn jvp(f Func, primals []Array, tangents []Array) ([]Array, []Array) {
	mut c := new_closure(f)
	defer {
		c.free()
	}
	setup()
	begin_op()
	p := arrays_to_vector(primals)
	t := arrays_to_vector(tangents)
	defer {
		C.mlx_vector_array_free(p)
		C.mlx_vector_array_free(t)
	}
	vout := C.mlx_vector_array_new()
	vdout := C.mlx_vector_array_new()
	check(C.mlx_jvp(&vout, &vdout, c.ctx, p, t))
	return array_vector_to_slice(vout), array_vector_to_slice(vdout)
}

// vjp computes the vector-Jacobian product of `f` at `primals` with
// `cotangents`.  Returns (values, vjps).
pub fn vjp(f Func, primals []Array, cotangents []Array) ([]Array, []Array) {
	mut c := new_closure(f)
	defer {
		c.free()
	}
	setup()
	begin_op()
	p := arrays_to_vector(primals)
	ct := arrays_to_vector(cotangents)
	defer {
		C.mlx_vector_array_free(p)
		C.mlx_vector_array_free(ct)
	}
	vout := C.mlx_vector_array_new()
	vdout := C.mlx_vector_array_new()
	check(C.mlx_vjp(&vout, &vdout, c.ctx, p, ct))
	return array_vector_to_slice(vout), array_vector_to_slice(vdout)
}

// compile traces and compiles `f`, returning a faster closure.
// `mlx_compile` copies the source closure into the result, so freeing `c`
// below is safe (unlike value_and_grad, which retains it defensively).
pub fn compile(f Func, shapeless bool) Closure {
	mut c := new_closure(f)
	res := C.mlx_closure_new()
	setup()
	begin_op()
	check(C.mlx_compile(&res, c.ctx, shapeless))
	c.free()
	return Closure{
		ctx: res
	}
}

// checkpoint wraps `f` with gradient checkpointing (trades memory for compute).
// `mlx_checkpoint` copies the source closure, so freeing `c` is safe.
pub fn checkpoint(f Func) Closure {
	mut c := new_closure(f)
	res := C.mlx_closure_new()
	setup()
	begin_op()
	check(C.mlx_checkpoint(&res, c.ctx))
	c.free()
	return Closure{
		ctx: res
	}
}

// eval forces evaluation of all `outputs`.
pub fn eval(outputs []Array) {
	setup()
	begin_op()
	mut vec := array_vector(outputs)
	defer {
		vec.free()
	}
	check(C.mlx_eval(vec.ctx))
}

// async_eval schedules evaluation of `outputs` without blocking.
pub fn async_eval(outputs []Array) {
	setup()
	begin_op()
	mut vec := array_vector(outputs)
	defer {
		vec.free()
	}
	check(C.mlx_async_eval(vec.ctx))
}
