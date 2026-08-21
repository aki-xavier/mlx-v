module mlx

// transforms.v — autograd transforms and evaluation helpers.

// ValueAndGrad evaluates a function and its gradient w.r.t. selected arguments.
// The handle is GC-managed; `free()` releases it deterministically.
pub struct ValueAndGrad {
mut:
	box     &HandleBox = unsafe { nil }
	closure Closure // kept alive while the boxed handle may still reference it
}

// raw returns the underlying MLX handle (low level).
@[inline]
pub fn (vg ValueAndGrad) raw() C.mlx_closure_value_and_grad {
	if isnil(vg.box) {
		panic('mlx: ValueAndGrad is uninitialised (zero value); build it with value_and_grad() before using it')
	}
	return C.mlx_closure_value_and_grad{
		ctx: vg.box.ctx
	}
}

// value_and_grad wraps `f`, differentiating with respect to `argnums`.
pub fn value_and_grad(f Func, argnums []int) ValueAndGrad {
	c := new_closure(f)
	ctx := C.mlx_closure_value_and_grad_new()
	setup()
	begin_op()
	rc := C.mlx_value_and_grad(&ctx, c.raw(), argnums.data, argnums.len)
	if rc != 0 {
		C.mlx_closure_value_and_grad_free(ctx)
		c.free()
		check(rc)
	}
	return ValueAndGrad{
		box:     wrap_handle(ctx.ctx, free_closure_value_and_grad_handle, true)
		closure: c
	}
}

// free releases the value-and-grad closure (idempotent; optional with the GC).
pub fn (vg &ValueAndGrad) free() {
	if !isnil(vg.box) {
		mut box := vg.box
		box.release()
	}
	vg.closure.free()
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
	check_vec2(C.mlx_closure_value_and_grad_apply(&vout, &vdout, vg.raw(), vin), vout, vdout)
	return array_vector_to_slice(vout), array_vector_to_slice(vdout)
}

// jvp computes the Jacobian-vector product of `f` at `primals` along `tangents`.
// Returns (values, jvps).
pub fn jvp(f Func, primals []Array, tangents []Array) ([]Array, []Array) {
	c := new_closure(f)
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
	check_vec2(C.mlx_jvp(&vout, &vdout, c.raw(), p, t), vout, vdout)
	return array_vector_to_slice(vout), array_vector_to_slice(vdout)
}

// vjp computes the vector-Jacobian product of `f` at `primals` with
// `cotangents`.  Returns (values, vjps).
pub fn vjp(f Func, primals []Array, cotangents []Array) ([]Array, []Array) {
	c := new_closure(f)
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
	check_vec2(C.mlx_vjp(&vout, &vdout, c.raw(), p, ct), vout, vdout)
	return array_vector_to_slice(vout), array_vector_to_slice(vdout)
}

// compile traces and compiles `f`, returning a faster closure.
// `mlx_compile` copies the source closure into the result, so freeing `c`
// below is safe (unlike value_and_grad, which retains it defensively).
pub fn compile(f Func, shapeless bool) Closure {
	c := new_closure(f)
	res := C.mlx_closure_new()
	setup()
	begin_op()
	rc := C.mlx_compile(&res, c.raw(), shapeless)
	if rc != 0 {
		C.mlx_closure_free(res)
		c.free()
		check(rc)
	}
	c.free()
	return Closure{
		box: wrap_handle(res.ctx, free_closure_handle, true)
	}
}

// checkpoint wraps `f` with gradient checkpointing (trades memory for compute).
// `mlx_checkpoint` copies the source closure, so freeing `c` is safe.
pub fn checkpoint(f Func) Closure {
	c := new_closure(f)
	res := C.mlx_closure_new()
	setup()
	begin_op()
	rc := C.mlx_checkpoint(&res, c.raw())
	if rc != 0 {
		C.mlx_closure_free(res)
		c.free()
		check(rc)
	}
	c.free()
	return Closure{
		box: wrap_handle(res.ctx, free_closure_handle, true)
	}
}

// eval forces evaluation of all `outputs`.
pub fn eval(outputs []Array) {
	setup()
	begin_op()
	vec := array_vector(outputs)
	defer {
		vec.free()
	}
	check(C.mlx_eval(vec.raw()))
}

// async_eval schedules evaluation of `outputs` without blocking.
pub fn async_eval(outputs []Array) {
	setup()
	begin_op()
	vec := array_vector(outputs)
	defer {
		vec.free()
	}
	check(C.mlx_async_eval(vec.raw()))
}
