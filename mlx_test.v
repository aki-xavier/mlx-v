module mlx

import math

fn test_version() {
	v := version()
	assert v.len > 0
}

// alloc_and_drop creates an array and lets it go out of scope (no explicit
// free), so the GC should reclaim it.
fn alloc_and_drop(i int) {
	x := array_f32([f32(i)], [1])
	assert x.size() == 1
}

fn test_gc() {
	base := live_arrays()
	for i in 0 .. 200 {
		alloc_and_drop(i)
	}
	gc_collect()
	// conservative GC is lazy; allow a small slack, but most must be reclaimed
	assert live_arrays() - base < 10

	// explicit free() decrements immediately
	a := array_f32([f32(1), 2], [2])
	before := live_arrays()
	a.free()
	assert live_arrays() == before - 1
}

fn test_array_create_and_props() {
	a := array_f32([f32(1), 2, 3, 4], [2, 2])
	defer {
		a.free()
	}
	assert a.shape() == [2, 2]
	assert a.ndim() == 2
	assert a.size() == 4
	assert a.dtype() == .float32
	assert a.dim(0) == 2
	assert a.data_f32() == [f32(1), 2, 3, 4]
}

fn test_scalars() {
	i := int_scalar(42)
	f := f32_scalar(2.5)
	b := bool_scalar(true)
	defer {
		i.free()
		f.free()
		b.free()
	}
	assert i.item_i32() == 42
	assert f.item_f32() == 2.5
	assert b.item_bool() == true
}

fn test_elementwise_ops() {
	a := array_f32([f32(1), 2, 3, 4], [2, 2])
	b := array_f32([f32(10), 20, 30, 40], [2, 2])
	defer {
		a.free()
		b.free()
	}

	add := a + b
	sub := b - a
	mul := a * b
	div := b / a
	defer {
		add.free()
		sub.free()
		mul.free()
		div.free()
	}
	assert add.data_f32() == [f32(11), 22, 33, 44]
	assert sub.data_f32() == [f32(9), 18, 27, 36]
	assert mul.data_f32() == [f32(10), 40, 90, 160]
	assert div.data_f32() == [f32(10), 10, 10, 10]
}

fn test_unary_ops() {
	a := array_f32([f32(1), 4, 9, 16], [4])
	defer {
		a.free()
	}
	sq := a.sqrt()
	ex := a.exp()
	defer {
		sq.free()
		ex.free()
	}
	assert sq.data_f32() == [f32(1), 2, 3, 4]
	assert math.abs(f64(ex.data_f32()[0]) - 2.718281828) < 1e-5
}

fn test_reductions() {
	a := array_f32([f32(1), 2, 3, 4], [2, 2])
	defer {
		a.free()
	}
	sum := a.sum()
	mean := a.mean()
	mx := a.max()
	mn := a.min()
	defer {
		sum.free()
		mean.free()
		mx.free()
		mn.free()
	}
	assert sum.item_f32() == 10.0
	assert mean.item_f32() == 2.5
	assert mx.item_f32() == 4.0
	assert mn.item_f32() == 1.0

	// axis reduction
	rowsum := a.sum_axis(1, false)
	defer {
		rowsum.free()
	}
	assert rowsum.data_f32() == [f32(3), 7]
}

fn test_matmul() {
	a := array_f32([f32(1), 2, 3, 4, 5, 6], [2, 3])
	b := array_f32([f32(7), 8, 9, 10, 11, 12], [3, 2])
	defer {
		a.free()
		b.free()
	}
	c := a.matmul(b)
	defer {
		c.free()
	}
	assert c.shape() == [2, 2]
	assert c.data_f32() == [f32(58), 64, 139, 154]
}

fn test_shape_ops() {
	a := arange(0, 6, 1, .float32)
	defer {
		a.free()
	}
	r := a.reshape([2, 3])
	defer {
		r.free()
	}
	assert r.shape() == [2, 3]
	t := r.transpose()
	defer {
		t.free()
	}
	assert t.shape() == [3, 2]
	assert t.data_f32() == [f32(0), 3, 1, 4, 2, 5]
}

fn test_softmax() {
	a := array_f32([f32(1), 2, 3], [3])
	defer {
		a.free()
	}
	sm := a.softmax(false)
	defer {
		sm.free()
	}
	sum := sm.sum()
	defer {
		sum.free()
	}
	assert math.abs(f64(sum.item_f32()) - 1.0) < 1e-5
}

fn test_random() {
	seed := random_key(1234)
	defer {
		seed.free()
	}
	x := random_normal([100], .float32, 0.0, 1.0, seed)
	defer {
		x.free()
	}
	assert x.shape() == [100]
	assert x.dtype() == .float32
}

fn test_linalg_inv_cpu() {
	use_cpu()
	defer {
		use_gpu()
	}
	i := array_f32([f32(4), 7, 2, 6], [2, 2])
	defer {
		i.free()
	}
	inv := i.inv()
	defer {
		inv.free()
	}
	d := inv.data_f32()
	assert math.abs(f64(d[0]) - 0.6) < 1e-5
	assert math.abs(f64(d[1]) + 0.7) < 1e-5
}

fn test_arange_linspace() {
	a := arange(0, 10, 2, .float32)
	defer {
		a.free()
	}
	assert a.data_f32() == [f32(0), 2, 4, 6, 8]

	l := linspace(0, 1, 5, .float32)
	defer {
		l.free()
	}
	assert l.data_f32() == [f32(0), 0.25, 0.5, 0.75, 1.0]
}

// sum_of_squares returns sum(x^2) for the first argument.
fn sum_of_squares(xs []Array) []Array {
	y := xs[0].square().sum()
	return [y]
}

fn test_value_and_grad() {
	x := array_f32([f32(1), 2, 3, 4], [2, 2])
	defer {
		x.free()
	}
	mut vag := value_and_grad(sum_of_squares, [0])
	defer {
		vag.free()
	}
	values, grads := vag.apply([x])
	defer {
		values[0].free()
		grads[0].free()
	}
	assert values[0].item_f32() == 30.0
	assert grads[0].data_f32() == [f32(2), 4, 6, 8]
}

fn test_jvp_vjp() {
	x := array_f32([f32(1), 2, 3, 4], [2, 2])
	defer {
		x.free()
	}
	one := ones([2, 2], .float32)
	defer {
		one.free()
	}

	vals, jvps := jvp(sum_of_squares, [x], [one])
	defer {
		vals[0].free()
		jvps[0].free()
	}
	assert vals[0].item_f32() == 30.0
	assert jvps[0].item_f32() == 20.0

	vals2, vjps := vjp(sum_of_squares, [x], [f32_scalar(1.0)])
	defer {
		vals2[0].free()
		vjps[0].free()
	}
	assert vals2[0].item_f32() == 30.0
	assert vjps[0].data_f32() == [f32(2), 4, 6, 8]
}

fn test_compile_checkpoint() {
	x := array_f32([f32(1), 2, 3, 4], [2, 2])
	defer {
		x.free()
	}

	mut cc := compile(sum_of_squares, false)
	vals := cc.apply([x])
	assert vals[0].item_f32() == 30.0
	vals[0].free()
	cc.free()

	mut cp := checkpoint(sum_of_squares)
	vals2 := cp.apply([x])
	assert vals2[0].item_f32() == 30.0
	vals2[0].free()
	cp.free()
}

fn test_float16_bfloat16() {
	x := array_f32([f32(1.5), -2.5, 0.5], [3])
	defer {
		x.free()
	}
	h := x.astype(.float16)
	defer {
		h.free()
	}
	hd := h.data_f16()
	assert hd[0] == 1.5
	assert hd[1] == -2.5
	assert hd[2] == 0.5

	bf := x.astype(.bfloat16)
	defer {
		bf.free()
	}
	bd := bf.data_bf16()
	assert bd[0] == 1.5
	assert bd[1] == -2.5
}

fn test_complex64() {
	x := array_f32([f32(1), 2, 3, 4], [4])
	defer {
		x.free()
	}
	f := x.rfft(4, 0, .backward)
	defer {
		f.free()
	}
	assert f.dtype() == .complex64
	d := f.data_complex64()
	assert d[0].real == 10.0
	assert d[0].imag == 0.0
	assert d[1].real == -2.0
	assert d[1].imag == 2.0
}
