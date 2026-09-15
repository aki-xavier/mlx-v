module mlx

// mlx-c declares these as { int value; bool has_value; } and
// { mlx_dtype value; bool has_value; }, i.e. 8 bytes each on arm64.  The value
// field must stay 32-bit: declaring it as V's 64-bit `int` would make the
// struct 16 bytes and move `has_value` off its C offset, so any call that
// passes one by value would silently drop the supplied setting.
fn test_optional_struct_layout() {
	assert sizeof(C.mlx_optional_int) == 8
	assert sizeof(C.mlx_optional_dtype) == 8
	assert sizeof(C.mlx_optional_float) == 8
}

fn opt_i(v int) C.mlx_optional_int {
	return C.mlx_optional_int{
		value:     i32(v)
		has_value: true
	}
}

// test_quantize_dequantize_optional_structs round-trips a known matrix through
// quantize/dequantize, which passes mlx_optional_int (group_size, bits) and
// mlx_optional_dtype by value across the C ABI.
//
// group_size/bits are deliberately non-default (MLX defaults to 64/4): if the
// structs were mis-laid-out the settings would be ignored and `scales` would
// come back shaped [64, 2] instead of [64, 4].
fn test_quantize_dequantize_optional_structs() {
	mut data := []f32{len: 64 * 128}
	for i in 0 .. data.len {
		data[i] = f32(i % 17) * 0.25 - 2.0
	}
	w := array_f32(data, [64, 128])
	defer {
		w.free()
	}
	// An empty array is MLX's "not provided" for optional array arguments.
	no_scale := empty()
	defer {
		no_scale.free()
	}

	group_size := 32
	bits := 8
	qs := C.mlx_vector_array_new()
	defer {
		C.mlx_vector_array_free(qs)
	}
	begin_op()
	check(C.mlx_quantize(&qs, w.raw(), opt_i(group_size), opt_i(bits), c'affine', no_scale.raw(),
		def_stream()))

	mut wq := C.mlx_array_new()
	mut scales := C.mlx_array_new()
	mut biases := C.mlx_array_new()
	C.mlx_vector_array_get(&wq, qs, 0)
	C.mlx_vector_array_get(&scales, qs, 1)
	C.mlx_vector_array_get(&biases, qs, 2)
	a_wq := wrap_array(wq)
	defer {
		a_wq.free()
	}
	a_scales := wrap_array(scales)
	defer {
		a_scales.free()
	}
	a_biases := wrap_array(biases)
	defer {
		a_biases.free()
	}
	// Quantized weights are packed into 32-bit words: last dim = cols * bits / 32.
	assert a_wq.shape() == [64, 128 * bits / 32]
	assert a_scales.shape() == [64, 128 / group_size]
	assert a_biases.shape() == [64, 128 / group_size]

	mut dq := C.mlx_array_new()
	begin_op()
	check(C.mlx_dequantize(&dq, a_wq.raw(), a_scales.raw(), a_biases.raw(), opt_i(group_size),
		opt_i(bits), c'affine', no_scale.raw(), C.mlx_optional_dtype{
		value:     i32(Dtype.float32)
		has_value: true
	}, def_stream()))
	a_dq := wrap_array(dq)
	defer {
		a_dq.free()
	}

	back := a_dq.data_f32()
	assert back.len == data.len
	mut maxerr := f32(0)
	for i in 0 .. data.len {
		e := if data[i] > back[i] { data[i] - back[i] } else { back[i] - data[i] }
		if e > maxerr {
			maxerr = e
		}
	}
	// 8-bit affine quantization of this data is well within this bound.
	assert maxerr < 0.5
}
