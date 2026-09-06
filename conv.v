module mlx

// conv.v — convolution and padding operations.
//
// MLX convolutions operate on NHWC inputs with weights shaped
// [out_channels, kH, kW, in_channels].

// conv2d computes a 2D convolution with symmetric stride/padding/dilation.
pub fn conv2d(x Array, w Array, stride int, padding int, groups int) Array {
	res := new_result()
	check_res(C.mlx_conv2d(&res, x.raw(), w.raw(), stride, stride, padding, padding, 1, 1,
		groups, def_stream()), res)
	return wrap_array(res)
}

// conv2d_full computes a 2D convolution with per-axis stride/padding/dilation.
pub fn conv2d_full(x Array, w Array, stride []int, padding []int, dilation []int, groups int) Array {
	res := new_result()
	check_res(C.mlx_conv2d(&res, x.raw(), w.raw(), stride[0], stride[1], padding[0],
		padding[1], dilation[0], dilation[1], groups, def_stream()), res)
	return wrap_array(res)
}

// conv_transpose2d computes a 2D transposed convolution with symmetric
// stride/padding/dilation.
pub fn conv_transpose2d(x Array, w Array, stride int, padding int, output_padding int, groups int) Array {
	res := new_result()
	check_res(C.mlx_conv_transpose2d(&res, x.raw(), w.raw(), stride, stride, padding, padding,
		1, 1, output_padding, output_padding, groups, def_stream()), res)
	return wrap_array(res)
}

// conv_transpose2d_full computes a 2D transposed convolution with per-axis
// stride/padding/dilation/output_padding.
pub fn conv_transpose2d_full(x Array, w Array, stride []int, padding []int, dilation []int, output_padding []int, groups int) Array {
	res := new_result()
	check_res(C.mlx_conv_transpose2d(&res, x.raw(), w.raw(), stride[0], stride[1], padding[0],
		padding[1], dilation[0], dilation[1], output_padding[0], output_padding[1],
		groups, def_stream()), res)
	return wrap_array(res)
}

// conv1d computes a 1D convolution with symmetric stride/padding/dilation.
// x is [n, l, in], weights [out, k, in].
pub fn conv1d(x Array, w Array, stride int, padding int, groups int) Array {
	res := new_result()
	check_res(C.mlx_conv1d(&res, x.raw(), w.raw(), stride, padding, 1, groups, def_stream()),
		res)
	return wrap_array(res)
}

// conv3d computes a 3D convolution with symmetric stride/padding/dilation.
// x is [n, d, h, w, in], weights [out, kd, kh, kw, in].
pub fn conv3d(x Array, w Array, stride int, padding int, groups int) Array {
	res := new_result()
	check_res(C.mlx_conv3d(&res, x.raw(), w.raw(), stride, stride, stride, padding, padding,
		padding, 1, 1, 1, groups, def_stream()), res)
	return wrap_array(res)
}

// pad pads `a` along `axes` with the given low/high widths and a constant value.
pub fn pad(a Array, axes []int, low []int, high []int, value f32) Array {
	v := f32_scalar(value)
	defer {
		v.free()
	}
	res := new_result()
	check_res(C.mlx_pad(&res, a.raw(), axes.data, axes.len, low.data, low.len, high.data,
		high.len, v.raw(), c'constant', def_stream()), res)
	return wrap_array(res)
}

// pad_symmetric pads every axis of `a` by `width` with a constant value.
pub fn pad_symmetric(a Array, width int, value f32) Array {
	v := f32_scalar(value)
	defer {
		v.free()
	}
	res := new_result()
	check_res(C.mlx_pad_symmetric(&res, a.raw(), width, v.raw(), c'constant', def_stream()),
		res)
	return wrap_array(res)
}
