module mlx

// ops.v — ergonomic wrappers around the core MLX array operations.

// --- internal helpers --------------------------------------------------------

// def_stream returns the raw default stream handle (GPU if available, unless
// `use_cpu()` forced the CPU backend).  The wrapper is cached (mlx.c) because
// mlx_default_*_stream_new() heap-allocates a new wrapper on every call.
@[inline]
fn def_stream() C.mlx_stream {
	if C.mlx_v_get_force_cpu() != 0 || !gpu_available() {
		return C.mlx_v_cached_cpu_stream()
	}
	return C.mlx_v_cached_gpu_stream()
}

// new_result prepares error state and returns an empty result array handle.
@[inline]
fn new_result() C.mlx_array {
	setup()
	begin_op()
	return C.mlx_array_new()
}

// --- creation ----------------------------------------------------------------

// zeros creates an array of `shape` filled with zeros.
pub fn zeros(shape []int, dtype Dtype) Array {
	res := new_result()
	check(C.mlx_zeros(&res, shape.data, shape.len, int(dtype), def_stream()))
	return wrap_array(res)
}

// ones creates an array of `shape` filled with ones.
pub fn ones(shape []int, dtype Dtype) Array {
	res := new_result()
	check(C.mlx_ones(&res, shape.data, shape.len, int(dtype), def_stream()))
	return wrap_array(res)
}

// full creates an array of `shape` filled with the value `val`.
pub fn full(shape []int, val Array, dtype Dtype) Array {
	res := new_result()
	check(C.mlx_full(&res, shape.data, shape.len, val.raw(), int(dtype), def_stream()))
	return wrap_array(res)
}

// full_value creates an array of `shape` filled with a float32 scalar.
pub fn full_value(shape []int, val f32, dtype Dtype) Array {
	v := f32_scalar(val)
	defer {
		v.free()
	}
	return full(shape, v, dtype)
}

// zeros_like creates an array of zeros with the shape and dtype of `a`.
pub fn zeros_like(a Array) Array {
	res := new_result()
	check(C.mlx_zeros_like(&res, a.raw(), def_stream()))
	return wrap_array(res)
}

// ones_like creates an array of ones with the shape and dtype of `a`.
pub fn ones_like(a Array) Array {
	res := new_result()
	check(C.mlx_ones_like(&res, a.raw(), def_stream()))
	return wrap_array(res)
}

// full_like creates an array with the shape and dtype of `a`, filled with `val`.
pub fn full_like(a Array, val Array, dtype Dtype) Array {
	res := new_result()
	check(C.mlx_full_like(&res, a.raw(), val.raw(), int(dtype), def_stream()))
	return wrap_array(res)
}

// eye creates an identity-like matrix (ones on the k-th diagonal).
pub fn eye(n int, m int, k int, dtype Dtype) Array {
	res := new_result()
	check(C.mlx_eye(&res, n, m, k, int(dtype), def_stream()))
	return wrap_array(res)
}

// identity creates an n-by-n identity matrix.
pub fn identity(n int, dtype Dtype) Array {
	res := new_result()
	check(C.mlx_identity(&res, n, int(dtype), def_stream()))
	return wrap_array(res)
}

// arange creates a 1-d array of evenly spaced values in [start, stop).
pub fn arange(start f64, stop f64, step f64, dtype Dtype) Array {
	res := new_result()
	check(C.mlx_arange(&res, start, stop, step, int(dtype), def_stream()))
	return wrap_array(res)
}

// linspace creates `num` evenly spaced values between `start` and `stop`.
pub fn linspace(start f64, stop f64, num int, dtype Dtype) Array {
	res := new_result()
	check(C.mlx_linspace(&res, start, stop, num, int(dtype), def_stream()))
	return wrap_array(res)
}

// tri creates a triangular matrix.
pub fn tri(n int, m int, k int, dtype Dtype) Array {
	res := new_result()
	check(C.mlx_tri(&res, n, m, k, int(dtype), def_stream()))
	return wrap_array(res)
}

// --- unary elementwise -------------------------------------------------------

pub fn (a Array) abs() Array {
	res := new_result()
	check(C.mlx_abs(&res, a.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) negative() Array {
	res := new_result()
	check(C.mlx_negative(&res, a.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) exp() Array {
	res := new_result()
	check(C.mlx_exp(&res, a.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) expm1() Array {
	res := new_result()
	check(C.mlx_expm1(&res, a.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) log() Array {
	res := new_result()
	check(C.mlx_log(&res, a.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) log2() Array {
	res := new_result()
	check(C.mlx_log2(&res, a.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) log10() Array {
	res := new_result()
	check(C.mlx_log10(&res, a.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) log1p() Array {
	res := new_result()
	check(C.mlx_log1p(&res, a.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) sqrt() Array {
	res := new_result()
	check(C.mlx_sqrt(&res, a.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) rsqrt() Array {
	res := new_result()
	check(C.mlx_rsqrt(&res, a.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) square() Array {
	res := new_result()
	check(C.mlx_square(&res, a.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) reciprocal() Array {
	res := new_result()
	check(C.mlx_reciprocal(&res, a.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) sin() Array {
	res := new_result()
	check(C.mlx_sin(&res, a.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) cos() Array {
	res := new_result()
	check(C.mlx_cos(&res, a.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) tan() Array {
	res := new_result()
	check(C.mlx_tan(&res, a.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) sinh() Array {
	res := new_result()
	check(C.mlx_sinh(&res, a.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) cosh() Array {
	res := new_result()
	check(C.mlx_cosh(&res, a.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) tanh() Array {
	res := new_result()
	check(C.mlx_tanh(&res, a.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) arcsin() Array {
	res := new_result()
	check(C.mlx_arcsin(&res, a.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) arccos() Array {
	res := new_result()
	check(C.mlx_arccos(&res, a.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) arctan() Array {
	res := new_result()
	check(C.mlx_arctan(&res, a.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) arcsinh() Array {
	res := new_result()
	check(C.mlx_arcsinh(&res, a.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) arccosh() Array {
	res := new_result()
	check(C.mlx_arccosh(&res, a.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) arctanh() Array {
	res := new_result()
	check(C.mlx_arctanh(&res, a.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) degrees() Array {
	res := new_result()
	check(C.mlx_degrees(&res, a.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) radians() Array {
	res := new_result()
	check(C.mlx_radians(&res, a.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) sign() Array {
	res := new_result()
	check(C.mlx_sign(&res, a.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) sigmoid() Array {
	res := new_result()
	check(C.mlx_sigmoid(&res, a.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) erf() Array {
	res := new_result()
	check(C.mlx_erf(&res, a.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) erfinv() Array {
	res := new_result()
	check(C.mlx_erfinv(&res, a.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) floor() Array {
	res := new_result()
	check(C.mlx_floor(&res, a.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) ceil() Array {
	res := new_result()
	check(C.mlx_ceil(&res, a.raw(), def_stream()))
	return wrap_array(res)
}

// round rounds to `decimals` decimal places.
pub fn (a Array) round(decimals int) Array {
	res := new_result()
	check(C.mlx_round(&res, a.raw(), decimals, def_stream()))
	return wrap_array(res)
}

pub fn (a Array) logical_not() Array {
	res := new_result()
	check(C.mlx_logical_not(&res, a.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) bitwise_invert() Array {
	res := new_result()
	check(C.mlx_bitwise_invert(&res, a.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) isnan() Array {
	res := new_result()
	check(C.mlx_isnan(&res, a.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) isinf() Array {
	res := new_result()
	check(C.mlx_isinf(&res, a.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) isfinite() Array {
	res := new_result()
	check(C.mlx_isfinite(&res, a.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) isposinf() Array {
	res := new_result()
	check(C.mlx_isposinf(&res, a.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) isneginf() Array {
	res := new_result()
	check(C.mlx_isneginf(&res, a.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) conjugate() Array {
	res := new_result()
	check(C.mlx_conjugate(&res, a.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) real() Array {
	res := new_result()
	check(C.mlx_real(&res, a.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) imag() Array {
	res := new_result()
	check(C.mlx_imag(&res, a.raw(), def_stream()))
	return wrap_array(res)
}

// astype casts the array to `dtype`.
pub fn (a Array) astype(dtype Dtype) Array {
	res := new_result()
	check(C.mlx_astype(&res, a.raw(), int(dtype), def_stream()))
	return wrap_array(res)
}

// view reinterprets the array as `dtype` without copying.
pub fn (a Array) view(dtype Dtype) Array {
	res := new_result()
	check(C.mlx_view(&res, a.raw(), int(dtype), def_stream()))
	return wrap_array(res)
}

// copy returns a copy of the array.
pub fn (a Array) copy() Array {
	res := new_result()
	check(C.mlx_copy(&res, a.raw(), def_stream()))
	return wrap_array(res)
}

// contiguous returns a contiguous copy of the array.
pub fn (a Array) contiguous(allow_col_major bool) Array {
	res := new_result()
	check(C.mlx_contiguous(&res, a.raw(), allow_col_major, def_stream()))
	return wrap_array(res)
}

// stop_gradient returns the array detached from the autograd graph.
pub fn (a Array) stop_gradient() Array {
	res := new_result()
	check(C.mlx_stop_gradient(&res, a.raw(), def_stream()))
	return wrap_array(res)
}

// --- binary elementwise ------------------------------------------------------

pub fn (a Array) add(b Array) Array {
	res := new_result()
	check(C.mlx_add(&res, a.raw(), b.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) subtract(b Array) Array {
	res := new_result()
	check(C.mlx_subtract(&res, a.raw(), b.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) multiply(b Array) Array {
	res := new_result()
	check(C.mlx_multiply(&res, a.raw(), b.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) divide(b Array) Array {
	res := new_result()
	check(C.mlx_divide(&res, a.raw(), b.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) floor_divide(b Array) Array {
	res := new_result()
	check(C.mlx_floor_divide(&res, a.raw(), b.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) power(b Array) Array {
	res := new_result()
	check(C.mlx_power(&res, a.raw(), b.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) remainder(b Array) Array {
	res := new_result()
	check(C.mlx_remainder(&res, a.raw(), b.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) maximum(b Array) Array {
	res := new_result()
	check(C.mlx_maximum(&res, a.raw(), b.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) minimum(b Array) Array {
	res := new_result()
	check(C.mlx_minimum(&res, a.raw(), b.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) arctan2(b Array) Array {
	res := new_result()
	check(C.mlx_arctan2(&res, a.raw(), b.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) logaddexp(b Array) Array {
	res := new_result()
	check(C.mlx_logaddexp(&res, a.raw(), b.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) equal(b Array) Array {
	res := new_result()
	check(C.mlx_equal(&res, a.raw(), b.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) not_equal(b Array) Array {
	res := new_result()
	check(C.mlx_not_equal(&res, a.raw(), b.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) greater(b Array) Array {
	res := new_result()
	check(C.mlx_greater(&res, a.raw(), b.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) greater_equal(b Array) Array {
	res := new_result()
	check(C.mlx_greater_equal(&res, a.raw(), b.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) less(b Array) Array {
	res := new_result()
	check(C.mlx_less(&res, a.raw(), b.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) less_equal(b Array) Array {
	res := new_result()
	check(C.mlx_less_equal(&res, a.raw(), b.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) logical_and(b Array) Array {
	res := new_result()
	check(C.mlx_logical_and(&res, a.raw(), b.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) logical_or(b Array) Array {
	res := new_result()
	check(C.mlx_logical_or(&res, a.raw(), b.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) bitwise_and(b Array) Array {
	res := new_result()
	check(C.mlx_bitwise_and(&res, a.raw(), b.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) bitwise_or(b Array) Array {
	res := new_result()
	check(C.mlx_bitwise_or(&res, a.raw(), b.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) bitwise_xor(b Array) Array {
	res := new_result()
	check(C.mlx_bitwise_xor(&res, a.raw(), b.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) left_shift(b Array) Array {
	res := new_result()
	check(C.mlx_left_shift(&res, a.raw(), b.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) right_shift(b Array) Array {
	res := new_result()
	check(C.mlx_right_shift(&res, a.raw(), b.raw(), def_stream()))
	return wrap_array(res)
}

// --- operators ---------------------------------------------------------------

pub fn (a Array) + (b Array) Array {
	return a.add(b)
}

pub fn (a Array) - (b Array) Array {
	return a.subtract(b)
}

pub fn (a Array) * (b Array) Array {
	return a.multiply(b)
}

pub fn (a Array) / (b Array) Array {
	return a.divide(b)
}

pub fn (a Array) % (b Array) Array {
	return a.remainder(b)
}

// --- reductions --------------------------------------------------------------

// sum reduces `a` over all elements.
pub fn (a Array) sum() Array {
	res := new_result()
	check(C.mlx_sum(&res, a.raw(), false, def_stream()))
	return wrap_array(res)
}

// sum_axis reduces `a` along `axis`.
pub fn (a Array) sum_axis(axis int, keepdims bool) Array {
	res := new_result()
	check(C.mlx_sum_axis(&res, a.raw(), axis, keepdims, def_stream()))
	return wrap_array(res)
}

// sum_axes reduces `a` along `axes`.
pub fn (a Array) sum_axes(axes []int, keepdims bool) Array {
	res := new_result()
	check(C.mlx_sum_axes(&res, a.raw(), axes.data, axes.len, keepdims, def_stream()))
	return wrap_array(res)
}

pub fn (a Array) mean() Array {
	res := new_result()
	check(C.mlx_mean(&res, a.raw(), false, def_stream()))
	return wrap_array(res)
}

pub fn (a Array) mean_axis(axis int, keepdims bool) Array {
	res := new_result()
	check(C.mlx_mean_axis(&res, a.raw(), axis, keepdims, def_stream()))
	return wrap_array(res)
}

pub fn (a Array) mean_axes(axes []int, keepdims bool) Array {
	res := new_result()
	check(C.mlx_mean_axes(&res, a.raw(), axes.data, axes.len, keepdims, def_stream()))
	return wrap_array(res)
}

pub fn (a Array) max() Array {
	res := new_result()
	check(C.mlx_max(&res, a.raw(), false, def_stream()))
	return wrap_array(res)
}

pub fn (a Array) max_axis(axis int, keepdims bool) Array {
	res := new_result()
	check(C.mlx_max_axis(&res, a.raw(), axis, keepdims, def_stream()))
	return wrap_array(res)
}

pub fn (a Array) max_axes(axes []int, keepdims bool) Array {
	res := new_result()
	check(C.mlx_max_axes(&res, a.raw(), axes.data, axes.len, keepdims, def_stream()))
	return wrap_array(res)
}

pub fn (a Array) min() Array {
	res := new_result()
	check(C.mlx_min(&res, a.raw(), false, def_stream()))
	return wrap_array(res)
}

pub fn (a Array) min_axis(axis int, keepdims bool) Array {
	res := new_result()
	check(C.mlx_min_axis(&res, a.raw(), axis, keepdims, def_stream()))
	return wrap_array(res)
}

pub fn (a Array) min_axes(axes []int, keepdims bool) Array {
	res := new_result()
	check(C.mlx_min_axes(&res, a.raw(), axes.data, axes.len, keepdims, def_stream()))
	return wrap_array(res)
}

pub fn (a Array) prod() Array {
	res := new_result()
	check(C.mlx_prod(&res, a.raw(), false, def_stream()))
	return wrap_array(res)
}

pub fn (a Array) prod_axis(axis int, keepdims bool) Array {
	res := new_result()
	check(C.mlx_prod_axis(&res, a.raw(), axis, keepdims, def_stream()))
	return wrap_array(res)
}

pub fn (a Array) all() Array {
	res := new_result()
	check(C.mlx_all(&res, a.raw(), false, def_stream()))
	return wrap_array(res)
}

pub fn (a Array) any() Array {
	res := new_result()
	check(C.mlx_any(&res, a.raw(), false, def_stream()))
	return wrap_array(res)
}

pub fn (a Array) std(ddof int) Array {
	res := new_result()
	check(C.mlx_std(&res, a.raw(), false, ddof, def_stream()))
	return wrap_array(res)
}

pub fn (a Array) var(ddof int) Array {
	res := new_result()
	check(C.mlx_var(&res, a.raw(), false, ddof, def_stream()))
	return wrap_array(res)
}

pub fn (a Array) logsumexp() Array {
	res := new_result()
	check(C.mlx_logsumexp(&res, a.raw(), false, def_stream()))
	return wrap_array(res)
}

// median returns the median of all elements.
pub fn (a Array) median() Array {
	res := new_result()
	check(C.mlx_median(&res, a.raw(), unsafe { nil }, 0, false, def_stream()))
	return wrap_array(res)
}

// var_axis returns the variance along `axis` (population variance ddof=0 by default).
pub fn (a Array) var_axis(axis int, keepdims bool, ddof int) Array {
	res := new_result()
	check(C.mlx_var_axis(&res, a.raw(), axis, keepdims, ddof, def_stream()))
	return wrap_array(res)
}

// std_axis returns the standard deviation along `axis`.
pub fn (a Array) std_axis(axis int, keepdims bool, ddof int) Array {
	res := new_result()
	check(C.mlx_std_axis(&res, a.raw(), axis, keepdims, ddof, def_stream()))
	return wrap_array(res)
}

// --- cumulative & sorting ----------------------------------------------------

pub fn (a Array) cumsum(axis int, reverse bool, inclusive bool) Array {
	res := new_result()
	check(C.mlx_cumsum(&res, a.raw(), axis, reverse, inclusive, def_stream()))
	return wrap_array(res)
}

pub fn (a Array) cumprod(axis int, reverse bool, inclusive bool) Array {
	res := new_result()
	check(C.mlx_cumprod(&res, a.raw(), axis, reverse, inclusive, def_stream()))
	return wrap_array(res)
}

pub fn (a Array) cummax(axis int, reverse bool, inclusive bool) Array {
	res := new_result()
	check(C.mlx_cummax(&res, a.raw(), axis, reverse, inclusive, def_stream()))
	return wrap_array(res)
}

pub fn (a Array) cummin(axis int, reverse bool, inclusive bool) Array {
	res := new_result()
	check(C.mlx_cummin(&res, a.raw(), axis, reverse, inclusive, def_stream()))
	return wrap_array(res)
}

pub fn (a Array) argmax() Array {
	res := new_result()
	check(C.mlx_argmax(&res, a.raw(), false, def_stream()))
	return wrap_array(res)
}

pub fn (a Array) argmax_axis(axis int, keepdims bool) Array {
	res := new_result()
	check(C.mlx_argmax_axis(&res, a.raw(), axis, keepdims, def_stream()))
	return wrap_array(res)
}

pub fn (a Array) argmin() Array {
	res := new_result()
	check(C.mlx_argmin(&res, a.raw(), false, def_stream()))
	return wrap_array(res)
}

pub fn (a Array) argmin_axis(axis int, keepdims bool) Array {
	res := new_result()
	check(C.mlx_argmin_axis(&res, a.raw(), axis, keepdims, def_stream()))
	return wrap_array(res)
}

pub fn (a Array) sort() Array {
	res := new_result()
	check(C.mlx_sort(&res, a.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) sort_axis(axis int) Array {
	res := new_result()
	check(C.mlx_sort_axis(&res, a.raw(), axis, def_stream()))
	return wrap_array(res)
}

pub fn (a Array) argsort() Array {
	res := new_result()
	check(C.mlx_argsort(&res, a.raw(), def_stream()))
	return wrap_array(res)
}

pub fn (a Array) argsort_axis(axis int) Array {
	res := new_result()
	check(C.mlx_argsort_axis(&res, a.raw(), axis, def_stream()))
	return wrap_array(res)
}

pub fn (a Array) topk(k int) Array {
	res := new_result()
	check(C.mlx_topk(&res, a.raw(), k, def_stream()))
	return wrap_array(res)
}

pub fn (a Array) topk_axis(k int, axis int) Array {
	res := new_result()
	check(C.mlx_topk_axis(&res, a.raw(), k, axis, def_stream()))
	return wrap_array(res)
}

// --- softmax -----------------------------------------------------------------

pub fn (a Array) softmax(precise bool) Array {
	res := new_result()
	check(C.mlx_softmax(&res, a.raw(), precise, def_stream()))
	return wrap_array(res)
}

pub fn (a Array) softmax_axis(axis int, precise bool) Array {
	res := new_result()
	check(C.mlx_softmax_axis(&res, a.raw(), axis, precise, def_stream()))
	return wrap_array(res)
}

// --- shape & manipulation ----------------------------------------------------

// reshape returns the array viewed with a new `shape`.
pub fn (a Array) reshape(shape []int) Array {
	res := new_result()
	check(C.mlx_reshape(&res, a.raw(), shape.data, shape.len, def_stream()))
	return wrap_array(res)
}

// transpose reverses the array dimensions.
pub fn (a Array) transpose() Array {
	res := new_result()
	check(C.mlx_transpose(&res, a.raw(), def_stream()))
	return wrap_array(res)
}

// transpose_axes permutes dimensions according to `axes`.
pub fn (a Array) transpose_axes(axes []int) Array {
	res := new_result()
	check(C.mlx_transpose_axes(&res, a.raw(), axes.data, axes.len, def_stream()))
	return wrap_array(res)
}

// swapaxes swaps two dimensions.
pub fn (a Array) swapaxes(axis1 int, axis2 int) Array {
	res := new_result()
	check(C.mlx_swapaxes(&res, a.raw(), axis1, axis2, def_stream()))
	return wrap_array(res)
}

// moveaxis moves `source` to `destination`.
pub fn (a Array) moveaxis(source int, destination int) Array {
	res := new_result()
	check(C.mlx_moveaxis(&res, a.raw(), source, destination, def_stream()))
	return wrap_array(res)
}

// expand_dims inserts a new dimension of size 1 at `axis`.
pub fn (a Array) expand_dims(axis int) Array {
	res := new_result()
	check(C.mlx_expand_dims(&res, a.raw(), axis, def_stream()))
	return wrap_array(res)
}

// squeeze removes singleton dimensions.
pub fn (a Array) squeeze() Array {
	res := new_result()
	check(C.mlx_squeeze(&res, a.raw(), def_stream()))
	return wrap_array(res)
}

// squeeze_axis removes the singleton dimension at `axis`.
pub fn (a Array) squeeze_axis(axis int) Array {
	res := new_result()
	check(C.mlx_squeeze_axis(&res, a.raw(), axis, def_stream()))
	return wrap_array(res)
}

// flatten collapses dimensions from `start_axis` to `end_axis`.
pub fn (a Array) flatten(start_axis int, end_axis int) Array {
	res := new_result()
	check(C.mlx_flatten(&res, a.raw(), start_axis, end_axis, def_stream()))
	return wrap_array(res)
}

// broadcast_to broadcasts the array to `shape`.
pub fn (a Array) broadcast_to(shape []int) Array {
	res := new_result()
	check(C.mlx_broadcast_to(&res, a.raw(), shape.data, shape.len, def_stream()))
	return wrap_array(res)
}

// tile repeats the array `reps` times along each dimension.
pub fn (a Array) tile(reps []int) Array {
	res := new_result()
	check(C.mlx_tile(&res, a.raw(), reps.data, reps.len, def_stream()))
	return wrap_array(res)
}

// repeat repeats the array `repeats` times.
pub fn (a Array) repeat(repeats int) Array {
	res := new_result()
	check(C.mlx_repeat(&res, a.raw(), repeats, def_stream()))
	return wrap_array(res)
}

// tril returns the lower triangle of the array.
pub fn (a Array) tril(k int) Array {
	res := new_result()
	check(C.mlx_tril(&res, a.raw(), k, def_stream()))
	return wrap_array(res)
}

// triu returns the upper triangle of the array.
pub fn (a Array) triu(k int) Array {
	res := new_result()
	check(C.mlx_triu(&res, a.raw(), k, def_stream()))
	return wrap_array(res)
}

// diag extracts the k-th diagonal or builds a diagonal matrix.
pub fn (a Array) diag(k int) Array {
	res := new_result()
	check(C.mlx_diag(&res, a.raw(), k, def_stream()))
	return wrap_array(res)
}

// --- matmul & tensor products ------------------------------------------------

// matmul returns the matrix product of `a` and `b`.
pub fn (a Array) matmul(b Array) Array {
	res := new_result()
	check(C.mlx_matmul(&res, a.raw(), b.raw(), def_stream()))
	return wrap_array(res)
}

// inner returns the inner product of `a` and `b`.
pub fn (a Array) inner(b Array) Array {
	res := new_result()
	check(C.mlx_inner(&res, a.raw(), b.raw(), def_stream()))
	return wrap_array(res)
}

// outer returns the outer product of `a` and `b`.
pub fn (a Array) outer(b Array) Array {
	res := new_result()
	check(C.mlx_outer(&res, a.raw(), b.raw(), def_stream()))
	return wrap_array(res)
}

// kron returns the Kronecker product of `a` and `b`.
pub fn (a Array) kron(b Array) Array {
	res := new_result()
	check(C.mlx_kron(&res, a.raw(), b.raw(), def_stream()))
	return wrap_array(res)
}

// einsum evaluates an Einstein summation over the operands.
pub fn einsum(subscripts string, operands []Array) Array {
	res := new_result()
	mut vec := array_vector(operands)
	defer {
		vec.free()
	}
	check(C.mlx_einsum(&res, subscripts.str, vec.ctx, def_stream()))
	return wrap_array(res)
}

// tensordot contracts two arrays along given axes.
pub fn (a Array) tensordot(b Array, axes_a []int, axes_b []int) Array {
	res := new_result()
	check(C.mlx_tensordot(&res, a.raw(), b.raw(), axes_a.data, axes_a.len, axes_b.data, axes_b.len,
		def_stream()))
	return wrap_array(res)
}

// --- indexing & updates ------------------------------------------------------

// take takes elements at `indices`.
pub fn (a Array) take(indices Array) Array {
	res := new_result()
	check(C.mlx_take(&res, a.raw(), indices.raw(), def_stream()))
	return wrap_array(res)
}

// take_axis takes elements along `axis` at `indices`.
pub fn (a Array) take_axis(indices Array, axis int) Array {
	res := new_result()
	check(C.mlx_take_axis(&res, a.raw(), indices.raw(), axis, def_stream()))
	return wrap_array(res)
}

// take_along_axis takes elements along `axis` according to `indices`.
pub fn (a Array) take_along_axis(indices Array, axis int) Array {
	res := new_result()
	check(C.mlx_take_along_axis(&res, a.raw(), indices.raw(), axis, def_stream()))
	return wrap_array(res)
}

// clip clamps the array between `a_min` and `a_max` (either may be an empty array).
pub fn (a Array) clip(a_min Array, a_max Array) Array {
	res := new_result()
	check(C.mlx_clip(&res, a.raw(), a_min.raw(), a_max.raw(), def_stream()))
	return wrap_array(res)
}

// where selects elements from `x` or `y` based on `condition`.
pub fn where(condition Array, x Array, y Array) Array {
	res := new_result()
	check(C.mlx_where(&res, condition.raw(), x.raw(), y.raw(), def_stream()))
	return wrap_array(res)
}

// pad pads the array along `axes` by `low`/`high` amounts (mode: "edge",
// "constant", "symmetric", "reflect").
pub fn (a Array) pad(axes []int, low []int, high []int, value Array, mode string) Array {
	res := new_result()
	check(C.mlx_pad(&res, a.raw(), axes.data, usize(axes.len), low.data, usize(low.len),
		high.data, usize(high.len), value.raw(), mode.str, def_stream()))
	return wrap_array(res)
}

// --- combine / split ---------------------------------------------------------

// concatenate joins `arrays` along `axis`.
pub fn concatenate(arrays []Array, axis int) Array {
	res := new_result()
	mut vec := array_vector(arrays)
	defer {
		vec.free()
	}
	check(C.mlx_concatenate_axis(&res, vec.ctx, axis, def_stream()))
	return wrap_array(res)
}

// stack joins `arrays` along a new `axis`.
pub fn stack(arrays []Array, axis int) Array {
	res := new_result()
	mut vec := array_vector(arrays)
	defer {
		vec.free()
	}
	check(C.mlx_stack_axis(&res, vec.ctx, axis, def_stream()))
	return wrap_array(res)
}

// split splits the array into `num_splits` equal parts along `axis`.
pub fn (a Array) split(num_splits int, axis int) []Array {
	res := C.mlx_vector_array_new()
	check(C.mlx_split(&res, a.raw(), num_splits, axis, def_stream()))
	return array_vector_to_slice(res)
}
