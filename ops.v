module mlx

// ops.v — ergonomic wrappers around the core MLX array operations.

// --- internal helpers --------------------------------------------------------

// def_stream returns the raw default stream handle (GPU if available, unless
// `use_cpu()` forced the CPU backend).
@[inline]
fn def_stream() C.mlx_stream {
	if C.mlx_v_get_force_cpu() != 0 || !gpu_available() {
		return C.mlx_default_cpu_stream_new()
	}
	return C.mlx_default_gpu_stream_new()
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
	return Array{
		ctx: res
	}
}

// ones creates an array of `shape` filled with ones.
pub fn ones(shape []int, dtype Dtype) Array {
	res := new_result()
	check(C.mlx_ones(&res, shape.data, shape.len, int(dtype), def_stream()))
	return Array{
		ctx: res
	}
}

// full creates an array of `shape` filled with the value `val`.
pub fn full(shape []int, val Array, dtype Dtype) Array {
	res := new_result()
	check(C.mlx_full(&res, shape.data, shape.len, val.ctx, int(dtype), def_stream()))
	return Array{
		ctx: res
	}
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
	check(C.mlx_zeros_like(&res, a.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

// ones_like creates an array of ones with the shape and dtype of `a`.
pub fn ones_like(a Array) Array {
	res := new_result()
	check(C.mlx_ones_like(&res, a.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

// full_like creates an array with the shape and dtype of `a`, filled with `val`.
pub fn full_like(a Array, val Array, dtype Dtype) Array {
	res := new_result()
	check(C.mlx_full_like(&res, a.ctx, val.ctx, int(dtype), def_stream()))
	return Array{
		ctx: res
	}
}

// eye creates an identity-like matrix (ones on the k-th diagonal).
pub fn eye(n int, m int, k int, dtype Dtype) Array {
	res := new_result()
	check(C.mlx_eye(&res, n, m, k, int(dtype), def_stream()))
	return Array{
		ctx: res
	}
}

// identity creates an n-by-n identity matrix.
pub fn identity(n int, dtype Dtype) Array {
	res := new_result()
	check(C.mlx_identity(&res, n, int(dtype), def_stream()))
	return Array{
		ctx: res
	}
}

// arange creates a 1-d array of evenly spaced values in [start, stop).
pub fn arange(start f64, stop f64, step f64, dtype Dtype) Array {
	res := new_result()
	check(C.mlx_arange(&res, start, stop, step, int(dtype), def_stream()))
	return Array{
		ctx: res
	}
}

// linspace creates `num` evenly spaced values between `start` and `stop`.
pub fn linspace(start f64, stop f64, num int, dtype Dtype) Array {
	res := new_result()
	check(C.mlx_linspace(&res, start, stop, num, int(dtype), def_stream()))
	return Array{
		ctx: res
	}
}

// tri creates a triangular matrix.
pub fn tri(n int, m int, k int, dtype Dtype) Array {
	res := new_result()
	check(C.mlx_tri(&res, n, m, k, int(dtype), def_stream()))
	return Array{
		ctx: res
	}
}

// --- unary elementwise -------------------------------------------------------

pub fn (a Array) abs() Array {
	res := new_result()
	check(C.mlx_abs(&res, a.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) negative() Array {
	res := new_result()
	check(C.mlx_negative(&res, a.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) exp() Array {
	res := new_result()
	check(C.mlx_exp(&res, a.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) expm1() Array {
	res := new_result()
	check(C.mlx_expm1(&res, a.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) log() Array {
	res := new_result()
	check(C.mlx_log(&res, a.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) log2() Array {
	res := new_result()
	check(C.mlx_log2(&res, a.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) log10() Array {
	res := new_result()
	check(C.mlx_log10(&res, a.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) log1p() Array {
	res := new_result()
	check(C.mlx_log1p(&res, a.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) sqrt() Array {
	res := new_result()
	check(C.mlx_sqrt(&res, a.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) rsqrt() Array {
	res := new_result()
	check(C.mlx_rsqrt(&res, a.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) square() Array {
	res := new_result()
	check(C.mlx_square(&res, a.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) reciprocal() Array {
	res := new_result()
	check(C.mlx_reciprocal(&res, a.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) sin() Array {
	res := new_result()
	check(C.mlx_sin(&res, a.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) cos() Array {
	res := new_result()
	check(C.mlx_cos(&res, a.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) tan() Array {
	res := new_result()
	check(C.mlx_tan(&res, a.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) sinh() Array {
	res := new_result()
	check(C.mlx_sinh(&res, a.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) cosh() Array {
	res := new_result()
	check(C.mlx_cosh(&res, a.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) tanh() Array {
	res := new_result()
	check(C.mlx_tanh(&res, a.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) arcsin() Array {
	res := new_result()
	check(C.mlx_arcsin(&res, a.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) arccos() Array {
	res := new_result()
	check(C.mlx_arccos(&res, a.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) arctan() Array {
	res := new_result()
	check(C.mlx_arctan(&res, a.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) arcsinh() Array {
	res := new_result()
	check(C.mlx_arcsinh(&res, a.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) arccosh() Array {
	res := new_result()
	check(C.mlx_arccosh(&res, a.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) arctanh() Array {
	res := new_result()
	check(C.mlx_arctanh(&res, a.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) degrees() Array {
	res := new_result()
	check(C.mlx_degrees(&res, a.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) radians() Array {
	res := new_result()
	check(C.mlx_radians(&res, a.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) sign() Array {
	res := new_result()
	check(C.mlx_sign(&res, a.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) sigmoid() Array {
	res := new_result()
	check(C.mlx_sigmoid(&res, a.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) erf() Array {
	res := new_result()
	check(C.mlx_erf(&res, a.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) erfinv() Array {
	res := new_result()
	check(C.mlx_erfinv(&res, a.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) floor() Array {
	res := new_result()
	check(C.mlx_floor(&res, a.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) ceil() Array {
	res := new_result()
	check(C.mlx_ceil(&res, a.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

// round rounds to `decimals` decimal places.
pub fn (a Array) round(decimals int) Array {
	res := new_result()
	check(C.mlx_round(&res, a.ctx, decimals, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) logical_not() Array {
	res := new_result()
	check(C.mlx_logical_not(&res, a.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) bitwise_invert() Array {
	res := new_result()
	check(C.mlx_bitwise_invert(&res, a.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) isnan() Array {
	res := new_result()
	check(C.mlx_isnan(&res, a.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) isinf() Array {
	res := new_result()
	check(C.mlx_isinf(&res, a.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) isfinite() Array {
	res := new_result()
	check(C.mlx_isfinite(&res, a.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) isposinf() Array {
	res := new_result()
	check(C.mlx_isposinf(&res, a.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) isneginf() Array {
	res := new_result()
	check(C.mlx_isneginf(&res, a.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) conjugate() Array {
	res := new_result()
	check(C.mlx_conjugate(&res, a.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) real() Array {
	res := new_result()
	check(C.mlx_real(&res, a.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) imag() Array {
	res := new_result()
	check(C.mlx_imag(&res, a.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

// astype casts the array to `dtype`.
pub fn (a Array) astype(dtype Dtype) Array {
	res := new_result()
	check(C.mlx_astype(&res, a.ctx, int(dtype), def_stream()))
	return Array{
		ctx: res
	}
}

// view reinterprets the array as `dtype` without copying.
pub fn (a Array) view(dtype Dtype) Array {
	res := new_result()
	check(C.mlx_view(&res, a.ctx, int(dtype), def_stream()))
	return Array{
		ctx: res
	}
}

// copy returns a copy of the array.
pub fn (a Array) copy() Array {
	res := new_result()
	check(C.mlx_copy(&res, a.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

// contiguous returns a contiguous copy of the array.
pub fn (a Array) contiguous(allow_col_major bool) Array {
	res := new_result()
	check(C.mlx_contiguous(&res, a.ctx, allow_col_major, def_stream()))
	return Array{
		ctx: res
	}
}

// stop_gradient returns the array detached from the autograd graph.
pub fn (a Array) stop_gradient() Array {
	res := new_result()
	check(C.mlx_stop_gradient(&res, a.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

// --- binary elementwise ------------------------------------------------------

pub fn (a Array) add(b Array) Array {
	res := new_result()
	check(C.mlx_add(&res, a.ctx, b.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) subtract(b Array) Array {
	res := new_result()
	check(C.mlx_subtract(&res, a.ctx, b.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) multiply(b Array) Array {
	res := new_result()
	check(C.mlx_multiply(&res, a.ctx, b.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) divide(b Array) Array {
	res := new_result()
	check(C.mlx_divide(&res, a.ctx, b.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) floor_divide(b Array) Array {
	res := new_result()
	check(C.mlx_floor_divide(&res, a.ctx, b.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) power(b Array) Array {
	res := new_result()
	check(C.mlx_power(&res, a.ctx, b.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) remainder(b Array) Array {
	res := new_result()
	check(C.mlx_remainder(&res, a.ctx, b.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) maximum(b Array) Array {
	res := new_result()
	check(C.mlx_maximum(&res, a.ctx, b.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) minimum(b Array) Array {
	res := new_result()
	check(C.mlx_minimum(&res, a.ctx, b.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) arctan2(b Array) Array {
	res := new_result()
	check(C.mlx_arctan2(&res, a.ctx, b.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) logaddexp(b Array) Array {
	res := new_result()
	check(C.mlx_logaddexp(&res, a.ctx, b.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) equal(b Array) Array {
	res := new_result()
	check(C.mlx_equal(&res, a.ctx, b.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) not_equal(b Array) Array {
	res := new_result()
	check(C.mlx_not_equal(&res, a.ctx, b.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) greater(b Array) Array {
	res := new_result()
	check(C.mlx_greater(&res, a.ctx, b.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) greater_equal(b Array) Array {
	res := new_result()
	check(C.mlx_greater_equal(&res, a.ctx, b.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) less(b Array) Array {
	res := new_result()
	check(C.mlx_less(&res, a.ctx, b.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) less_equal(b Array) Array {
	res := new_result()
	check(C.mlx_less_equal(&res, a.ctx, b.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) logical_and(b Array) Array {
	res := new_result()
	check(C.mlx_logical_and(&res, a.ctx, b.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) logical_or(b Array) Array {
	res := new_result()
	check(C.mlx_logical_or(&res, a.ctx, b.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) bitwise_and(b Array) Array {
	res := new_result()
	check(C.mlx_bitwise_and(&res, a.ctx, b.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) bitwise_or(b Array) Array {
	res := new_result()
	check(C.mlx_bitwise_or(&res, a.ctx, b.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) bitwise_xor(b Array) Array {
	res := new_result()
	check(C.mlx_bitwise_xor(&res, a.ctx, b.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) left_shift(b Array) Array {
	res := new_result()
	check(C.mlx_left_shift(&res, a.ctx, b.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) right_shift(b Array) Array {
	res := new_result()
	check(C.mlx_right_shift(&res, a.ctx, b.ctx, def_stream()))
	return Array{
		ctx: res
	}
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
	check(C.mlx_sum(&res, a.ctx, false, def_stream()))
	return Array{
		ctx: res
	}
}

// sum_axis reduces `a` along `axis`.
pub fn (a Array) sum_axis(axis int, keepdims bool) Array {
	res := new_result()
	check(C.mlx_sum_axis(&res, a.ctx, axis, keepdims, def_stream()))
	return Array{
		ctx: res
	}
}

// sum_axes reduces `a` along `axes`.
pub fn (a Array) sum_axes(axes []int, keepdims bool) Array {
	res := new_result()
	check(C.mlx_sum_axes(&res, a.ctx, axes.data, axes.len, keepdims, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) mean() Array {
	res := new_result()
	check(C.mlx_mean(&res, a.ctx, false, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) mean_axis(axis int, keepdims bool) Array {
	res := new_result()
	check(C.mlx_mean_axis(&res, a.ctx, axis, keepdims, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) mean_axes(axes []int, keepdims bool) Array {
	res := new_result()
	check(C.mlx_mean_axes(&res, a.ctx, axes.data, axes.len, keepdims, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) max() Array {
	res := new_result()
	check(C.mlx_max(&res, a.ctx, false, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) max_axis(axis int, keepdims bool) Array {
	res := new_result()
	check(C.mlx_max_axis(&res, a.ctx, axis, keepdims, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) max_axes(axes []int, keepdims bool) Array {
	res := new_result()
	check(C.mlx_max_axes(&res, a.ctx, axes.data, axes.len, keepdims, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) min() Array {
	res := new_result()
	check(C.mlx_min(&res, a.ctx, false, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) min_axis(axis int, keepdims bool) Array {
	res := new_result()
	check(C.mlx_min_axis(&res, a.ctx, axis, keepdims, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) min_axes(axes []int, keepdims bool) Array {
	res := new_result()
	check(C.mlx_min_axes(&res, a.ctx, axes.data, axes.len, keepdims, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) prod() Array {
	res := new_result()
	check(C.mlx_prod(&res, a.ctx, false, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) prod_axis(axis int, keepdims bool) Array {
	res := new_result()
	check(C.mlx_prod_axis(&res, a.ctx, axis, keepdims, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) all() Array {
	res := new_result()
	check(C.mlx_all(&res, a.ctx, false, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) any() Array {
	res := new_result()
	check(C.mlx_any(&res, a.ctx, false, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) std(ddof int) Array {
	res := new_result()
	check(C.mlx_std(&res, a.ctx, false, ddof, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) var(ddof int) Array {
	res := new_result()
	check(C.mlx_var(&res, a.ctx, false, ddof, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) logsumexp() Array {
	res := new_result()
	check(C.mlx_logsumexp(&res, a.ctx, false, def_stream()))
	return Array{
		ctx: res
	}
}

// --- cumulative & sorting ----------------------------------------------------

pub fn (a Array) cumsum(axis int, reverse bool, inclusive bool) Array {
	res := new_result()
	check(C.mlx_cumsum(&res, a.ctx, axis, reverse, inclusive, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) cumprod(axis int, reverse bool, inclusive bool) Array {
	res := new_result()
	check(C.mlx_cumprod(&res, a.ctx, axis, reverse, inclusive, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) cummax(axis int, reverse bool, inclusive bool) Array {
	res := new_result()
	check(C.mlx_cummax(&res, a.ctx, axis, reverse, inclusive, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) cummin(axis int, reverse bool, inclusive bool) Array {
	res := new_result()
	check(C.mlx_cummin(&res, a.ctx, axis, reverse, inclusive, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) argmax() Array {
	res := new_result()
	check(C.mlx_argmax(&res, a.ctx, false, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) argmax_axis(axis int, keepdims bool) Array {
	res := new_result()
	check(C.mlx_argmax_axis(&res, a.ctx, axis, keepdims, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) argmin() Array {
	res := new_result()
	check(C.mlx_argmin(&res, a.ctx, false, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) argmin_axis(axis int, keepdims bool) Array {
	res := new_result()
	check(C.mlx_argmin_axis(&res, a.ctx, axis, keepdims, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) sort() Array {
	res := new_result()
	check(C.mlx_sort(&res, a.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) sort_axis(axis int) Array {
	res := new_result()
	check(C.mlx_sort_axis(&res, a.ctx, axis, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) argsort() Array {
	res := new_result()
	check(C.mlx_argsort(&res, a.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) argsort_axis(axis int) Array {
	res := new_result()
	check(C.mlx_argsort_axis(&res, a.ctx, axis, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) topk(k int) Array {
	res := new_result()
	check(C.mlx_topk(&res, a.ctx, k, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) topk_axis(k int, axis int) Array {
	res := new_result()
	check(C.mlx_topk_axis(&res, a.ctx, k, axis, def_stream()))
	return Array{
		ctx: res
	}
}

// --- softmax -----------------------------------------------------------------

pub fn (a Array) softmax(precise bool) Array {
	res := new_result()
	check(C.mlx_softmax(&res, a.ctx, precise, def_stream()))
	return Array{
		ctx: res
	}
}

pub fn (a Array) softmax_axis(axis int, precise bool) Array {
	res := new_result()
	check(C.mlx_softmax_axis(&res, a.ctx, axis, precise, def_stream()))
	return Array{
		ctx: res
	}
}

// --- shape & manipulation ----------------------------------------------------

// reshape returns the array viewed with a new `shape`.
pub fn (a Array) reshape(shape []int) Array {
	res := new_result()
	check(C.mlx_reshape(&res, a.ctx, shape.data, shape.len, def_stream()))
	return Array{
		ctx: res
	}
}

// transpose reverses the array dimensions.
pub fn (a Array) transpose() Array {
	res := new_result()
	check(C.mlx_transpose(&res, a.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

// transpose_axes permutes dimensions according to `axes`.
pub fn (a Array) transpose_axes(axes []int) Array {
	res := new_result()
	check(C.mlx_transpose_axes(&res, a.ctx, axes.data, axes.len, def_stream()))
	return Array{
		ctx: res
	}
}

// swapaxes swaps two dimensions.
pub fn (a Array) swapaxes(axis1 int, axis2 int) Array {
	res := new_result()
	check(C.mlx_swapaxes(&res, a.ctx, axis1, axis2, def_stream()))
	return Array{
		ctx: res
	}
}

// moveaxis moves `source` to `destination`.
pub fn (a Array) moveaxis(source int, destination int) Array {
	res := new_result()
	check(C.mlx_moveaxis(&res, a.ctx, source, destination, def_stream()))
	return Array{
		ctx: res
	}
}

// expand_dims inserts a new dimension of size 1 at `axis`.
pub fn (a Array) expand_dims(axis int) Array {
	res := new_result()
	check(C.mlx_expand_dims(&res, a.ctx, axis, def_stream()))
	return Array{
		ctx: res
	}
}

// squeeze removes singleton dimensions.
pub fn (a Array) squeeze() Array {
	res := new_result()
	check(C.mlx_squeeze(&res, a.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

// squeeze_axis removes the singleton dimension at `axis`.
pub fn (a Array) squeeze_axis(axis int) Array {
	res := new_result()
	check(C.mlx_squeeze_axis(&res, a.ctx, axis, def_stream()))
	return Array{
		ctx: res
	}
}

// flatten collapses dimensions from `start_axis` to `end_axis`.
pub fn (a Array) flatten(start_axis int, end_axis int) Array {
	res := new_result()
	check(C.mlx_flatten(&res, a.ctx, start_axis, end_axis, def_stream()))
	return Array{
		ctx: res
	}
}

// broadcast_to broadcasts the array to `shape`.
pub fn (a Array) broadcast_to(shape []int) Array {
	res := new_result()
	check(C.mlx_broadcast_to(&res, a.ctx, shape.data, shape.len, def_stream()))
	return Array{
		ctx: res
	}
}

// tile repeats the array `reps` times along each dimension.
pub fn (a Array) tile(reps []int) Array {
	res := new_result()
	check(C.mlx_tile(&res, a.ctx, reps.data, reps.len, def_stream()))
	return Array{
		ctx: res
	}
}

// repeat repeats the array `repeats` times.
pub fn (a Array) repeat(repeats int) Array {
	res := new_result()
	check(C.mlx_repeat(&res, a.ctx, repeats, def_stream()))
	return Array{
		ctx: res
	}
}

// tril returns the lower triangle of the array.
pub fn (a Array) tril(k int) Array {
	res := new_result()
	check(C.mlx_tril(&res, a.ctx, k, def_stream()))
	return Array{
		ctx: res
	}
}

// triu returns the upper triangle of the array.
pub fn (a Array) triu(k int) Array {
	res := new_result()
	check(C.mlx_triu(&res, a.ctx, k, def_stream()))
	return Array{
		ctx: res
	}
}

// diag extracts the k-th diagonal or builds a diagonal matrix.
pub fn (a Array) diag(k int) Array {
	res := new_result()
	check(C.mlx_diag(&res, a.ctx, k, def_stream()))
	return Array{
		ctx: res
	}
}

// --- matmul & tensor products ------------------------------------------------

// matmul returns the matrix product of `a` and `b`.
pub fn (a Array) matmul(b Array) Array {
	res := new_result()
	check(C.mlx_matmul(&res, a.ctx, b.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

// inner returns the inner product of `a` and `b`.
pub fn (a Array) inner(b Array) Array {
	res := new_result()
	check(C.mlx_inner(&res, a.ctx, b.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

// outer returns the outer product of `a` and `b`.
pub fn (a Array) outer(b Array) Array {
	res := new_result()
	check(C.mlx_outer(&res, a.ctx, b.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

// kron returns the Kronecker product of `a` and `b`.
pub fn (a Array) kron(b Array) Array {
	res := new_result()
	check(C.mlx_kron(&res, a.ctx, b.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

// einsum evaluates an Einstein summation over the operands.
pub fn einsum(subscripts string, operands []Array) Array {
	res := new_result()
	vec := array_vector(operands)
	defer {
		vec.free()
	}
	check(C.mlx_einsum(&res, subscripts.str, vec.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

// tensordot contracts two arrays along given axes.
pub fn (a Array) tensordot(b Array, axes_a []int, axes_b []int) Array {
	res := new_result()
	check(C.mlx_tensordot(&res, a.ctx, b.ctx, axes_a.data, axes_a.len, axes_b.data, axes_b.len,
		def_stream()))
	return Array{
		ctx: res
	}
}

// --- indexing & updates ------------------------------------------------------

// take takes elements at `indices`.
pub fn (a Array) take(indices Array) Array {
	res := new_result()
	check(C.mlx_take(&res, a.ctx, indices.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

// take_axis takes elements along `axis` at `indices`.
pub fn (a Array) take_axis(indices Array, axis int) Array {
	res := new_result()
	check(C.mlx_take_axis(&res, a.ctx, indices.ctx, axis, def_stream()))
	return Array{
		ctx: res
	}
}

// take_along_axis takes elements along `axis` according to `indices`.
pub fn (a Array) take_along_axis(indices Array, axis int) Array {
	res := new_result()
	check(C.mlx_take_along_axis(&res, a.ctx, indices.ctx, axis, def_stream()))
	return Array{
		ctx: res
	}
}

// clip clamps the array between `a_min` and `a_max` (either may be an empty array).
pub fn (a Array) clip(a_min Array, a_max Array) Array {
	res := new_result()
	check(C.mlx_clip(&res, a.ctx, a_min.ctx, a_max.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

// where selects elements from `x` or `y` based on `condition`.
pub fn where(condition Array, x Array, y Array) Array {
	res := new_result()
	check(C.mlx_where(&res, condition.ctx, x.ctx, y.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

// --- combine / split ---------------------------------------------------------

// concatenate joins `arrays` along `axis`.
pub fn concatenate(arrays []Array, axis int) Array {
	res := new_result()
	vec := array_vector(arrays)
	defer {
		vec.free()
	}
	check(C.mlx_concatenate_axis(&res, vec.ctx, axis, def_stream()))
	return Array{
		ctx: res
	}
}

// stack joins `arrays` along a new `axis`.
pub fn stack(arrays []Array, axis int) Array {
	res := new_result()
	vec := array_vector(arrays)
	defer {
		vec.free()
	}
	check(C.mlx_stack_axis(&res, vec.ctx, axis, def_stream()))
	return Array{
		ctx: res
	}
}

// split splits the array into `num_splits` equal parts along `axis`.
pub fn (a Array) split(num_splits int, axis int) []Array {
	res := C.mlx_vector_array_new()
	check(C.mlx_split(&res, a.ctx, num_splits, axis, def_stream()))
	return array_vector_to_slice(res)
}
