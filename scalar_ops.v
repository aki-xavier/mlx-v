module mlx

// Thin scalar-broadcast helpers over `Array`.  V forbids mixed-type operator
// methods, so these free functions cover `array OP scalar` elementwise ops.
// They all operate in float32, matching the render-kernel dtype.

// fs returns a 0-d float32 scalar array (broadcasts in elementwise ops).
@[inline]
pub fn fs(v f64) Array {
	return f32_scalar(f32(v))
}

// arr3 builds a (3,) float32 array from three f64 values.
pub fn arr3(a f64, b f64, c f64) Array {
	return array_f32([f32(a), f32(b), f32(c)], [3])
}

// arr3v builds a (3,) float32 array from a [3]f64.
pub fn arr3v(v [3]f64) Array {
	return array_f32([f32(v[0]), f32(v[1]), f32(v[2])], [3])
}

@[inline]
pub fn s_add(a Array, v f64) Array {
	return a.add(fs(v))
}

@[inline]
pub fn s_sub(a Array, v f64) Array {
	return a.subtract(fs(v))
}

@[inline]
pub fn s_mul(a Array, v f64) Array {
	return a.multiply(fs(v))
}

@[inline]
pub fn s_div(a Array, v f64) Array {
	return a.divide(fs(v))
}

@[inline]
pub fn s_pow(a Array, v f64) Array {
	return a.power(fs(v))
}

@[inline]
pub fn s_rsub(a Array, v f64) Array {
	return fs(v).subtract(a)
}

@[inline]
pub fn s_rdiv(a Array, v f64) Array {
	return fs(v).divide(a)
}

@[inline]
pub fn s_lt(a Array, v f64) Array {
	return a.less(fs(v))
}

@[inline]
pub fn s_le(a Array, v f64) Array {
	return a.less_equal(fs(v))
}

@[inline]
pub fn s_gt(a Array, v f64) Array {
	return a.greater(fs(v))
}

@[inline]
pub fn s_ge(a Array, v f64) Array {
	return a.greater_equal(fs(v))
}

@[inline]
pub fn s_eq(a Array, v f64) Array {
	return a.equal(fs(v))
}

@[inline]
pub fn s_max(a Array, v f64) Array {
	return a.maximum(fs(v))
}

@[inline]
pub fn s_min(a Array, v f64) Array {
	return a.minimum(fs(v))
}

@[inline]
pub fn s_clip(a Array, lo f64, hi f64) Array {
	return a.clip(fs(lo), fs(hi))
}
