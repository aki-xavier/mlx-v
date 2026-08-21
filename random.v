module mlx

// random.v — pseudo-random number generation.

// no_key returns an empty array usable as the `key` argument of the random
// functions below (an empty key means "use the default PRNG state").
pub fn no_key() Array {
	return empty()
}

// random_key returns a new PRNG key from a 64-bit seed.
pub fn random_key(seed u64) Array {
	setup()
	begin_op()
	ctx := C.mlx_array_new()
	check_res(C.mlx_random_key(&ctx, seed), ctx)
	return wrap_array(ctx)
}

// random_seed seeds the default PRNG.
pub fn random_seed(seed u64) {
	setup()
	begin_op()
	check(C.mlx_random_seed(seed))
}

// random_split splits `key` into two new keys.
pub fn random_split(key Array) (Array, Array) {
	setup()
	begin_op()
	r0 := C.mlx_array_new()
	r1 := C.mlx_array_new()
	check_res2(C.mlx_random_split(&r0, &r1, key.raw(), def_stream()), r0, r1)
	return wrap_array(r0), wrap_array(r1)
}

// random_split_n splits `key` into `num` new keys, returning a single (num, 2)
// uint32 array (matching mlx's mx.random.split(key, num)).
pub fn random_split_n(key Array, num int) Array {
	setup()
	begin_op()
	res := C.mlx_array_new()
	check_res(C.mlx_random_split_num(&res, key.raw(), num, def_stream()), res)
	return wrap_array(res)
}

// random_normal samples from a normal distribution with mean `loc`, std `scale`.
pub fn random_normal(shape []int, dtype Dtype, loc f32, scale f32, key Array) Array {
	setup()
	begin_op()
	res := C.mlx_array_new()
	check_res(C.mlx_random_normal(&res, shape.data, shape.len, int(dtype), loc, scale, key.raw(),
		def_stream()), res)
	return wrap_array(res)
}

// random_uniform samples uniformly from [low, high).
pub fn random_uniform(low Array, high Array, shape []int, dtype Dtype, key Array) Array {
	setup()
	begin_op()
	res := C.mlx_array_new()
	check_res(C.mlx_random_uniform(&res, low.raw(), high.raw(), shape.data, shape.len, int(dtype),
		key.raw(), def_stream()), res)
	return wrap_array(res)
}

// random_randint samples integers uniformly from [low, high).
pub fn random_randint(low Array, high Array, shape []int, dtype Dtype, key Array) Array {
	setup()
	begin_op()
	res := C.mlx_array_new()
	check_res(C.mlx_random_randint(&res, low.raw(), high.raw(), shape.data, shape.len, int(dtype),
		key.raw(), def_stream()), res)
	return wrap_array(res)
}

// random_bernoulli samples Bernoulli variables with parameter `p`.
pub fn random_bernoulli(p Array, shape []int, key Array) Array {
	setup()
	begin_op()
	res := C.mlx_array_new()
	check_res(C.mlx_random_bernoulli(&res, p.raw(), shape.data, shape.len, key.raw(), def_stream()),
		res)
	return wrap_array(res)
}

// random_gumbel samples from the Gumbel distribution.
pub fn random_gumbel(shape []int, dtype Dtype, key Array) Array {
	setup()
	begin_op()
	res := C.mlx_array_new()
	check_res(C.mlx_random_gumbel(&res, shape.data, shape.len, int(dtype), key.raw(), def_stream()),
		res)
	return wrap_array(res)
}

// random_laplace samples from the Laplace distribution.
pub fn random_laplace(shape []int, dtype Dtype, loc f32, scale f32, key Array) Array {
	setup()
	begin_op()
	res := C.mlx_array_new()
	check_res(C.mlx_random_laplace(&res, shape.data, shape.len, int(dtype), loc, scale, key.raw(),
		def_stream()), res)
	return wrap_array(res)
}

// random_truncated_normal samples from a truncated normal distribution.
pub fn random_truncated_normal(lower Array, upper Array, shape []int, dtype Dtype, key Array) Array {
	setup()
	begin_op()
	res := C.mlx_array_new()
	check_res(C.mlx_random_truncated_normal(&res, lower.raw(), upper.raw(), shape.data, shape.len,
		int(dtype), key.raw(), def_stream()), res)
	return wrap_array(res)
}

// random_categorical samples categories from `logits` along `axis`.
pub fn random_categorical(logits Array, axis int, key Array) Array {
	setup()
	begin_op()
	res := C.mlx_array_new()
	check_res(C.mlx_random_categorical(&res, logits.raw(), axis, key.raw(), def_stream()), res)
	return wrap_array(res)
}

// random_permutation returns a random permutation of `x` along `axis`.
pub fn random_permutation(x Array, axis int, key Array) Array {
	setup()
	begin_op()
	res := C.mlx_array_new()
	check_res(C.mlx_random_permutation(&res, x.raw(), axis, key.raw(), def_stream()), res)
	return wrap_array(res)
}

// split_keys splits a PRNG key from `seed` into `num` independent keys
// (mx.random.split semantics).
pub fn split_keys(seed u64, num int) []Array {
	keys := random_split_n(random_key(seed), num)
	mut out := []Array{len: num}
	for i in 0 .. num {
		out[i] = keys.take_axis(sel1(i), 0).squeeze_axis(0)
	}
	return out
}
