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
	check(C.mlx_random_key(&ctx, seed))
	return Array{
		ctx: ctx
	}
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
	check(C.mlx_random_split(&r0, &r1, key.ctx, def_stream()))
	return Array{
		ctx: r0
	}, Array{
		ctx: r1
	}
}

// random_normal samples from a normal distribution with mean `loc`, std `scale`.
pub fn random_normal(shape []int, dtype Dtype, loc f32, scale f32, key Array) Array {
	setup()
	begin_op()
	res := C.mlx_array_new()
	check(C.mlx_random_normal(&res, shape.data, shape.len, int(dtype), loc, scale, key.ctx,
		def_stream()))
	return Array{
		ctx: res
	}
}

// random_uniform samples uniformly from [low, high).
pub fn random_uniform(low Array, high Array, shape []int, dtype Dtype, key Array) Array {
	setup()
	begin_op()
	res := C.mlx_array_new()
	check(C.mlx_random_uniform(&res, low.ctx, high.ctx, shape.data, shape.len, int(dtype), key.ctx,
		def_stream()))
	return Array{
		ctx: res
	}
}

// random_randint samples integers uniformly from [low, high).
pub fn random_randint(low Array, high Array, shape []int, dtype Dtype, key Array) Array {
	setup()
	begin_op()
	res := C.mlx_array_new()
	check(C.mlx_random_randint(&res, low.ctx, high.ctx, shape.data, shape.len, int(dtype), key.ctx,
		def_stream()))
	return Array{
		ctx: res
	}
}

// random_bernoulli samples Bernoulli variables with parameter `p`.
pub fn random_bernoulli(p Array, shape []int, key Array) Array {
	setup()
	begin_op()
	res := C.mlx_array_new()
	check(C.mlx_random_bernoulli(&res, p.ctx, shape.data, shape.len, key.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

// random_gumbel samples from the Gumbel distribution.
pub fn random_gumbel(shape []int, dtype Dtype, key Array) Array {
	setup()
	begin_op()
	res := C.mlx_array_new()
	check(C.mlx_random_gumbel(&res, shape.data, shape.len, int(dtype), key.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

// random_laplace samples from the Laplace distribution.
pub fn random_laplace(shape []int, dtype Dtype, loc f32, scale f32, key Array) Array {
	setup()
	begin_op()
	res := C.mlx_array_new()
	check(C.mlx_random_laplace(&res, shape.data, shape.len, int(dtype), loc, scale, key.ctx,
		def_stream()))
	return Array{
		ctx: res
	}
}

// random_truncated_normal samples from a truncated normal distribution.
pub fn random_truncated_normal(lower Array, upper Array, shape []int, dtype Dtype, key Array) Array {
	setup()
	begin_op()
	res := C.mlx_array_new()
	check(C.mlx_random_truncated_normal(&res, lower.ctx, upper.ctx, shape.data, shape.len,
		int(dtype), key.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

// random_categorical samples categories from `logits` along `axis`.
pub fn random_categorical(logits Array, axis int, key Array) Array {
	setup()
	begin_op()
	res := C.mlx_array_new()
	check(C.mlx_random_categorical(&res, logits.ctx, axis, key.ctx, def_stream()))
	return Array{
		ctx: res
	}
}

// random_permutation returns a random permutation of `x` along `axis`.
pub fn random_permutation(x Array, axis int, key Array) Array {
	setup()
	begin_op()
	res := C.mlx_array_new()
	check(C.mlx_random_permutation(&res, x.ctx, axis, key.ctx, def_stream()))
	return Array{
		ctx: res
	}
}
