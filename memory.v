module mlx

// memory.v — memory statistics and limits (bytes).

// active_memory returns the currently active memory in bytes.
pub fn active_memory() usize {
	setup()
	begin_op()
	mut res := usize(0)
	check(C.mlx_get_active_memory(&res))
	return res
}

// peak_memory returns the peak memory usage in bytes.
pub fn peak_memory() usize {
	setup()
	begin_op()
	mut res := usize(0)
	check(C.mlx_get_peak_memory(&res))
	return res
}

// cache_memory returns the current cache memory in bytes.
pub fn cache_memory() usize {
	setup()
	begin_op()
	mut res := usize(0)
	check(C.mlx_get_cache_memory(&res))
	return res
}

// memory_limit returns the current memory limit in bytes.
pub fn memory_limit() usize {
	setup()
	begin_op()
	mut res := usize(0)
	check(C.mlx_get_memory_limit(&res))
	return res
}

// set_memory_limit sets the memory limit and returns the previous limit.
pub fn set_memory_limit(limit usize) usize {
	setup()
	begin_op()
	mut res := usize(0)
	check(C.mlx_set_memory_limit(&res, limit))
	return res
}

// set_cache_limit sets the cache limit and returns the previous limit.
pub fn set_cache_limit(limit usize) usize {
	setup()
	begin_op()
	mut res := usize(0)
	check(C.mlx_set_cache_limit(&res, limit))
	return res
}

// set_wired_limit sets the wired memory limit and returns the previous limit.
pub fn set_wired_limit(limit usize) usize {
	setup()
	begin_op()
	mut res := usize(0)
	check(C.mlx_set_wired_limit(&res, limit))
	return res
}

// reset_peak_memory resets the peak memory counter.
pub fn reset_peak_memory() {
	setup()
	begin_op()
	check(C.mlx_reset_peak_memory())
}

// clear_cache clears the cache.
pub fn clear_cache() {
	setup()
	begin_op()
	check(C.mlx_clear_cache())
}
