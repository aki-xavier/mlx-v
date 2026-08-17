module mlx

// Stream is an MLX execution stream (queue) on a given device.
pub struct Stream {
mut:
	ctx   C.mlx_stream
	owned bool // false for the shared cached default streams (free() is a no-op)
	freed bool
}

// cpu_stream returns the default CPU stream.  It returns the shared cached
// wrapper, so `free()` on it is a no-op (do not rely on it to release).
pub fn cpu_stream() Stream {
	return Stream{
		ctx: C.mlx_v_cached_cpu_stream()
	}
}

// gpu_stream returns the default GPU (Metal/CUDA) stream (shared cached wrapper).
pub fn gpu_stream() Stream {
	return Stream{
		ctx: C.mlx_v_cached_gpu_stream()
	}
}

// default_stream returns the default GPU stream when a GPU is available (and
// `use_cpu()` has not forced the CPU), and the default CPU stream otherwise.
pub fn default_stream() Stream {
	if C.mlx_v_get_force_cpu() != 0 || !gpu_available() {
		return cpu_stream()
	}
	return gpu_stream()
}

// new_stream returns an empty (invalid) stream (owned; free it).
pub fn new_stream() Stream {
	return Stream{
		ctx:   C.mlx_stream_new()
		owned: true
	}
}

// on_device returns a stream for `dev` (owned; free it).
pub fn on_device(dev Device) Stream {
	return Stream{
		ctx:   C.mlx_stream_new_device(dev.ctx)
		owned: true
	}
}

// free releases the stream (idempotent; a no-op for the cached default streams).
pub fn (mut s Stream) free() {
	if s.owned && !s.freed {
		s.freed = true
		C.mlx_stream_free(s.ctx)
	}
}

// str returns a human-readable description of the stream.
pub fn (s Stream) str() string {
	str_ := C.mlx_string_new()
	C.mlx_stream_tostring(&str_, s.ctx)
	res := cstr(C.mlx_string_data(str_))
	C.mlx_string_free(str_)
	return res
}

// device returns the device this stream runs on.
pub fn (s Stream) device() Device {
	dev := C.mlx_device_new()
	C.mlx_stream_get_device(&dev, s.ctx)
	return Device{
		ctx: dev
	}
}

// index returns the stream index.
pub fn (s Stream) index() int {
	mut idx := 0
	C.mlx_stream_get_index(&idx, s.ctx)
	return idx
}

// synchronize blocks until all work on this stream is complete.
pub fn (s Stream) synchronize() {
	setup()
	begin_op()
	check(C.mlx_synchronize(s.ctx))
}

// set_default makes this stream the default.
pub fn (s Stream) set_default() {
	setup()
	begin_op()
	check(C.mlx_set_default_stream(s.ctx))
}
