module mlx

// Stream is an MLX execution stream (queue) on a given device.
pub struct Stream {
mut:
	ctx C.mlx_stream
}

// cpu_stream returns the default CPU stream.
pub fn cpu_stream() Stream {
	return Stream{
		ctx: C.mlx_default_cpu_stream_new()
	}
}

// gpu_stream returns the default GPU (Metal/CUDA) stream.
pub fn gpu_stream() Stream {
	return Stream{
		ctx: C.mlx_default_gpu_stream_new()
	}
}

// default_stream returns the default GPU stream when a GPU is available, and
// the default CPU stream otherwise.
pub fn default_stream() Stream {
	if gpu_available() {
		return gpu_stream()
	}
	return cpu_stream()
}

// new_stream returns an empty (invalid) stream.
pub fn new_stream() Stream {
	return Stream{
		ctx: C.mlx_stream_new()
	}
}

// on_device returns a stream for `dev`.
pub fn on_device(dev Device) Stream {
	return Stream{
		ctx: C.mlx_stream_new_device(dev.ctx)
	}
}

// free releases the stream.
pub fn (s &Stream) free() {
	C.mlx_stream_free(s.ctx)
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
