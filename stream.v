module mlx

// Stream is an MLX execution stream (queue) on a given device.
//
// The handle is reclaimed automatically by the GC once the last `Stream` copy
// goes away; `free()` releases it deterministically.  The shared cached
// default streams (cpu_stream()/gpu_stream()) are not owned, so `free()` on
// them is a no-op.
pub struct Stream {
mut:
	box &HandleBox = unsafe { nil }
}

// raw returns the underlying MLX handle (low level).
@[inline]
pub fn (s Stream) raw() C.mlx_stream {
	if isnil(s.box) {
		panic('mlx: Stream is uninitialised (zero value); build it with cpu_stream()/gpu_stream()/on_device() before using it')
	}
	return C.mlx_stream{
		ctx: s.box.ctx
	}
}

// cpu_stream returns the default CPU stream.  It wraps the shared cached
// handle, so `free()` on it is a no-op (do not rely on it to release).
pub fn cpu_stream() Stream {
	return Stream{
		box: wrap_handle(C.mlx_v_cached_cpu_stream().ctx, free_stream_handle, false)
	}
}

// gpu_stream returns the default GPU (Metal/CUDA) stream (shared cached handle).
pub fn gpu_stream() Stream {
	return Stream{
		box: wrap_handle(C.mlx_v_cached_gpu_stream().ctx, free_stream_handle, false)
	}
}

// default_stream returns the stream subsequent ops run on: the stream override
// installed by Stream.set_default() when one is active, otherwise the default
// GPU stream when a GPU is available (and `use_cpu()` has not forced the CPU),
// or the default CPU stream.
pub fn default_stream() Stream {
	return Stream{
		box: wrap_handle(C.mlx_v_stream_for_ops().ctx, free_stream_handle, false)
	}
}

// new_stream returns an empty (invalid) stream (owned; free it).
pub fn new_stream() Stream {
	return Stream{
		box: wrap_handle(C.mlx_stream_new().ctx, free_stream_handle, true)
	}
}

// on_device returns a stream for `dev` (owned; free it).
pub fn on_device(dev Device) Stream {
	return Stream{
		box: wrap_handle(C.mlx_stream_new_device(dev.raw()).ctx, free_stream_handle, true)
	}
}

// free releases the stream (idempotent; a no-op for the shared cached default
// streams and for a stream installed as the process-wide override).
pub fn (s &Stream) free() {
	if isnil(s.box) {
		return
	}
	mut box := s.box
	box.release()
}

// str returns a human-readable description of the stream.
pub fn (s Stream) str() string {
	str_ := C.mlx_string_new()
	C.mlx_stream_tostring(&str_, s.raw())
	res := cstr(C.mlx_string_data(str_))
	C.mlx_string_free(str_)
	return res
}

// device returns the device this stream runs on.
pub fn (s Stream) device() Device {
	dev := C.mlx_device_new()
	C.mlx_stream_get_device(&dev, s.raw())
	return Device{
		box: wrap_handle(dev.ctx, free_device_handle, true)
	}
}

// index returns the stream index.
pub fn (s Stream) index() int {
	mut idx := 0
	C.mlx_stream_get_index(&idx, s.raw())
	return idx
}

// synchronize blocks until all work on this stream is complete.
pub fn (s Stream) synchronize() {
	setup()
	begin_op()
	check(C.mlx_synchronize(s.raw()))
}

// set_default makes this stream the default: subsequent ops run on it.  This
// is process-wide and not thread-safe.
//
// The process-wide override holds the handle for the rest of the process
// (like the cached default streams), so this Stream is disowned: `free()` and
// the GC finalizer no longer release it.
pub fn (s Stream) set_default() {
	setup()
	begin_op()
	check(C.mlx_set_default_stream(s.raw()))
	C.mlx_v_set_stream_override(s.raw())
	mut box := s.box
	box.owned = false
}
