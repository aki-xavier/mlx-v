module mlx

// Device represents an MLX execution device (a CPU or a specific GPU).
//
// The handle is reclaimed automatically by the GC once the last `Device`
// copy goes away; `free()` releases it deterministically.
pub struct Device {
mut:
	box &HandleBox = unsafe { nil }
}

// raw returns the underlying MLX handle (low level).
@[inline]
pub fn (d Device) raw() C.mlx_device {
	if isnil(d.box) {
		panic('mlx: Device is uninitialised (zero value); build it with device()/default_device() before using it')
	}
	return C.mlx_device{
		ctx: d.box.ctx
	}
}

// device returns a new device of the given `dtype` and `index`.
pub fn device(dtype DeviceType, index int) Device {
	return Device{
		box: wrap_handle(C.mlx_device_new_type(int(dtype), index).ctx, free_device_handle, true)
	}
}

// default_device returns the current default MLX device.
pub fn default_device() Device {
	dev := C.mlx_device_new()
	setup()
	begin_op()
	check(C.mlx_get_default_device(&dev))
	return Device{
		box: wrap_handle(dev.ctx, free_device_handle, true)
	}
}

// device_count returns the number of available devices of `dtype`.
pub fn device_count(dtype DeviceType) int {
	mut n := 0
	setup()
	begin_op()
	check(C.mlx_device_count(&n, int(dtype)))
	return n
}

// free releases the device (idempotent; optional with the GC).
pub fn (d &Device) free() {
	if isnil(d.box) {
		return
	}
	mut box := d.box
	box.release()
}

// str returns a human-readable description of the device.
pub fn (d Device) str() string {
	str_ := C.mlx_string_new()
	C.mlx_device_tostring(&str_, d.raw())
	res := cstr(C.mlx_string_data(str_))
	C.mlx_string_free(str_)
	return res
}

// index returns the device index.
pub fn (d Device) index() int {
	mut idx := 0
	setup()
	begin_op()
	check(C.mlx_device_get_index(&idx, d.raw()))
	return idx
}

// dtype returns the device type (cpu/gpu).
pub fn (d Device) dtype() DeviceType {
	mut t := 0
	setup()
	begin_op()
	check(C.mlx_device_get_type(&t, d.raw()))
	return unsafe { DeviceType(t) }
}

// available reports whether the device is available.
pub fn (d Device) available() bool {
	mut a := false
	setup()
	begin_op()
	check(C.mlx_device_is_available(&a, d.raw()))
	return a
}

// set_default makes this device the default MLX device.  This is process-wide
// and not thread-safe.  It also clears any stream override (see
// Stream.set_default) and syncs the force-CPU flag used by def_stream(), so
// use_cpu()/use_gpu() stay consistent.
pub fn (d Device) set_default() {
	setup()
	begin_op()
	check(C.mlx_set_default_device(d.raw()))
	C.mlx_v_clear_stream_override()
	if d.dtype() == .cpu {
		C.mlx_v_set_force_cpu(1)
	} else {
		C.mlx_v_set_force_cpu(0)
	}
}

// == reports whether two devices are the same.
pub fn (d Device) == (o Device) bool {
	return C.mlx_device_equal(d.raw(), o.raw())
}
