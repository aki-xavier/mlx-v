module mlx

// Device represents an MLX execution device (a CPU or a specific GPU).
pub struct Device {
mut:
	ctx   C.mlx_device
	freed bool
}

// device returns a new device of the given `dtype` and `index`.
pub fn device(dtype DeviceType, index int) Device {
	return Device{
		ctx: C.mlx_device_new_type(int(dtype), index)
	}
}

// default_device returns the current default MLX device.
pub fn default_device() Device {
	dev := C.mlx_device_new()
	setup()
	begin_op()
	check(C.mlx_get_default_device(&dev))
	return Device{
		ctx: dev
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

// free releases the device (idempotent).
pub fn (mut d Device) free() {
	if !d.freed {
		d.freed = true
		C.mlx_device_free(d.ctx)
	}
}

// str returns a human-readable description of the device.
pub fn (d Device) str() string {
	str_ := C.mlx_string_new()
	C.mlx_device_tostring(&str_, d.ctx)
	res := cstr(C.mlx_string_data(str_))
	C.mlx_string_free(str_)
	return res
}

// index returns the device index.
pub fn (d Device) index() int {
	mut idx := 0
	setup()
	begin_op()
	check(C.mlx_device_get_index(&idx, d.ctx))
	return idx
}

// dtype returns the device type (cpu/gpu).
pub fn (d Device) dtype() DeviceType {
	mut t := 0
	setup()
	begin_op()
	check(C.mlx_device_get_type(&t, d.ctx))
	return unsafe { DeviceType(t) }
}

// available reports whether the device is available.
pub fn (d Device) available() bool {
	mut a := false
	setup()
	begin_op()
	check(C.mlx_device_is_available(&a, d.ctx))
	return a
}

// set_default makes this device the default MLX device.
pub fn (d Device) set_default() {
	setup()
	begin_op()
	check(C.mlx_set_default_device(d.ctx))
}

// == reports whether two devices are the same.
pub fn (d Device) == (o Device) bool {
	return C.mlx_device_equal(d.ctx, o.ctx)
}
