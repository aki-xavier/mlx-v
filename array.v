module mlx

// ArrayBox holds a raw MLX handle on the GC heap.  A Boehm finalizer attached
// to it releases the handle when the box becomes unreachable.
struct ArrayBox {
mut:
	ctx   C.mlx_array
	freed bool
}

// Array is an N-dimensional MLX array (a lazy tensor).
//
// The underlying MLX handle is reclaimed automatically by the GC once the last
// `Array` referencing it goes away.  Call `free()` to release it deterministically.
//
// `box` defaults to nil so that `Array{}` is a valid zero value (V requires
// reference fields to be initialised otherwise); a nil-box Array must be
// assigned a real array before any operation reads `raw()`.
pub struct Array {
mut:
	box &ArrayBox = unsafe { nil }
}

// raw returns the underlying MLX handle (low level).
@[inline]
pub fn (a Array) raw() C.mlx_array {
	if isnil(a.box) {
		panic('mlx: Array is uninitialised (zero value); build it with array_f32/zeros/empty()/… before using it')
	}
	return a.box.ctx
}

// array_finalizer releases the MLX handle when its box is garbage-collected.
fn array_finalizer(obj voidptr, _cd voidptr) {
	mut box := unsafe { &ArrayBox(obj) }
	if !box.freed {
		box.freed = true
		C.mlx_array_free(box.ctx)
		C.mlx_v_note_box_free()
	}
}

// wrap_array boxes a raw MLX handle and registers its finalizer.
fn wrap_array(ctx C.mlx_array) Array {
	// Allocate the box from the Boehm GC heap explicitly (GC_MALLOC), rather
	// than via V's `&ArrayBox{}`, so the GC tracks it and runs the finalizer.
	mut box := unsafe { &ArrayBox(C.mlx_v_gc_malloc(sizeof(ArrayBox))) }
	box.ctx = ctx
	box.freed = false
	C.mlx_v_note_box_alloc()
	register_finalizer(box, array_finalizer)
	return Array{
		box: box
	}
}

// empty returns a new empty (uninitialised) array.
pub fn empty() Array {
	return wrap_array(C.mlx_array_new())
}

// bool_scalar returns a 0-d boolean array.
pub fn bool_scalar(v bool) Array {
	setup()
	begin_op()
	ctx := C.mlx_array_new_bool(v)
	fail_on_error()
	return wrap_array(ctx)
}

// int_scalar returns a 0-d int array.
pub fn int_scalar(v int) Array {
	setup()
	begin_op()
	ctx := C.mlx_array_new_int(v)
	fail_on_error()
	return wrap_array(ctx)
}

// f32_scalar returns a 0-d float32 array.
pub fn f32_scalar(v f32) Array {
	setup()
	begin_op()
	ctx := C.mlx_array_new_float32(v)
	fail_on_error()
	return wrap_array(ctx)
}

// f64_scalar returns a 0-d float64 array.
pub fn f64_scalar(v f64) Array {
	setup()
	begin_op()
	ctx := C.mlx_array_new_float64(v)
	fail_on_error()
	return wrap_array(ctx)
}

// complex_scalar returns a 0-d complex64 array re + i·im.
pub fn complex_scalar(re f32, im f32) Array {
	setup()
	begin_op()
	ctx := C.mlx_array_new_complex(re, im)
	fail_on_error()
	return wrap_array(ctx)
}

// array builds an array from `data` and `shape` using the generic dtype `dtype`.
pub fn array_with[T](data []T, shape []int, dtype Dtype) Array {
	setup()
	begin_op()
	ctx := C.mlx_array_new_data(voidptr(data.data), shape.data, shape.len, int(dtype))
	fail_on_error()
	return wrap_array(ctx)
}

// array_f32 builds a float32 array.
pub fn array_f32(data []f32, shape []int) Array {
	return array_with(data, shape, .float32)
}

// array_f64 builds a float64 array.
pub fn array_f64(data []f64, shape []int) Array {
	return array_with(data, shape, .float64)
}

// array_i32 builds an int32 array.
pub fn array_i32(data []i32, shape []int) Array {
	return array_with(data, shape, .int32)
}

// array_i64 builds an int64 array.
pub fn array_i64(data []i64, shape []int) Array {
	return array_with(data, shape, .int64)
}

// array_u32 builds a uint32 array.
pub fn array_u32(data []u32, shape []int) Array {
	return array_with(data, shape, .uint32)
}

// array_u64 builds a uint64 array.
pub fn array_u64(data []u64, shape []int) Array {
	return array_with(data, shape, .uint64)
}

// array_bool builds a boolean array.
pub fn array_bool(data []bool, shape []int) Array {
	return array_with(data, shape, .bool_)
}

// arr32 builds a float32 array from f64 literals (avoiding f32() casts everywhere).
pub fn arr32(vals []f64, shape []int) Array {
	mut f := []f32{len: vals.len}
	for i, v in vals {
		f[i] = f32(v)
	}
	return array_f32(f, shape)
}

// sel1 returns a length-1 int32 index array selecting element `n` along an axis.
pub fn sel1(n int) Array {
	return array_i32([i32(n)], [1])
}

// complex_from builds a complex64 array re + i·im from two real arrays.
pub fn complex_from(re Array, im Array) Array {
	re_c := re.astype(.complex64)
	im_c := im.astype(.complex64)
	i := complex_scalar(0.0, 1.0)
	return re_c.add(i.multiply(im_c))
}

// fail_on_error panics when the last constructor call recorded an error.
@[inline]
fn fail_on_error() {
	msg := cstr(C.mlx_v_get_error())
	if msg.len > 0 {
		panic('MLX error: ${msg}')
	}
}

// free releases the array deterministically (optional with the GC).
pub fn (a &Array) free() {
	mut box := a.box
	if !box.freed {
		box.freed = true
		C.mlx_array_free(box.ctx)
		C.mlx_v_note_box_free()
	}
}

// clone returns a new array sharing the same value/graph as `a`.
pub fn (a Array) clone() Array {
	res := C.mlx_array_new()
	C.mlx_array_set(&res, a.raw())
	return wrap_array(res)
}

// str returns a human-readable description of the array.
pub fn (a Array) str() string {
	str_ := C.mlx_string_new()
	C.mlx_array_tostring(&str_, a.raw())
	res := cstr(C.mlx_string_data(str_))
	C.mlx_string_free(str_)
	return res
}

// eval forces evaluation of the array.
pub fn (a Array) eval() {
	setup()
	begin_op()
	check(C.mlx_array_eval(a.raw()))
}

// itemsize returns the size of one element in bytes.
pub fn (a Array) itemsize() usize {
	return C.mlx_array_itemsize(a.raw())
}

// size returns the number of elements.
pub fn (a Array) size() usize {
	return C.mlx_array_size(a.raw())
}

// nbytes returns the total number of bytes.
pub fn (a Array) nbytes() usize {
	return C.mlx_array_nbytes(a.raw())
}

// ndim returns the number of dimensions.
pub fn (a Array) ndim() int {
	return int(C.mlx_array_ndim(a.raw()))
}

// shape returns the array shape.
pub fn (a Array) shape() []int {
	n := int(C.mlx_array_ndim(a.raw()))
	ptr := C.mlx_array_shape(a.raw())
	mut out := []int{len: n}
	unsafe {
		for i in 0 .. n {
			out[i] = ptr[i]
		}
	}
	return out
}

// strides returns the array strides (in elements).
pub fn (a Array) strides() []usize {
	n := int(C.mlx_array_ndim(a.raw()))
	ptr := C.mlx_array_strides(a.raw())
	mut out := []usize{len: n}
	unsafe {
		for i in 0 .. n {
			out[i] = ptr[i]
		}
	}
	return out
}

// dim returns the size of the array along dimension `dim`.
pub fn (a Array) dim(dim int) int {
	return C.mlx_array_dim(a.raw(), dim)
}

// dtype returns the array element type.
pub fn (a Array) dtype() Dtype {
	return unsafe { Dtype(C.mlx_array_dtype(a.raw())) }
}

// item_bool reads a scalar boolean array.
pub fn (a Array) item_bool() bool {
	a.eval()
	mut res := false
	check(C.mlx_array_item_bool(&res, a.raw()))
	return res
}

// item_i32 reads a scalar int32 array.
pub fn (a Array) item_i32() int {
	a.eval()
	mut res := 0
	check(C.mlx_array_item_int32(&res, a.raw()))
	return res
}

// item_i64 reads a scalar int64 array.
pub fn (a Array) item_i64() i64 {
	a.eval()
	mut res := i64(0)
	check(C.mlx_array_item_int64(&res, a.raw()))
	return res
}

// item_f32 reads a scalar float32 array.
pub fn (a Array) item_f32() f32 {
	a.eval()
	mut res := f32(0)
	check(C.mlx_array_item_float32(&res, a.raw()))
	return res
}

// item_f64 reads a scalar float64 array.
pub fn (a Array) item_f64() f64 {
	a.eval()
	mut res := f64(0)
	check(C.mlx_array_item_float64(&res, a.raw()))
	return res
}

// data_f32 copies the array contents (row-major) into a V `[]f32`.
pub fn (a Array) data_f32() []f32 {
	c := a.contiguous(false)
	defer {
		c.free()
	}
	c.eval()
	ptr := C.mlx_array_data_float32(c.raw())
	n := int(C.mlx_array_size(c.raw()))
	mut out := []f32{len: n}
	unsafe {
		for i in 0 .. n {
			out[i] = ptr[i]
		}
	}
	return out
}

// data_f64 copies the array contents (row-major) into a V `[]f64`.
pub fn (a Array) data_f64() []f64 {
	c := a.contiguous(false)
	defer {
		c.free()
	}
	c.eval()
	ptr := C.mlx_array_data_float64(c.raw())
	n := int(C.mlx_array_size(c.raw()))
	mut out := []f64{len: n}
	unsafe {
		for i in 0 .. n {
			out[i] = ptr[i]
		}
	}
	return out
}

// data_i32 copies the array contents (row-major) into a V `[]int`.
pub fn (a Array) data_i32() []int {
	c := a.contiguous(false)
	defer {
		c.free()
	}
	c.eval()
	ptr := C.mlx_array_data_int32(c.raw())
	n := int(C.mlx_array_size(c.raw()))
	mut out := []int{len: n}
	unsafe {
		for i in 0 .. n {
			out[i] = ptr[i]
		}
	}
	return out
}

// data_i64 copies the array contents (row-major) into a V `[]i64`.
pub fn (a Array) data_i64() []i64 {
	c := a.contiguous(false)
	defer {
		c.free()
	}
	c.eval()
	ptr := C.mlx_array_data_int64(c.raw())
	n := int(C.mlx_array_size(c.raw()))
	mut out := []i64{len: n}
	unsafe {
		for i in 0 .. n {
			out[i] = ptr[i]
		}
	}
	return out
}

// data_bool copies the array contents (row-major) into a V `[]bool`.
pub fn (a Array) data_bool() []bool {
	c := a.contiguous(false)
	defer {
		c.free()
	}
	c.eval()
	ptr := C.mlx_array_data_bool(c.raw())
	n := int(C.mlx_array_size(c.raw()))
	mut out := []bool{len: n}
	unsafe {
		for i in 0 .. n {
			out[i] = ptr[i]
		}
	}
	return out
}

// is_available reports whether the array value is already computed.
pub fn (a Array) is_available() bool {
	mut res := false
	C._mlx_array_is_available(&res, a.raw())
	return res
}

// wait blocks until the array value is available.
pub fn (a Array) wait() {
	setup()
	begin_op()
	check(C._mlx_array_wait(a.raw()))
}

// item_complex64 reads a scalar complex64 array.
pub fn (a Array) item_complex64() Complex64 {
	a.eval()
	mut res := Complex64{}
	check(C.mlx_array_item_complex64(voidptr(&res), a.raw()))
	return res
}

// data_complex64 copies the array contents into a V `[]Complex64`.
pub fn (a Array) data_complex64() []Complex64 {
	c := a.contiguous(false)
	defer {
		c.free()
	}
	c.eval()
	ptr := C.mlx_array_data_complex64(c.raw())
	n := int(C.mlx_array_size(c.raw()))
	mut out := []Complex64{len: n}
	unsafe {
		p := &Complex64(ptr)
		for i in 0 .. n {
			out[i] = p[i]
		}
	}
	return out
}

// item_f16 reads a scalar float16 array as an f32.
pub fn (a Array) item_f16() f32 {
	a.eval()
	mut bits := u16(0)
	check(C.mlx_array_item_float16(voidptr(&bits), a.raw()))
	return C.mlx_v_f16_to_f32(bits)
}

// data_f16 copies a float16 array into a V `[]f32`.
pub fn (a Array) data_f16() []f32 {
	c := a.contiguous(false)
	defer {
		c.free()
	}
	c.eval()
	ptr := C.mlx_array_data_float16(c.raw())
	n := int(C.mlx_array_size(c.raw()))
	mut out := []f32{len: n}
	unsafe {
		p := &u16(ptr)
		for i in 0 .. n {
			out[i] = C.mlx_v_f16_to_f32(p[i])
		}
	}
	return out
}

// item_bf16 reads a scalar bfloat16 array as an f32.
pub fn (a Array) item_bf16() f32 {
	a.eval()
	mut bits := u16(0)
	check(C.mlx_array_item_bfloat16(voidptr(&bits), a.raw()))
	return C.mlx_v_bf16_to_f32(bits)
}

// data_bf16 copies a bfloat16 array into a V `[]f32`.
pub fn (a Array) data_bf16() []f32 {
	c := a.contiguous(false)
	defer {
		c.free()
	}
	c.eval()
	ptr := C.mlx_array_data_bfloat16(c.raw())
	n := int(C.mlx_array_size(c.raw()))
	mut out := []f32{len: n}
	unsafe {
		p := &u16(ptr)
		for i in 0 .. n {
			out[i] = C.mlx_v_bf16_to_f32(p[i])
		}
	}
	return out
}
