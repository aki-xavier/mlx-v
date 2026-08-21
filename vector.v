module mlx

// vector.v — wrappers for MLX vector-of-array / int / string containers.
// All handles are GC-managed; `free()` releases them deterministically.

// VectorArray is a growable list of MLX arrays.
pub struct VectorArray {
mut:
	box &HandleBox = unsafe { nil }
}

// raw returns the underlying MLX handle (low level).
@[inline]
pub fn (v VectorArray) raw() C.mlx_vector_array {
	if isnil(v.box) {
		panic('mlx: VectorArray is uninitialised (zero value); build it with new_vector_array()/array_vector() before using it')
	}
	return C.mlx_vector_array{
		ctx: v.box.ctx
	}
}

// new_vector_array returns an empty vector of arrays.
pub fn new_vector_array() VectorArray {
	return VectorArray{
		box: wrap_handle(C.mlx_vector_array_new().ctx, free_vector_array_handle, true)
	}
}

// array_vector builds a VectorArray from a V slice of arrays.
pub fn array_vector(arrays []Array) VectorArray {
	vec := C.mlx_vector_array_new()
	for a in arrays {
		C.mlx_vector_array_append_value(vec, a.raw())
	}
	return VectorArray{
		box: wrap_handle(vec.ctx, free_vector_array_handle, true)
	}
}

// free releases the vector (but not the arrays it holds); idempotent.
pub fn (v &VectorArray) free() {
	if isnil(v.box) {
		return
	}
	mut box := v.box
	box.release()
}

// len returns the number of arrays.
pub fn (v VectorArray) len() int {
	return int(C.mlx_vector_array_size(v.raw()))
}

// get returns the array at `idx`.
pub fn (v VectorArray) get(idx int) Array {
	res := C.mlx_array_new()
	C.mlx_vector_array_get(&res, v.raw(), usize(idx))
	return wrap_array(res)
}

// to_slice copies the vector into a V slice of arrays.
pub fn (v VectorArray) to_slice() []Array {
	n := v.len()
	mut out := []Array{cap: n}
	for i in 0 .. n {
		out << v.get(i)
	}
	return out
}

// array_vector_to_slice converts a raw vector into a V slice of arrays.
fn array_vector_to_slice(vec C.mlx_vector_array) []Array {
	v := VectorArray{
		box: wrap_handle(vec.ctx, free_vector_array_handle, true)
	}
	defer {
		v.free()
	}
	return v.to_slice()
}

// VectorInt is a growable list of ints.
pub struct VectorInt {
mut:
	box &HandleBox = unsafe { nil }
}

// raw returns the underlying MLX handle (low level).
@[inline]
pub fn (v VectorInt) raw() C.mlx_vector_int {
	if isnil(v.box) {
		panic('mlx: VectorInt is uninitialised (zero value)')
	}
	return C.mlx_vector_int{
		ctx: v.box.ctx
	}
}

// free releases the vector (idempotent; optional with the GC).
pub fn (v &VectorInt) free() {
	if isnil(v.box) {
		return
	}
	mut box := v.box
	box.release()
}

pub fn (v VectorInt) len() int {
	return int(C.mlx_vector_int_size(v.raw()))
}

pub fn (v VectorInt) get(idx int) int {
	mut res := 0
	C.mlx_vector_int_get(&res, v.raw(), usize(idx))
	return res
}

// VectorString is a growable list of strings.
pub struct VectorString {
mut:
	box &HandleBox = unsafe { nil }
}

// raw returns the underlying MLX handle (low level).
@[inline]
pub fn (v VectorString) raw() C.mlx_vector_string {
	if isnil(v.box) {
		panic('mlx: VectorString is uninitialised (zero value); build it with new_vector_string() before using it')
	}
	return C.mlx_vector_string{
		ctx: v.box.ctx
	}
}

pub fn new_vector_string() VectorString {
	return VectorString{
		box: wrap_handle(C.mlx_vector_string_new().ctx, free_vector_string_handle, true)
	}
}

// free releases the vector (idempotent; optional with the GC).
pub fn (v &VectorString) free() {
	if isnil(v.box) {
		return
	}
	mut box := v.box
	box.release()
}

pub fn (v VectorString) len() int {
	return int(C.mlx_vector_string_size(v.raw()))
}

pub fn (v VectorString) get(idx int) string {
	mut res := &char(unsafe { nil })
	C.mlx_vector_string_get(&res, v.raw(), usize(idx))
	return cstr(res)
}

// to_slice copies the vector into a V slice of strings.
pub fn (v VectorString) to_slice() []string {
	n := v.len()
	mut out := []string{cap: n}
	for i in 0 .. n {
		out << v.get(i)
	}
	return out
}

// strings_to_vector builds a raw string vector from a V slice.
fn strings_to_vector(xs []string) C.mlx_vector_string {
	v := C.mlx_vector_string_new()
	for x in xs {
		C.mlx_vector_string_append_value(v, x.str)
	}
	return v
}
