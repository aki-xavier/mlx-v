module mlx

// vector.v — wrappers for MLX vector-of-array / int / string containers.

// VectorArray is a growable list of MLX arrays.
pub struct VectorArray {
mut:
	ctx C.mlx_vector_array
}

// new_vector_array returns an empty vector of arrays.
pub fn new_vector_array() VectorArray {
	return VectorArray{
		ctx: C.mlx_vector_array_new()
	}
}

// array_vector builds a VectorArray from a V slice of arrays.
pub fn array_vector(arrays []Array) VectorArray {
	vec := C.mlx_vector_array_new()
	for a in arrays {
		C.mlx_vector_array_append_value(vec, a.ctx)
	}
	return VectorArray{
		ctx: vec
	}
}

// free releases the vector (but not the arrays it holds).
pub fn (v &VectorArray) free() {
	C.mlx_vector_array_free(v.ctx)
}

// len returns the number of arrays.
pub fn (v VectorArray) len() int {
	return int(C.mlx_vector_array_size(v.ctx))
}

// get returns the array at `idx`.
pub fn (v VectorArray) get(idx int) Array {
	res := C.mlx_array_new()
	C.mlx_vector_array_get(&res, v.ctx, usize(idx))
	return Array{
		ctx: res
	}
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
		ctx: vec
	}
	defer {
		v.free()
	}
	return v.to_slice()
}

// VectorInt is a growable list of ints.
pub struct VectorInt {
mut:
	ctx C.mlx_vector_int
}

pub fn (v &VectorInt) free() {
	C.mlx_vector_int_free(v.ctx)
}

pub fn (v VectorInt) len() int {
	return int(C.mlx_vector_int_size(v.ctx))
}

pub fn (v VectorInt) get(idx int) int {
	mut res := 0
	C.mlx_vector_int_get(&res, v.ctx, usize(idx))
	return res
}

// VectorString is a growable list of strings.
pub struct VectorString {
mut:
	ctx C.mlx_vector_string
}

pub fn new_vector_string() VectorString {
	return VectorString{
		ctx: C.mlx_vector_string_new()
	}
}

pub fn (v &VectorString) free() {
	C.mlx_vector_string_free(v.ctx)
}

pub fn (v VectorString) len() int {
	return int(C.mlx_vector_string_size(v.ctx))
}

pub fn (v VectorString) get(idx int) string {
	mut res := &char(unsafe { nil })
	C.mlx_vector_string_get(&res, v.ctx, usize(idx))
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
