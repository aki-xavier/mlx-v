module mlx

// io.v — array serialisation (.npy/.npz/.safetensors/.gguf).

// load reads an array from `file` (.npy or a single-tensor .safetensors).
pub fn load(file string) Array {
	setup()
	begin_op()
	res := C.mlx_array_new()
	check_res(C.mlx_load(&res, file.str, def_stream()), res)
	return wrap_array(res)
}

// save writes `a` to `file` (format chosen by extension).
pub fn save(file string, a Array) {
	setup()
	begin_op()
	check(C.mlx_save(file.str, a.raw()))
}

// load_safetensors reads a safetensors file, returning tensors + metadata.
pub fn load_safetensors(file string) (MapStringToArray, MapStringToString) {
	setup()
	begin_op()
	m0 := C.mlx_map_string_to_array_new()
	m1 := C.mlx_map_string_to_string_new()
	rc := C.mlx_load_safetensors(&m0, &m1, file.str, def_stream())
	if rc != 0 {
		C.mlx_map_string_to_array_free(m0)
		C.mlx_map_string_to_string_free(m1)
		check(rc)
	}
	return MapStringToArray{
		box: wrap_handle(m0.ctx, free_map_string_to_array_handle, true)
	}, MapStringToString{
		box: wrap_handle(m1.ctx, free_map_string_to_string_handle, true)
	}
}

// save_safetensors writes `tensors` and `metadata` to `file`.
pub fn save_safetensors(file string, tensors MapStringToArray, metadata MapStringToString) {
	setup()
	begin_op()
	check(C.mlx_save_safetensors(file.str, tensors.raw(), metadata.raw()))
}

// Gguf wraps an MLX GGUF object.  The handle is GC-managed; `free()` releases
// it deterministically.
pub struct Gguf {
mut:
	box &HandleBox = unsafe { nil }
}

// raw returns the underlying MLX handle (low level).
@[inline]
pub fn (g Gguf) raw() C.mlx_io_gguf {
	if isnil(g.box) {
		panic('mlx: Gguf is uninitialised (zero value); build it with load_gguf()/new_gguf() before using it')
	}
	return C.mlx_io_gguf{
		ctx: g.box.ctx
	}
}

// load_gguf reads a GGUF file.
pub fn load_gguf(file string) Gguf {
	setup()
	begin_op()
	g := C.mlx_io_gguf_new()
	rc := C.mlx_load_gguf(&g, file.str, def_stream())
	if rc != 0 {
		C.mlx_io_gguf_free(g)
		check(rc)
	}
	return Gguf{
		box: wrap_handle(g.ctx, free_gguf_handle, true)
	}
}

// new_gguf returns an empty GGUF object (for writing).
pub fn new_gguf() Gguf {
	return Gguf{
		box: wrap_handle(C.mlx_io_gguf_new().ctx, free_gguf_handle, true)
	}
}

// free releases the GGUF object (idempotent; optional with the GC).
pub fn (g &Gguf) free() {
	if isnil(g.box) {
		return
	}
	mut box := g.box
	box.release()
}

// save writes the GGUF object to `file`.
pub fn (g Gguf) save(file string) {
	setup()
	begin_op()
	check(C.mlx_save_gguf(file.str, g.raw()))
}

// keys returns the tensor names in the GGUF.
pub fn (g Gguf) keys() []string {
	v := C.mlx_vector_string_new()
	setup()
	begin_op()
	rc := C.mlx_io_gguf_get_keys(&v, g.raw())
	if rc != 0 {
		C.mlx_vector_string_free(v)
		check(rc)
	}
	vs := VectorString{
		box: wrap_handle(v.ctx, free_vector_string_handle, true)
	}
	defer {
		vs.free()
	}
	return vs.to_slice()
}

// get_array returns the tensor `key`.
pub fn (g Gguf) get_array(key string) Array {
	res := C.mlx_array_new()
	setup()
	begin_op()
	check_res(C.mlx_io_gguf_get_array(&res, g.raw(), key.str), res)
	return wrap_array(res)
}

// metadata_string returns a string metadata entry.
pub fn (g Gguf) metadata_string(key string) string {
	s := C.mlx_string_new()
	setup()
	begin_op()
	rc := C.mlx_io_gguf_get_metadata_string(&s, g.raw(), key.str)
	if rc != 0 {
		C.mlx_string_free(s)
		check(rc)
	}
	res := cstr(C.mlx_string_data(s))
	C.mlx_string_free(s)
	return res
}

// set_array stores `arr` under `key`.
pub fn (g Gguf) set_array(key string, arr Array) {
	setup()
	begin_op()
	check(C.mlx_io_gguf_set_array(g.raw(), key.str, arr.raw()))
}

// set_metadata_string stores a string metadata entry.
pub fn (g Gguf) set_metadata_string(key string, value string) {
	setup()
	begin_op()
	check(C.mlx_io_gguf_set_metadata_string(g.raw(), key.str, value.str))
}
