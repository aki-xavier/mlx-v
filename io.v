module mlx

// io.v — array serialisation (.npy/.npz/.safetensors/.gguf).

// load reads an array from `file` (.npy or a single-tensor .safetensors).
pub fn load(file string) Array {
	setup()
	begin_op()
	res := C.mlx_array_new()
	check(C.mlx_load(&res, file.str, def_stream()))
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
	check(C.mlx_load_safetensors(&m0, &m1, file.str, def_stream()))
	return MapStringToArray{
		ctx: m0
	}, MapStringToString{
		ctx: m1
	}
}

// save_safetensors writes `tensors` and `metadata` to `file`.
pub fn save_safetensors(file string, tensors MapStringToArray, metadata MapStringToString) {
	setup()
	begin_op()
	check(C.mlx_save_safetensors(file.str, tensors.ctx, metadata.ctx))
}

// Gguf wraps an MLX GGUF object.
pub struct Gguf {
mut:
	ctx C.mlx_io_gguf
}

// load_gguf reads a GGUF file.
pub fn load_gguf(file string) Gguf {
	setup()
	begin_op()
	g := C.mlx_io_gguf_new()
	check(C.mlx_load_gguf(&g, file.str, def_stream()))
	return Gguf{
		ctx: g
	}
}

// new_gguf returns an empty GGUF object (for writing).
pub fn new_gguf() Gguf {
	return Gguf{
		ctx: C.mlx_io_gguf_new()
	}
}

pub fn (g &Gguf) free() {
	C.mlx_io_gguf_free(g.ctx)
}

// save writes the GGUF object to `file`.
pub fn (g Gguf) save(file string) {
	setup()
	begin_op()
	check(C.mlx_save_gguf(file.str, g.ctx))
}

// keys returns the tensor names in the GGUF.
pub fn (g Gguf) keys() []string {
	v := C.mlx_vector_string_new()
	setup()
	begin_op()
	check(C.mlx_io_gguf_get_keys(&v, g.ctx))
	vs := VectorString{
		ctx: v
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
	check(C.mlx_io_gguf_get_array(&res, g.ctx, key.str))
	return wrap_array(res)
}

// metadata_string returns a string metadata entry.
pub fn (g Gguf) metadata_string(key string) string {
	s := C.mlx_string_new()
	setup()
	begin_op()
	check(C.mlx_io_gguf_get_metadata_string(&s, g.ctx, key.str))
	res := cstr(C.mlx_string_data(s))
	C.mlx_string_free(s)
	return res
}

// set_array stores `arr` under `key`.
pub fn (g Gguf) set_array(key string, arr Array) {
	setup()
	begin_op()
	check(C.mlx_io_gguf_set_array(g.ctx, key.str, arr.raw()))
}

// set_metadata_string stores a string metadata entry.
pub fn (g Gguf) set_metadata_string(key string, value string) {
	setup()
	begin_op()
	check(C.mlx_io_gguf_set_metadata_string(g.ctx, key.str, value.str))
}
