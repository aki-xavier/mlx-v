module mlx

// map.v — string-keyed maps (used by safetensors loading/saving).

// MapStringToArray is a string -> Array map.
pub struct MapStringToArray {
mut:
	ctx   C.mlx_map_string_to_array
	freed bool
}

// new_map_string_to_array returns an empty map.
pub fn new_map_string_to_array() MapStringToArray {
	return MapStringToArray{
		ctx: C.mlx_map_string_to_array_new()
	}
}

pub fn (mut m MapStringToArray) free() {
	if !m.freed {
		m.freed = true
		C.mlx_map_string_to_array_free(m.ctx)
	}
}

// insert stores `value` under `key`.
pub fn (m MapStringToArray) insert(key string, value Array) {
	setup()
	begin_op()
	check(C.mlx_map_string_to_array_insert(m.ctx, key.str, value.raw()))
}

// get returns the array at `key` (a new reference the caller should free).
// Panics if the key is absent (consistent with the rest of the error handling).
pub fn (m MapStringToArray) get(key string) Array {
	res := C.mlx_array_new()
	setup()
	begin_op()
	check(C.mlx_map_string_to_array_get(&res, m.ctx, key.str))
	return wrap_array(res)
}

// MapStringToString is a string -> string map (safetensors metadata).
pub struct MapStringToString {
mut:
	ctx   C.mlx_map_string_to_string
	freed bool
}

// new_map_string_to_string returns an empty map.
pub fn new_map_string_to_string() MapStringToString {
	return MapStringToString{
		ctx: C.mlx_map_string_to_string_new()
	}
}

pub fn (mut m MapStringToString) free() {
	if !m.freed {
		m.freed = true
		C.mlx_map_string_to_string_free(m.ctx)
	}
}

// insert stores `value` under `key`.
pub fn (m MapStringToString) insert(key string, value string) {
	setup()
	begin_op()
	check(C.mlx_map_string_to_string_insert(m.ctx, key.str, value.str))
}

// get returns the string at `key`, or '' if absent.
pub fn (m MapStringToString) get(key string) string {
	mut res := &char(unsafe { nil })
	setup()
	begin_op()
	if C.mlx_map_string_to_string_get(&res, m.ctx, key.str) == 0 {
		return cstr(res)
	}
	return ''
}
