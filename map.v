module mlx

// map.v — string-keyed maps (used by safetensors loading/saving).

// MapStringToArray is a string -> Array map.  The handle is GC-managed;
// `free()` releases it deterministically.
pub struct MapStringToArray {
mut:
	box &HandleBox = unsafe { nil }
}

// raw returns the underlying MLX handle (low level).
@[inline]
pub fn (m MapStringToArray) raw() C.mlx_map_string_to_array {
	if isnil(m.box) {
		panic('mlx: MapStringToArray is uninitialised (zero value); build it with new_map_string_to_array() before using it')
	}
	return C.mlx_map_string_to_array{
		ctx: m.box.ctx
	}
}

// new_map_string_to_array returns an empty map.
pub fn new_map_string_to_array() MapStringToArray {
	return MapStringToArray{
		box: wrap_handle(C.mlx_map_string_to_array_new().ctx, free_map_string_to_array_handle, true)
	}
}

// free releases the map (idempotent; optional with the GC).
pub fn (m &MapStringToArray) free() {
	if isnil(m.box) {
		return
	}
	mut box := m.box
	box.release()
}

// insert stores `value` under `key`.
pub fn (m MapStringToArray) insert(key string, value Array) {
	setup()
	begin_op()
	check(C.mlx_map_string_to_array_insert(m.raw(), key.str, value.raw()))
}

// get returns the array at `key` (a new reference the caller should free).
// Panics if the key is absent (consistent with the rest of the error handling).
pub fn (m MapStringToArray) get(key string) Array {
	res := C.mlx_array_new()
	setup()
	begin_op()
	check_res(C.mlx_map_string_to_array_get(&res, m.raw(), key.str), res)
	return wrap_array(res)
}

// MapStringToString is a string -> string map (safetensors metadata).  The
// handle is GC-managed; `free()` releases it deterministically.
pub struct MapStringToString {
mut:
	box &HandleBox = unsafe { nil }
}

// raw returns the underlying MLX handle (low level).
@[inline]
pub fn (m MapStringToString) raw() C.mlx_map_string_to_string {
	if isnil(m.box) {
		panic('mlx: MapStringToString is uninitialised (zero value); build it with new_map_string_to_string() before using it')
	}
	return C.mlx_map_string_to_string{
		ctx: m.box.ctx
	}
}

// new_map_string_to_string returns an empty map.
pub fn new_map_string_to_string() MapStringToString {
	return MapStringToString{
		box: wrap_handle(C.mlx_map_string_to_string_new().ctx, free_map_string_to_string_handle,
			true)
	}
}

// free releases the map (idempotent; optional with the GC).
pub fn (m &MapStringToString) free() {
	if isnil(m.box) {
		return
	}
	mut box := m.box
	box.release()
}

// insert stores `value` under `key`.
pub fn (m MapStringToString) insert(key string, value string) {
	setup()
	begin_op()
	check(C.mlx_map_string_to_string_insert(m.raw(), key.str, value.str))
}

// get returns the string at `key`, or '' if absent.
pub fn (m MapStringToString) get(key string) string {
	mut res := &char(unsafe { nil })
	setup()
	begin_op()
	if C.mlx_map_string_to_string_get(&res, m.raw(), key.str) == 0 {
		return cstr(res)
	}
	return ''
}
