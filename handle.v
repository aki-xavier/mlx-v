module mlx

// handle.v — generic shared box for non-array MLX handles.
//
// Same pattern as ArrayBox (array.v): the wrapper struct holds a `&HandleBox`
// pointer allocated on the Boehm GC heap, with a finalizer that releases the C
// handle once the last copy of the wrapper goes away.  `free()` releases the
// handle deterministically and idempotently.  With `owned == false` (shared
// handles such as the cached default streams or the stream override) both
// `free()` and the finalizer only mark the box, leaving the handle alive.
//
// Unlike ArrayBox these boxes do not feed the `live_arrays()` counter, which
// is reserved for Array handles.

// HandleFreeFn adapts a typed C `mlx_*_free` function to the box.
type HandleFreeFn = fn (voidptr)

// HandleBox holds a raw MLX handle (as its inner void pointer) on the GC heap.
@[heap]
struct HandleBox {
mut:
	ctx     voidptr
	free_fn HandleFreeFn = unsafe { nil }
	freed   bool
	owned   bool
	cached  bool
}

// handle_finalizer releases the C handle when its box is garbage-collected.
fn handle_finalizer(obj voidptr, _cd voidptr) {
	mut box := unsafe { &HandleBox(obj) }
	box.release()
}

// wrap_handle boxes a raw MLX handle and registers its finalizer.  `owned`
// must be false for shared handles whose lifetime is managed elsewhere.
//
// `cached` snapshots whether this box wraps a shared cached default handle
// (e.g. cpu_stream()/gpu_stream()).  It is captured at creation time and is
// *not* mutated by `set_default()`, which disowns user-owned boxes by flipping
// `owned`.  That lets Stream.set_default() keep the process-wide override
// "owned" even when it is called more than once on the same stream (otherwise
// the second call would mark the override borrowed and leak it).
fn wrap_handle(ctx voidptr, free_fn HandleFreeFn, owned bool) &HandleBox {
	// Allocate from the Boehm GC heap explicitly (see wrap_array).
	mut box := unsafe { &HandleBox(C.mlx_v_gc_malloc(sizeof(HandleBox))) }
	box.ctx = ctx
	box.free_fn = free_fn
	box.freed = false
	box.owned = owned
	box.cached = !owned
	register_finalizer(box, handle_finalizer)
	return box
}

// release frees the boxed handle once.  Non-owned boxes are only marked, so
// shared handles outlive their wrappers.  Call it from the wrappers' free()
// methods via `mut box := x.box` (the Array.free idiom).
fn (mut box HandleBox) release() {
	if !box.freed {
		box.freed = true
		if box.owned {
			box.free_fn(box.ctx)
		}
	}
}

// free_*_handle adapt the typed C free functions to the HandleFreeFn
// signature (all MLX handles share the `struct { void *ctx }` layout).
fn free_device_handle(p voidptr) {
	C.mlx_device_free(C.mlx_device{
		ctx: p
	})
}

fn free_stream_handle(p voidptr) {
	C.mlx_stream_free(C.mlx_stream{
		ctx: p
	})
}

fn free_map_string_to_array_handle(p voidptr) {
	C.mlx_map_string_to_array_free(C.mlx_map_string_to_array{
		ctx: p
	})
}

fn free_map_string_to_string_handle(p voidptr) {
	C.mlx_map_string_to_string_free(C.mlx_map_string_to_string{
		ctx: p
	})
}

fn free_vector_array_handle(p voidptr) {
	C.mlx_vector_array_free(C.mlx_vector_array{
		ctx: p
	})
}

fn free_vector_int_handle(p voidptr) {
	C.mlx_vector_int_free(C.mlx_vector_int{
		ctx: p
	})
}

fn free_vector_string_handle(p voidptr) {
	C.mlx_vector_string_free(C.mlx_vector_string{
		ctx: p
	})
}

fn free_closure_handle(p voidptr) {
	C.mlx_closure_free(C.mlx_closure{
		ctx: p
	})
}

fn free_closure_value_and_grad_handle(p voidptr) {
	C.mlx_closure_value_and_grad_free(C.mlx_closure_value_and_grad{
		ctx: p
	})
}

fn free_gguf_handle(p voidptr) {
	C.mlx_io_gguf_free(C.mlx_io_gguf{
		ctx: p
	})
}

fn free_metal_kernel_handle(p voidptr) {
	C.mlx_fast_metal_kernel_free(C.mlx_fast_metal_kernel{
		ctx: p
	})
}

fn free_metal_kernel_config_handle(p voidptr) {
	C.mlx_fast_metal_kernel_config_free(C.mlx_fast_metal_kernel_config{
		ctx: p
	})
}
