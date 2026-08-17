module mlx

// gc.v — automatic memory management for MLX handles.
//
// V uses the Boehm-Demers-Weiser conservative GC (gcboehm) by default.  MLX
// handles (`mlx_array`, `mlx_stream`, ...) are C++ shared_ptr wrappers, so they
// are invisible to that GC and would leak without an explicit `free()`.
//
// To fix that, every wrapper allocates a small "box" on the GC heap that holds
// the raw handle, and registers a Boehm finalizer on the box.  When the box
// becomes unreachable, the GC runs the finalizer and releases the handle.
// Copies of a wrapper share the same box (shared ownership), so the handle is
// released only after the *last* reference is gone.

// Finalizer is the Boehm finalization callback signature.
type Finalizer = fn (obj voidptr, cd voidptr)

// C wrappers in mlx.c (they call the Boehm GC directly, keeping the GC_*
// prototypes out of the V namespace to avoid conflicts with <gc/gc.h>).
fn C.mlx_v_register_finalizer(obj voidptr, fn_ Finalizer)
fn C.mlx_v_gc_collect()
fn C.mlx_v_gc_malloc(n usize) voidptr

// register_finalizer attaches `fn_` to `obj` so the GC calls it when `obj`
// becomes unreachable.
@[inline]
fn register_finalizer(obj voidptr, fn_ Finalizer) {
	C.mlx_v_register_finalizer(obj, fn_)
}

// gc_collect forces a full garbage-collection cycle, releasing any MLX handles
// that are no longer referenced.
pub fn gc_collect() {
	C.mlx_v_gc_collect()
}

// live_arrays returns the number of MLX array handles currently held by live
// wrapper boxes (useful for detecting leaks; the GC frees them lazily).
pub fn live_arrays() int {
	return C.mlx_v_get_live_boxes()
}
