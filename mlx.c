/*
 * mlx.c — tiny C helpers for the mlx V bindings.
 *
 * Keeps the last MLX error message, a "force CPU" flag, the live-box counter
 * and the cached default streams in C storage so that the V module does not
 * need mutable globals (`-enable-globals`).
 *
 * Everything here is written to be safe under concurrent use:
 *   - the error buffer is thread-local (MLX invokes the error handler on the
 *     thread that triggered the error, so a per-thread buffer prevents one
 *     thread's message from clobbering another's),
 *   - the force-CPU flag and live-box counter are atomic (the latter is
 *     decremented from Boehm finalizers, which may run on any thread),
 *   - the cached streams are initialised exactly once with pthread_once.
 */
#include <stdint.h>
#include <string.h>

#include <stdatomic.h>
#include <pthread.h>

#include <gc/gc.h>
#include <mlx/c/mlx.h>

#include "mlx_v.h"

static __thread char mlx_v_error_buf[2048];
static atomic_int mlx_v_force_cpu = 0;
static atomic_int mlx_v_live_boxes = 0;

static pthread_once_t mlx_v_cpu_stream_once = PTHREAD_ONCE_INIT;
static pthread_once_t mlx_v_gpu_stream_once = PTHREAD_ONCE_INIT;
static mlx_stream mlx_v_cpu_stream = {0};
static mlx_stream mlx_v_gpu_stream = {0};

/* Process-wide stream override installed by Stream.set_default(); stored as
 * the stream's inner pointer so it can be atomic.  `_override_owned` is 1 when
 * the override is the sole owner of its stream (a user-owned stream disowned
 * by set_default) and must be released when the override is replaced/cleared;
 * it is 0 for cached default streams, whose lifetime is managed elsewhere.
 * Only set/clear touch it (process-wide, not thread-safe); the reader below
 * (`mlx_v_stream_for_ops`) sees only the atomic pointer. */
static _Atomic(void *) mlx_v_stream_override = NULL;
static int mlx_v_stream_override_owned = 0;

/* The error handler is installed exactly once (see mlx_v_ensure_error_handler). */
static pthread_once_t mlx_v_error_once = PTHREAD_ONCE_INIT;
static _Atomic(mlx_error_handler_func) mlx_v_error_handler = NULL;

static void mlx_v_install_error_handler(void) {
    mlx_set_error_handler(atomic_load(&mlx_v_error_handler), NULL, NULL);
}

/* The CPU and GPU caches are initialised independently, so touching the CPU
 * stream never initialises the GPU backend (which may not exist). */
static void mlx_v_init_cpu_stream(void) {
    mlx_v_cpu_stream = mlx_default_cpu_stream_new();
}

static void mlx_v_init_gpu_stream(void) {
    /* On machines without a GPU, fall back to the CPU stream so that
     * mlx_v_stream_for_ops() never hands out an unusable GPU stream
     * (preserves the old gpu_available() check in def_stream()). */
    bool gpu = false;
    mlx_metal_is_available(&gpu);
    if (!gpu) {
        mlx_cuda_is_available(&gpu);
    }
    if (gpu) {
        mlx_v_gpu_stream = mlx_default_gpu_stream_new();
    } else {
        mlx_v_gpu_stream = mlx_v_cached_cpu_stream();
    }
}

void mlx_v_note_box_alloc(void) {
    atomic_fetch_add(&mlx_v_live_boxes, 1);
}

void mlx_v_note_box_free(void) {
    int v = atomic_load(&mlx_v_live_boxes);
    while (v > 0 &&
           !atomic_compare_exchange_weak(&mlx_v_live_boxes, &v, v - 1)) {
        // retry on contention; never let the counter go negative
    }
}

int mlx_v_get_live_boxes(void) {
    return atomic_load(&mlx_v_live_boxes);
}

/* Cached default streams.  mlx_default_*_stream_new() heap-allocates a new
 * Stream wrapper on every call, and the wrappers are never freed by the V
 * bindings, so caching one wrapper per device avoids a per-op allocation leak.
 * The cache is shared and initialised exactly once. */
mlx_stream mlx_v_cached_cpu_stream(void) {
    pthread_once(&mlx_v_cpu_stream_once, mlx_v_init_cpu_stream);
    return mlx_v_cpu_stream;
}

mlx_stream mlx_v_cached_gpu_stream(void) {
    pthread_once(&mlx_v_gpu_stream_once, mlx_v_init_gpu_stream);
    return mlx_v_gpu_stream;
}

void mlx_v_set_stream_override(mlx_stream s, int owned) {
    void *old = atomic_exchange(&mlx_v_stream_override, s.ctx);
    if (old != NULL && old != s.ctx && mlx_v_stream_override_owned) {
        mlx_stream o = {old};
        mlx_stream_free(o);
    }
    mlx_v_stream_override_owned = owned;
}

void mlx_v_clear_stream_override(void) {
    void *old = atomic_exchange(&mlx_v_stream_override, NULL);
    if (old != NULL && mlx_v_stream_override_owned) {
        mlx_stream o = {old};
        mlx_stream_free(o);
    }
    mlx_v_stream_override_owned = 0;
}

/* The stream ops run on: the override when one is set (Stream.set_default()),
 * otherwise the cached default stream selected by the force-CPU flag. */
mlx_stream mlx_v_stream_for_ops(void) {
    void *o = atomic_load(&mlx_v_stream_override);
    if (o != NULL) {
        mlx_stream s = {o};
        return s;
    }
    if (mlx_v_get_force_cpu()) {
        return mlx_v_cached_cpu_stream();
    }
    return mlx_v_cached_gpu_stream();
}

/* Installs the error handler exactly once: mlx_set_error_handler writes a
 * process-global variable, so repeating it on every op is an unsynchronised
 * write. */
void mlx_v_ensure_error_handler(mlx_error_handler_func handler) {
    atomic_store(&mlx_v_error_handler, handler);
    pthread_once(&mlx_v_error_once, mlx_v_install_error_handler);
}

/* Boehm GC wrappers (kept out of the V-facing declarations to avoid
 * conflicting with the GC_* prototypes that <gc/gc.h> already provides). */
void mlx_v_register_finalizer(void *obj, void (*fn)(void *, void *)) {
    GC_register_finalizer(obj, fn, 0, 0, 0);
}

void *mlx_v_gc_malloc(size_t n) {
    return GC_MALLOC(n);
}

void mlx_v_gc_collect(void) {
    GC_gcollect();
}

/* IEEE 754 half (float16) -> float32 */
float mlx_v_f16_to_f32(uint16_t h) {
    uint32_t sign = ((uint32_t)(h & 0x8000u)) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign; /* +/- zero */
        } else {
            /* subnormal: normalize */
            exp = 1;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3ffu;
            bits = sign | ((exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13); /* inf / nan */
    } else {
        bits = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

/* bfloat16 -> float32 */
float mlx_v_bf16_to_f32(uint16_t h) {
    uint32_t bits = ((uint32_t)h) << 16;
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

void mlx_v_clear_error(void) {
    mlx_v_error_buf[0] = '\0';
}

void mlx_v_set_error(const char *msg) {
    if (msg == 0) {
        mlx_v_error_buf[0] = '\0';
        return;
    }
    strncpy(mlx_v_error_buf, msg, sizeof(mlx_v_error_buf) - 1);
    mlx_v_error_buf[sizeof(mlx_v_error_buf) - 1] = '\0';
}

const char *mlx_v_get_error(void) {
    return mlx_v_error_buf;
}

void mlx_v_set_force_cpu(int v) {
    atomic_store(&mlx_v_force_cpu, v);
}

int mlx_v_get_force_cpu(void) {
    return atomic_load(&mlx_v_force_cpu);
}
