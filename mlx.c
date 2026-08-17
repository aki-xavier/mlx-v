/*
 * mlx.c — tiny C helpers for the mlx V bindings.
 *
 * Keeps the last MLX error message, a "force CPU" flag, and the cached default
 * streams in C static storage so that the V module does not need mutable
 * globals (`-enable-globals`).
 */
#include <stdint.h>
#include <string.h>

#include <gc/gc.h>
#include <mlx/c/mlx.h>

static char mlx_v_error_buf[2048];
static int mlx_v_force_cpu = 0;
static int mlx_v_live_boxes = 0;
static mlx_stream mlx_v_cpu_stream = {0};
static mlx_stream mlx_v_gpu_stream = {0};

void mlx_v_note_box_alloc(void) {
    mlx_v_live_boxes++;
}

void mlx_v_note_box_free(void) {
    if (mlx_v_live_boxes > 0) {
        mlx_v_live_boxes--;
    }
}

int mlx_v_get_live_boxes(void) {
    return mlx_v_live_boxes;
}

/* Cached default streams.  mlx_default_*_stream_new() heap-allocates a new
 * Stream wrapper on every call, and the wrappers are never freed by the V
 * bindings, so caching one wrapper per device avoids a per-op allocation leak. */
mlx_stream mlx_v_cached_cpu_stream(void) {
    if (!mlx_v_cpu_stream.ctx) {
        mlx_v_cpu_stream = mlx_default_cpu_stream_new();
    }
    return mlx_v_cpu_stream;
}

mlx_stream mlx_v_cached_gpu_stream(void) {
    if (!mlx_v_gpu_stream.ctx) {
        mlx_v_gpu_stream = mlx_default_gpu_stream_new();
    }
    return mlx_v_gpu_stream;
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
    mlx_v_force_cpu = v;
}

int mlx_v_get_force_cpu(void) {
    return mlx_v_force_cpu;
}
