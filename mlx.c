/*
 * mlx.c — tiny C helpers for the mlx V bindings.
 *
 * Keeps the last MLX error message and a "force CPU" flag in C static storage
 * so that the V module does not need mutable globals (`-enable-globals`).
 */
#include <stdint.h>
#include <string.h>

static char mlx_v_error_buf[2048];
static int mlx_v_force_cpu = 0;

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
