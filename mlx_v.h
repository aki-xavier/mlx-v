/*
 * mlx_v.h — prototypes for the tiny C helpers in mlx.c.
 */
#ifndef MLX_V_HELPERS_H
#define MLX_V_HELPERS_H

#include <stddef.h>
#include <stdint.h>

#include <mlx/c/mlx.h>

#ifdef __cplusplus
extern "C" {
#endif

void mlx_v_clear_error(void);
void mlx_v_set_error(const char *msg);
const char *mlx_v_get_error(void);
void mlx_v_set_force_cpu(int v);
int mlx_v_get_force_cpu(void);

float mlx_v_f16_to_f32(uint16_t h);
float mlx_v_bf16_to_f32(uint16_t h);

void mlx_v_note_box_alloc(void);
void mlx_v_note_box_free(void);
int mlx_v_get_live_boxes(void);

mlx_stream mlx_v_cached_cpu_stream(void);
mlx_stream mlx_v_cached_gpu_stream(void);

void mlx_v_register_finalizer(void *obj, void (*fn)(void *, void *));
void mlx_v_gc_collect(void);
void *mlx_v_gc_malloc(size_t n);

#ifdef __cplusplus
}
#endif

#endif
