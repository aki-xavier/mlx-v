/*
 * mlx_v.h — prototypes for the tiny C helpers in mlx.c.
 */
#ifndef MLX_V_HELPERS_H
#define MLX_V_HELPERS_H

#include <stdint.h>

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

#ifdef __cplusplus
}
#endif

#endif
