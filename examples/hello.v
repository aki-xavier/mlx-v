module main

import mlx

fn main() {
	println('MLX version: ${mlx.version()}')
	println('GPU (Metal) available: ${mlx.metal_available()}')

	a := mlx.array_f32([f32(1), 2, 3, 4], [2, 2])
	defer {
		a.free()
	}
	println('a = ${a}')
	println('shape = ${a.shape()}, dtype = ${a.dtype()}, size = ${a.size()}')
}
