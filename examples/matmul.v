module main

import mlx
import time

fn main() {
	println('MLX ${mlx.version()} — GPU available: ${mlx.gpu_available()}')

	// Small correctness check on the default device.
	a := mlx.array_f32([f32(1), 2, 3, 4, 5, 6], [2, 3])
	b := mlx.array_f32([f32(7), 8, 9, 10, 11, 12], [3, 2])
	c := a.matmul(b)
	println('a = ${a}')
	println('b = ${b}')
	println('a @ b = ${c}')
	println('result = ${c.data_f32()}')

	a.free()
	b.free()
	c.free()

	// Timed large matmul.
	n := 2048
	println('\nbenchmark ${n}x${n} @ ${n}x${n} (float32) ...')
	x := mlx.random_normal([n, n], .float32, 0.0, 0.02, mlx.no_key())
	y := mlx.random_normal([n, n], .float32, 0.0, 0.02, mlx.no_key())

	sw := time.new_stopwatch()
	z := x.matmul(y)
	z.eval()
	elapsed := sw.elapsed().milliseconds()
	println('matmul took ${elapsed} ms')
	println('peak memory: ${mlx.peak_memory() / (1024 * 1024)} MB')

	x.free()
	y.free()
	z.free()
}
