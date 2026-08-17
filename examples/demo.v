module main

import mlx

fn main() {
	println('=== mlx-v demo ===')
	println('version: ${mlx.version()}')
	println('metal: ${mlx.metal_available()}')

	// elementwise + operators
	a := mlx.array_f32([f32(1), 2, 3, 4], [2, 2])
	b := mlx.array_f32([f32(10), 20, 30, 40], [2, 2])
	s := a + b
	d := b - a
	p := a * b
	q := b / a
	println('a + b = ${s.data_f32()}')
	println('b - a = ${d.data_f32()}')
	println('a * b = ${p.data_f32()}')
	println('b / a = ${q.data_f32()}')

	// reductions
	println('sum = ${a.sum().item_f32()}')
	println('mean = ${a.mean().item_f32()}')
	println('max = ${a.max().item_f32()}')
	println('min = ${a.min().item_f32()}')

	// softmax
	sm := mlx.array_f32([f32(1), 2, 3], [3])
	println('softmax = ${sm.softmax(false).data_f32()}')

	// matmul
	m := mlx.array_f32([f32(1), 2, 3, 4], [2, 2])
	println('matmul = ${m.matmul(m).data_f32()}')

	// transpose / reshape
	t := m.transpose()
	println('transpose = ${t.data_f32()}')
	r := mlx.arange(0, 12, 1, .float32).reshape([3, 4])
	println('arange.reshape = ${r}')

	// random
	k := mlx.random_key(42)
	x := mlx.random_normal([3, 3], .float32, 0.0, 1.0, k)
	println('random_normal shape = ${x.shape()}')

	// linalg (some decompositions are CPU-only in MLX)
	mlx.use_cpu()
	i := mlx.array_f32([f32(4), 7, 2, 6], [2, 2])
	inv := i.inv()
	println('inv = ${inv.data_f32()}')
	mlx.use_gpu()

	// cleanup
	a.free()
	b.free()
	s.free()
	d.free()
	p.free()
	q.free()
	sm.free()
	m.free()
	t.free()
	r.free()
	k.free()
	x.free()
	i.free()
	inv.free()
	println('=== done ===')
}
