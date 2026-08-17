module main

import mlx

// Linear model: y = W * x + b, loss = mean((y - target)^2).
// We differentiate with respect to [W, b].
fn loss(params []mlx.Array) []mlx.Array {
	w := params[0]
	b := params[1]
	x := params[2]
	target := params[3]
	y := x.matmul(w).add(b)
	return [y.subtract(target).square().mean()]
}

fn main() {
	println('GPU available: ${mlx.gpu_available()}')

	// Ground truth: y = 3*x + 1
	x := mlx.array_f32([f32(1), 2, 3, 4], [4, 1])
	target := mlx.array_f32([f32(4), 7, 10, 13], [4, 1])
	mut w := mlx.array_f32([f32(0.5)], [1, 1])
	mut b := mlx.f32_scalar(0.0)
	defer {
		x.free()
		target.free()
		w.free()
		b.free()
	}

	vag := mlx.value_and_grad(loss, [0, 1])
	defer {
		vag.free()
	}

	lr := mlx.f32_scalar(0.05)
	defer {
		lr.free()
	}

	for step in 0 .. 300 {
		values, grads := vag.apply([w, b, x, target])
		l := values[0]
		dw := grads[0]
		db := grads[1]
		loss_v := l.item_f32()

		// w -= lr * dw ; b -= lr * db
		w2 := w.subtract(dw.multiply(lr))
		b2 := b.subtract(db.multiply(lr))
		w.free()
		b.free()
		w = w2
		b = b2

		l.free()
		dw.free()
		db.free()

		if step % 50 == 0 {
			println('step ${step:3d}: loss = ${loss_v:8.5f}, w = ${w.item_f32():7.4f}, b = ${b.item_f32():7.4f}')
		}
	}
	println('final: w = ${w.item_f32():.4f}, b = ${b.item_f32():.4f}  (expect w=3, b=1)')
}
