module mlx_ops

import math

import mlx { Array, arr32, array_bool, axis_logsumexp, col, complex_from, fft2, ifft2, nonzero_indices, overwrite_region, pad_edge, roll_axis, sel1, slice_rows, zeros }

// test_random and test_composite_helpers were split out of mlx_test.v when
// the compute layer moved to ops/ (they exercise mlx_ops functions; importing
// mlx_ops from a module-mlx test would create an import cycle).


fn test_random() {
	seed := random_key(1234)
	defer {
		seed.free()
	}
	x := random_normal([100], .float32, 0.0, 1.0, seed)
	defer {
		x.free()
	}
	assert x.shape() == [100]
	assert x.dtype() == .float32
}

fn test_composite_helpers() {
	// arr32 / sel1
	a := arr32([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
	assert a.dtype() == .float32
	assert a.shape() == [2, 3]
	assert a.data_f32() == [f32(1), 2, 3, 4, 5, 6]
	s := sel1(2)
	assert s.shape() == [1]
	assert s.data_i32() == [2]

	// col / slice_rows
	assert col(a, 1).data_f32() == [f32(2), 5]
	assert slice_rows(a, 0, 1).data_f32() == [f32(1), 2, 3]

	// nonzero_indices
	mask := array_bool([true, false, true, true, false], [5])
	assert nonzero_indices(mask).data_i32() == [0, 2, 3]

	// axis_logsumexp: log(exp(1)+exp(2)+exp(3)) ≈ 3.4076
	lse := axis_logsumexp(a, 1)
	ld := lse.data_f32()
	assert math.abs(ld[0] - 3.407606) < 1e-4
	assert math.abs(ld[1] - 6.407606) < 1e-4

	// roll_axis: shift -1 along axis 0 pulls the next row in
	assert roll_axis(a, -1, 0).data_f32() == [f32(4), 5, 6, 1, 2, 3]

	// overwrite_region: replace a[0:1, 1:3] with [7, 8]
	patch := arr32([7.0, 8.0], [1, 2])
	assert overwrite_region(a, patch, 0, 1).data_f32() == [f32(1), 7, 8, 4, 5, 6]

	// pad_edge
	p1 := arr32([1.0, 2.0, 3.0, 4.0], [2, 2])
	assert pad_edge(p1, 1).data_f32() == [f32(1), 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4]

	// complex_from + fft2/ifft2 round-trip
	re := complex_from(p1, zeros([2, 2], .float32))
	back := ifft2(fft2(re)).real()
	bd := back.data_f32()
	for i, v in [f32(1), 2, 3, 4] {
		assert math.abs(bd[i] - v) < 1e-4
	}

	// split_keys
	keys := split_keys(42, 3)
	assert keys.len == 3
	for k in keys {
		assert k.shape() == [2]
		assert k.dtype() == .uint32
	}
}