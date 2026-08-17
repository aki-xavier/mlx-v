module mlx

// linalg.v — linear algebra operations.

// inv returns the inverse of a square matrix.
pub fn (a Array) inv() Array {
	res := new_result()
	check(C.mlx_linalg_inv(&res, a.raw(), def_stream()))
	return wrap_array(res)
}

// pinv returns the Moore-Penrose pseudo-inverse.
pub fn (a Array) pinv() Array {
	res := new_result()
	check(C.mlx_linalg_pinv(&res, a.raw(), def_stream()))
	return wrap_array(res)
}

// cholesky returns the Cholesky decomposition.
pub fn (a Array) cholesky(upper bool) Array {
	res := new_result()
	check(C.mlx_linalg_cholesky(&res, a.raw(), upper, def_stream()))
	return wrap_array(res)
}

// cholesky_inv returns the inverse of a PSD matrix via Cholesky.
pub fn (a Array) cholesky_inv(upper bool) Array {
	res := new_result()
	check(C.mlx_linalg_cholesky_inv(&res, a.raw(), upper, def_stream()))
	return wrap_array(res)
}

// tri_inv returns the inverse of a triangular matrix.
pub fn (a Array) tri_inv(upper bool) Array {
	res := new_result()
	check(C.mlx_linalg_tri_inv(&res, a.raw(), upper, def_stream()))
	return wrap_array(res)
}

// solve solves `a x = b`.
pub fn (a Array) solve(b Array) Array {
	res := new_result()
	check(C.mlx_linalg_solve(&res, a.raw(), b.raw(), def_stream()))
	return wrap_array(res)
}

// solve_triangular solves a triangular system `a x = b`.
pub fn (a Array) solve_triangular(b Array, upper bool) Array {
	res := new_result()
	check(C.mlx_linalg_solve_triangular(&res, a.raw(), b.raw(), upper, def_stream()))
	return wrap_array(res)
}

// qr returns the QR decomposition (Q, R).
pub fn (a Array) qr() (Array, Array) {
	setup()
	begin_op()
	r0 := C.mlx_array_new()
	r1 := C.mlx_array_new()
	check(C.mlx_linalg_qr(&r0, &r1, a.raw(), def_stream()))
	return wrap_array(r0), wrap_array(r1)
}

// svd returns the singular value decomposition as [U, S, Vt].
pub fn (a Array) svd(compute_uv bool) []Array {
	setup()
	begin_op()
	vec := C.mlx_vector_array_new()
	check(C.mlx_linalg_svd(&vec, a.raw(), compute_uv, def_stream()))
	return array_vector_to_slice(vec)
}

// lu returns the LU decomposition as [P, L, U].
pub fn (a Array) lu() []Array {
	setup()
	begin_op()
	vec := C.mlx_vector_array_new()
	check(C.mlx_linalg_lu(&vec, a.raw(), def_stream()))
	return array_vector_to_slice(vec)
}

// lu_factor returns the packed LU factorisation and pivots.
pub fn (a Array) lu_factor() (Array, Array) {
	setup()
	begin_op()
	r0 := C.mlx_array_new()
	r1 := C.mlx_array_new()
	check(C.mlx_linalg_lu_factor(&r0, &r1, a.raw(), def_stream()))
	return wrap_array(r0), wrap_array(r1)
}

// eig returns the eigenvalues and eigenvectors of a general matrix.
pub fn (a Array) eig() (Array, Array) {
	setup()
	begin_op()
	r0 := C.mlx_array_new()
	r1 := C.mlx_array_new()
	check(C.mlx_linalg_eig(&r0, &r1, a.raw(), def_stream()))
	return wrap_array(r0), wrap_array(r1)
}

// eigh returns the eigenvalues and eigenvectors of a symmetric matrix.
pub fn (a Array) eigh(uplo string) (Array, Array) {
	setup()
	begin_op()
	r0 := C.mlx_array_new()
	r1 := C.mlx_array_new()
	check(C.mlx_linalg_eigh(&r0, &r1, a.raw(), uplo.str, def_stream()))
	return wrap_array(r0), wrap_array(r1)
}

// eigvals returns the eigenvalues of a general matrix.
pub fn (a Array) eigvals() Array {
	res := new_result()
	check(C.mlx_linalg_eigvals(&res, a.raw(), def_stream()))
	return wrap_array(res)
}

// eigvalsh returns the eigenvalues of a symmetric matrix.
pub fn (a Array) eigvalsh(uplo string) Array {
	res := new_result()
	check(C.mlx_linalg_eigvalsh(&res, a.raw(), uplo.str, def_stream()))
	return wrap_array(res)
}

// cross returns the cross product of `a` and `b` along `axis`.
pub fn (a Array) cross(b Array, axis int) Array {
	res := new_result()
	check(C.mlx_linalg_cross(&res, a.raw(), b.raw(), axis, def_stream()))
	return wrap_array(res)
}

// norm returns the vector/matrix norm with order `ord`.
pub fn (a Array) norm(ord f64) Array {
	res := new_result()
	check(C.mlx_linalg_norm(&res, a.raw(), ord, unsafe { nil }, 0, false, def_stream()))
	return wrap_array(res)
}

// norm_axes returns the norm over `axes` with order `ord`.
pub fn (a Array) norm_axes(ord f64, axes []int, keepdims bool) Array {
	res := new_result()
	check(C.mlx_linalg_norm(&res, a.raw(), ord, axes.data, axes.len, keepdims, def_stream()))
	return wrap_array(res)
}
