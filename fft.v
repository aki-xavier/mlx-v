module mlx

// fft.v — discrete Fourier transforms.

// fft returns the 1-d FFT along `axis`.
pub fn (a Array) fft(n int, axis int, norm FftNorm) Array {
	res := new_result()
	check(C.mlx_fft_fft(&res, a.ctx, n, axis, int(norm), def_stream()))
	return Array{
		ctx: res
	}
}

// ifft returns the 1-d inverse FFT along `axis`.
pub fn (a Array) ifft(n int, axis int, norm FftNorm) Array {
	res := new_result()
	check(C.mlx_fft_ifft(&res, a.ctx, n, axis, int(norm), def_stream()))
	return Array{
		ctx: res
	}
}

// rfft returns the 1-d real FFT along `axis`.
pub fn (a Array) rfft(n int, axis int, norm FftNorm) Array {
	res := new_result()
	check(C.mlx_fft_rfft(&res, a.ctx, n, axis, int(norm), def_stream()))
	return Array{
		ctx: res
	}
}

// irfft returns the 1-d inverse real FFT along `axis`.
pub fn (a Array) irfft(n int, axis int, norm FftNorm) Array {
	res := new_result()
	check(C.mlx_fft_irfft(&res, a.ctx, n, axis, int(norm), def_stream()))
	return Array{
		ctx: res
	}
}

// fftn returns the N-d FFT over `axes`.
pub fn (a Array) fftn(n []int, axes []int, norm FftNorm) Array {
	res := new_result()
	check(C.mlx_fft_fftn(&res, a.ctx, n.data, n.len, axes.data, axes.len, int(norm), def_stream()))
	return Array{
		ctx: res
	}
}

// ifftn returns the N-d inverse FFT over `axes`.
pub fn (a Array) ifftn(n []int, axes []int, norm FftNorm) Array {
	res := new_result()
	check(C.mlx_fft_ifftn(&res, a.ctx, n.data, n.len, axes.data, axes.len, int(norm), def_stream()))
	return Array{
		ctx: res
	}
}

// rfftn returns the N-d real FFT over `axes`.
pub fn (a Array) rfftn(n []int, axes []int, norm FftNorm) Array {
	res := new_result()
	check(C.mlx_fft_rfftn(&res, a.ctx, n.data, n.len, axes.data, axes.len, int(norm), def_stream()))
	return Array{
		ctx: res
	}
}

// irfftn returns the N-d inverse real FFT over `axes`.
pub fn (a Array) irfftn(n []int, axes []int, norm FftNorm) Array {
	res := new_result()
	check(C.mlx_fft_irfftn(&res, a.ctx, n.data, n.len, axes.data, axes.len, int(norm), def_stream()))
	return Array{
		ctx: res
	}
}

// fftshift shifts the zero-frequency component to the centre.
pub fn (a Array) fftshift() Array {
	res := new_result()
	check(C.mlx_fft_fftshift(&res, a.ctx, unsafe { nil }, 0, def_stream()))
	return Array{
		ctx: res
	}
}

// ifftshift undoes fftshift.
pub fn (a Array) ifftshift() Array {
	res := new_result()
	check(C.mlx_fft_ifftshift(&res, a.ctx, unsafe { nil }, 0, def_stream()))
	return Array{
		ctx: res
	}
}

// fftfreq returns the FFT sample frequencies.
pub fn fftfreq(n int, d f64) Array {
	res := new_result()
	check(C.mlx_fft_fftfreq(&res, n, d, def_stream()))
	return Array{
		ctx: res
	}
}

// rfftfreq returns the real FFT sample frequencies.
pub fn rfftfreq(n int, d f64) Array {
	res := new_result()
	check(C.mlx_fft_rfftfreq(&res, n, d, def_stream()))
	return Array{
		ctx: res
	}
}
