module mlx

// fast.v — fast fused kernels (Metal/CUDA).

// layer_norm applies layer normalisation.  `weight`/`bias` may be empty.
pub fn (a Array) layer_norm(weight Array, bias Array, eps f32) Array {
	res := new_result()
	check(C.mlx_fast_layer_norm(&res, a.raw(), weight.raw(), bias.raw(), eps, def_stream()))
	return wrap_array(res)
}

// rms_norm applies RMS normalisation.  `weight` may be empty.
pub fn (a Array) rms_norm(weight Array, eps f32) Array {
	res := new_result()
	check(C.mlx_fast_rms_norm(&res, a.raw(), weight.raw(), eps, def_stream()))
	return wrap_array(res)
}

// scaled_dot_product_attention computes attention with a fused kernel.
// `mask_arr`/`sinks` may be empty; `mask_mode` is one of "", "causal",
// "additive" or "masked".
pub fn scaled_dot_product_attention(queries Array, keys Array, values Array, scale f32, mask_mode string, mask_arr Array, sinks Array) Array {
	res := new_result()
	check(C.mlx_fast_scaled_dot_product_attention(&res, queries.raw(), keys.raw(), values.raw(), scale,
		mask_mode.str, mask_arr.raw(), sinks.raw(), def_stream()))
	return wrap_array(res)
}

// optional_float builds the C optional<float> used by the rope kernels.
pub fn optional_float(value f32) C.mlx_optional_float {
	return C.mlx_optional_float{
		value: value
		has_value: true
	}
}

// no_optional_float returns an empty optional<float>.
pub fn no_optional_float() C.mlx_optional_float {
	return C.mlx_optional_float{
		value: 0
		has_value: false
	}
}

// rope applies rotary position embeddings.  `freqs` may be empty.
pub fn (a Array) rope(dims int, traditional bool, base C.mlx_optional_float, scale f32, offset int, freqs Array) Array {
	res := new_result()
	check(C.mlx_fast_rope(&res, a.raw(), dims, traditional, base, scale, offset, freqs.raw(),
		def_stream()))
	return wrap_array(res)
}

// MetalKernelConfig configures a custom Metal kernel launch.
pub struct MetalKernelConfig {
mut:
	ctx C.mlx_fast_metal_kernel_config
}

// metal_kernel_config returns an empty kernel configuration.
pub fn metal_kernel_config() MetalKernelConfig {
	return MetalKernelConfig{
		ctx: C.mlx_fast_metal_kernel_config_new()
	}
}

pub fn (c MetalKernelConfig) add_output_arg(shape []int, dtype Dtype) {
	C.mlx_fast_metal_kernel_config_add_output_arg(c.ctx, shape.data, shape.len, int(dtype))
}

pub fn (c MetalKernelConfig) set_grid(g1 int, g2 int, g3 int) {
	C.mlx_fast_metal_kernel_config_set_grid(c.ctx, g1, g2, g3)
}

pub fn (c MetalKernelConfig) set_thread_group(t1 int, t2 int, t3 int) {
	C.mlx_fast_metal_kernel_config_set_thread_group(c.ctx, t1, t2, t3)
}

pub fn (c MetalKernelConfig) set_init_value(v f32) {
	C.mlx_fast_metal_kernel_config_set_init_value(c.ctx, v)
}

pub fn (c MetalKernelConfig) set_verbose(v bool) {
	C.mlx_fast_metal_kernel_config_set_verbose(c.ctx, v)
}

pub fn (c MetalKernelConfig) add_template_arg_dtype(name string, dtype Dtype) {
	C.mlx_fast_metal_kernel_config_add_template_arg_dtype(c.ctx, name.str, int(dtype))
}

pub fn (c MetalKernelConfig) add_template_arg_int(name string, v int) {
	C.mlx_fast_metal_kernel_config_add_template_arg_int(c.ctx, name.str, v)
}

pub fn (c MetalKernelConfig) add_template_arg_bool(name string, v bool) {
	C.mlx_fast_metal_kernel_config_add_template_arg_bool(c.ctx, name.str, v)
}

pub fn (c &MetalKernelConfig) free() {
	C.mlx_fast_metal_kernel_config_free(c.ctx)
}

// MetalKernel is a custom Metal (GPU) kernel compiled from MSL source.
pub struct MetalKernel {
mut:
	ctx C.mlx_fast_metal_kernel
}

// metal_kernel builds a kernel from Metal Shading Language source.
pub fn metal_kernel(name string, input_names []string, output_names []string, source string, header string, ensure_row_contiguous bool, atomic_outputs bool) MetalKernel {
	in_names := strings_to_vector(input_names)
	out_names := strings_to_vector(output_names)
	defer {
		C.mlx_vector_string_free(in_names)
		C.mlx_vector_string_free(out_names)
	}
	return MetalKernel{
		ctx: C.mlx_fast_metal_kernel_new(name.str, in_names, out_names, source.str, header.str,
			ensure_row_contiguous, atomic_outputs)
	}
}

// apply runs the kernel on `inputs`.
pub fn (k MetalKernel) apply(inputs []Array, config MetalKernelConfig) []Array {
	setup()
	begin_op()
	vin := arrays_to_vector(inputs)
	defer {
		C.mlx_vector_array_free(vin)
	}
	out := C.mlx_vector_array_new()
	check(C.mlx_fast_metal_kernel_apply(&out, k.ctx, vin, config.ctx, def_stream()))
	return array_vector_to_slice(out)
}

pub fn (k &MetalKernel) free() {
	C.mlx_fast_metal_kernel_free(k.ctx)
}
