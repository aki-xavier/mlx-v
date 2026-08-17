module mlx

// compile.v — graph compilation controls.
//
// Note: `mlx.compile(fun)` itself (wrapping a V callback) lives in
// transforms.v as `compile()`; these functions control the process-wide
// compile machinery (on/off toggle and compile mode).

// enable_compile turns on compilation.
pub fn enable_compile() {
	setup()
	begin_op()
	check(C.mlx_enable_compile())
}

// disable_compile turns off compilation.
pub fn disable_compile() {
	setup()
	begin_op()
	check(C.mlx_disable_compile())
}

// set_compile_mode sets the compile mode.
pub fn set_compile_mode(mode CompileMode) {
	setup()
	begin_op()
	check(C.mlx_set_compile_mode(int(mode)))
}
