#!/usr/bin/env python3
"""Generate `cdefs.c.v` (raw mlx-c bindings) from the installed mlx-c headers.

Run:  python3 gen/gen_cdefs.py

The output is deliberately kept in this compact layout (one declaration per
line, struct fields space-aligned) instead of `v fmt` style, which would insert
a blank line between every declaration and double the file size.  Regenerating
from unmodified headers should therefore produce no diff.
"""
import os
import re
import sys

# Location of the installed mlx-c headers.  Defaults to Homebrew (Apple
# Silicon); override with the MLX_INCLUDE_DIR environment variable (the same
# variable the module's #flag lines honour) for Intel Homebrew or Linux.
INC = os.environ.get("MLX_INCLUDE_DIR", "/opt/homebrew/include/mlx/c")
OUT = os.path.join(os.path.dirname(__file__), "..", "cdefs.v")

# Headers to process, in dependency order.
HEADERS = [
    "string.h",
    "array.h",
    "vector.h",
    "map.h",
    "optional.h",
    "device.h",
    "stream.h",
    "io_types.h",
    "distributed_group.h",
    "closure.h",
    "ops.h",
    "random.h",
    "linalg.h",
    "fft.h",
    "fast.h",
    "transforms.h",
    "transforms_impl.h",
    "compile.h",
    "io.h",
    "memory.h",
    "distributed.h",
    "graph_utils.h",
    "metal.h",
    "cuda.h",
    "version.h",
    "export.h",
]

# ---------------------------------------------------------------------------
# Type mappings
# ---------------------------------------------------------------------------

# V keywords / reserved identifiers that must not be used as parameter names.
RESERVED = {
    "type",
    "map",
    "str",
    "error",
    "match",
    "module",
    "interface",
    "for",
    "in",
    "as",
    "is",
}

SCALAR = {
    "bool": "bool",
    "int": "int",
    "float": "f32",
    "double": "f64",
    "size_t": "usize",
    "uint64_t": "u64",
    "uint32_t": "u32",
    "uint16_t": "u16",
    "uint8_t": "u8",
    "int64_t": "i64",
    "int32_t": "int",
    "int16_t": "i16",
    "int8_t": "i8",
    "uintptr_t": "usize",
}

# Opaque typedef'd structs (single `void* ctx` field).
OPAQUE = {
    "mlx_array",
    "mlx_device",
    "mlx_device_info",
    "mlx_stream",
    "mlx_string",
    "mlx_vector_array",
    "mlx_vector_vector_array",
    "mlx_vector_int",
    "mlx_vector_string",
    "mlx_map_string_to_array",
    "mlx_map_string_to_string",
    "mlx_closure",
    "mlx_closure_kwargs",
    "mlx_closure_value_and_grad",
    "mlx_closure_custom",
    "mlx_closure_custom_jvp",
    "mlx_closure_custom_vmap",
    "mlx_distributed_group",
    "mlx_io_reader",
    "mlx_io_writer",
    "mlx_io_gguf",
    "mlx_node_namer",
    "mlx_fast_cuda_kernel_config",
    "mlx_fast_cuda_kernel",
    "mlx_fast_metal_kernel_config",
    "mlx_fast_metal_kernel",
    "mlx_function_exporter",
    "mlx_imported_function",
}

# Structs with a non-trivial field layout (field names/types, V side).
#
# Field widths must match the C definition exactly.  `value` in the optional
# structs is a 32-bit C `int`/`mlx_dtype` followed by a `bool`, so it is `i32`:
# declaring it as V's 64-bit `int` would give the struct the wrong size and move
# `has_value` to the wrong offset whenever V constructs or copies one.
STRUCT_FIELDS = {
    "mlx_map_string_to_array_iterator": "ctx voidptr\n\tmap_ctx voidptr",
    "mlx_map_string_to_string_iterator": "ctx voidptr\n\tmap_ctx voidptr",
    "mlx_optional_int": "value i32\n\thas_value bool",
    "mlx_optional_float": "value f32\n\thas_value bool",
    "mlx_optional_dtype": "value i32\n\thas_value bool",
}

# C enum types -> mapped to V `int` in signatures.
ENUMS = {
    "mlx_dtype",
    "mlx_device_type",
    "mlx_compile_mode",
    "mlx_fft_norm",
}

# Additional typedef'd types available in V.
EXTRA_TYPES = []


def map_type(ty: str, as_return: bool = False) -> str:
    ty = ty.strip()
    # pointer / array suffix
    ptr = 0
    while ty.endswith("*"):
        ty = ty[:-1].strip()
        ptr += 1
    # strip qualifiers
    base = ty.replace("const ", "").replace(" ", "")
    if base == "char" and ptr >= 1:
        # const char* -> &char ; char** -> &&char
        return "&" * ptr + "char"
    if base == "void":
        return "voidptr" if ptr == 1 else ("&voidptr" if ptr == 2 else "voidptr")
    if base in SCALAR:
        v = SCALAR[base]
        # C `int` is 32-bit, V's `int` is 64-bit.  For a *parameter* V's C-ABI
        # shim converts a `[]int`/`&int` argument into a real `int*` for the
        # call, so `&int` is safe and keeps call sites natural.  For a *return*
        # value V binds the raw pointer with no conversion, so declaring an
        # `int*` as `&int` makes every read 8 bytes wide at a 4-byte stride --
        # e.g. shape (2, 2) came back as [8589934594, <garbage>].
        if ptr >= 1 and as_return and v == "int":
            v = "i32"
    elif base in OPAQUE or base in ENUMS or base in EXTRA_TYPES:
        v = "int" if base in ENUMS else "C." + base
    elif base in STRUCT_FIELDS:
        v = "C." + base
    else:
        raise ValueError(f"unmapped C type: {ty!r} (base {base!r}, ptr {ptr})")
    return "&" * ptr + v


# Functions to skip (varargs, function-pointer params, complex/half types, etc.).
SKIP = {
    # error.h
    "mlx_set_error_handler",
    "_mlx_error",
    # io_types.h (mlx_io_vtable is a struct of function pointers)
    "mlx_io_reader_new",
    "mlx_io_writer_new",
    # closure.h / function-pointer constructors
    "mlx_closure_new_func",
    "mlx_closure_new_func_payload",
    "mlx_closure_new_unary",
    "mlx_closure_kwargs_new_func",
    "mlx_closure_kwargs_new_func_payload",
    "mlx_closure_value_and_grad_new_func",
    "mlx_closure_value_and_grad_new_func_payload",
    "mlx_closure_custom_new_func",
    "mlx_closure_custom_new_func_payload",
    "mlx_closure_custom_jvp_new_func",
    "mlx_closure_custom_jvp_new_func_payload",
    "mlx_closure_custom_vmap_new_func",
    "mlx_closure_custom_vmap_new_func_payload",
    # graph_utils.h (need a C FILE*; V's os module owns C.FILE)
    "mlx_export_to_dot",
    "mlx_print_graph",
    # array.h managed-data constructors (function-pointer dtor params)
    "mlx_array_new_data_managed",
    "mlx_array_new_data_managed_payload",
    # array.h complex/half accessors (need _Complex / __fp16 types)
    "mlx_array_item_complex64",
    "mlx_array_data_complex64",
    "mlx_array_item_float16",
    "mlx_array_item_bfloat16",
    "mlx_array_data_float16",
    "mlx_array_data_bfloat16",
}


def strip_comments(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.DOTALL)
    src = re.sub(r"//[^\n]*", "", src)
    return src


def strip_preprocessor(src: str) -> str:
    lines = []
    for ln in src.splitlines():
        s = ln.strip()
        if s.startswith("#"):
            continue
        if s == 'extern "C" {' or s == "}":
            continue
        lines.append(ln)
    return "\n".join(lines)


def parse_functions(src: str):
    """Yield (name, ret, params) for simple function prototypes."""
    src = strip_comments(src)
    src = strip_preprocessor(src)
    # remove typedef struct/enum blocks (they contain ';' and braces)
    src = re.sub(r"typedef\s+struct\s+\{[^}]*\}\s*[^;]*;", "", src, flags=re.DOTALL)
    src = re.sub(r"typedef\s+enum\s+\{[^}]*\}\s*[^;]*;", "", src, flags=re.DOTALL)
    # join multi-line prototypes: a prototype ends with ');'
    # Split on ';' then filter chunks containing '(' and 'mlx_'
    decls = re.split(r";", src)
    for d in decls:
        d = d.strip()
        if "(" not in d or ")" not in d:
            continue
        if "mlx_" not in d and not d.startswith("mlx"):
            continue
        d = re.sub(r"\s+", " ", d)
        d = d.strip()
        # find the function name (identifier right before '(')
        m = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", d)
        if not m:
            continue
        name = m.group(1)
        if not name.startswith(("mlx", "_mlx")):
            continue
        if name in SKIP:
            continue
        # return type = everything before the name
        ret = d[: m.start()].strip()
        # params = inside parens
        params_raw = d[m.end() : d.rfind(")")]
        yield name, ret, params_raw


def map_params(params_raw: str):
    if params_raw.strip() == "" or params_raw.strip() == "void":
        return []
    # split on top-level commas (no nested parens/brackets in these headers)
    parts = [p.strip() for p in params_raw.split(",")]
    out = []
    for p in parts:
        if not p:
            continue
        # each part looks like: "const int* shape" or "mlx_array* res"
        # last token = param name
        toks = p.split()
        name = toks[-1]
        if name in RESERVED:
            name = name + "_"
        ty = " ".join(toks[:-1])
        out.append((name, map_type(ty)))
    return out


def build():
    structs = set(OPAQUE)
    for s in STRUCT_FIELDS:
        structs.add(s)
    funcs = []  # (name, ret_v, [(pname, ptype)])
    for h in HEADERS:
        path = os.path.join(INC, h)
        with open(path) as f:
            src = f.read()
        for name, ret, params in parse_functions(src):
            ret = ret.replace("const ", "").strip()
            if ret == "void":
                ret_v = ""
            else:
                try:
                    ret_v = map_type(ret, as_return=True) if ret else ""
                except ValueError:
                    ret_v = None
                    print(f"  skip {name}: unmapped return {ret!r}", file=sys.stderr)
                    continue
            try:
                ps = map_params(params)
            except ValueError as e:
                print(f"  skip {name}: {e}", file=sys.stderr)
                continue
            funcs.append((name, ret_v, ps))

    # dedupe by name, keep first
    seen = set()
    uniq = []
    for f in funcs:
        if f[0] in seen:
            continue
        seen.add(f[0])
        uniq.append(f)

    lines = []
    lines.append("// cdefs.v — raw mlx-c bindings.")
    lines.append("// Auto-generated by gen/gen_cdefs.py. Do not edit manually.")
    lines.append("")
    lines.append("module mlx")
    lines.append("")
    # Wrapped (not shortened) so the emitted comment stays byte-identical.
    lines.append(
        "// C struct typedefs (all opaque handles share a single `void* ctx`)."
    )
    lines.append("")
    for s in sorted(structs):
        if s in STRUCT_FIELDS:
            flds = [fl.strip() for fl in STRUCT_FIELDS[s].split("\n")]
            # Space-align the type column the way `v fmt` does, so that
            # regenerating cdefs.v does not reflow every struct.
            width = max(len(f.split(None, 1)[0]) for f in flds)
            lines.append("@[typedef]")
            lines.append(f"struct C.{s} {{")
            for fl in flds:
                name, ty = fl.split(None, 1)
                lines.append(f"\t{name.ljust(width)} {ty}")
            lines.append("}")
        else:
            lines.append("@[typedef]")
            lines.append(f"struct C.{s} {{")
            lines.append("\tctx voidptr")
            lines.append("}")
        lines.append("")

    lines.append("// C function declarations.")
    lines.append("")
    for name, ret_v, ps in uniq:
        plist = ", ".join(f"{pn} {pt}" for pn, pt in ps)
        if ret_v:
            lines.append(f"fn C.{name}({plist}) {ret_v}")
        else:
            lines.append(f"fn C.{name}({plist})")
    lines.append("")

    with open(OUT, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {OUT}: {len(uniq)} functions, {len(structs)} structs")


if __name__ == "__main__":
    build()
