"""StableHLO export tests (`etl.backends.stablehlo` — an export-only utility).

The exporter is NOT a registered Backend: it maps verified EvoXIR to
StableHLO MLIR text for external compilers (e.g. `iree-compile model.mlir`).
These tests pin the v1 mapping tables (`etl/backends/stablehlo/ops.py`) and
the emission conventions (`etl/backends/stablehlo/writer.py`): golden
mnemonics, MLIR type formatting, comparison attributes, decompositions,
program-order emission, determinism, symbolic dims, deferred-op rejection,
and input-shape validation (Graph vs ir.Module vs TypeError).

Deferred ops raise `core.BackendError` NAMING THE OP — never silently
skipped and never partial output.
"""

import numpy as np
import pytest

import etl

# --- shared fixtures -------------------------------------------------------

SPEC_2X3 = etl.TensorSpec((2, 3), etl.float32)

#: Structural substrings every export must contain (module shell, entry
#: function, return terminator).
SHELL_SUBSTRINGS = ("module {", "func.func @main", "func.return")


def _export(fn, *specs):
    """Trace ``fn`` over ``specs`` and export the graph as StableHLO MLIR text."""
    return etl.backends.stablehlo.export(etl.trace(fn, *specs))


def _export_error(fn, *specs):
    """Trace ``fn`` and expect a BackendError from export; return its message."""
    graph = etl.trace(fn, *specs)
    with pytest.raises(etl.BackendError) as excinfo:
        etl.backends.stablehlo.export(graph)
    return str(excinfo.value)


# --- 1. golden mnemonics: one graph per op group ---------------------------

def _fn_add(x):
    return etl.add(x, x)


def _fn_dot(a, b):
    return etl.dot(a, b)


def _fn_sum(x):
    return etl.sum(x)


def _fn_reshape(x):
    return etl.reshape(x, (3, 2))


def _fn_transpose(x):
    return etl.transpose(x, (1, 0))


def _fn_concatenate(a, b):
    return etl.concatenate([a, b], axis=0)


def _fn_broadcast(x):
    return etl.broadcast(x, (5, 2, 3))


def _fn_constant(x):
    w = etl.tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    return etl.add(x, etl.constant(w))


GOLDEN_CASES = [
    ("add", _fn_add, (SPEC_2X3,), "stablehlo.add"),
    ("dot", _fn_dot,
     (etl.TensorSpec((2, 3), etl.float32), etl.TensorSpec((3, 4), etl.float32)),
     "stablehlo.dot_general"),
    ("reduce", _fn_sum, (SPEC_2X3,), "stablehlo.reduce"),
    ("reshape", _fn_reshape, (SPEC_2X3,), "stablehlo.reshape"),
    ("transpose", _fn_transpose, (SPEC_2X3,), "stablehlo.transpose"),
    ("concatenate", _fn_concatenate,
     (etl.TensorSpec((2, 3), etl.float32), etl.TensorSpec((1, 3), etl.float32)),
     "stablehlo.concatenate"),
    ("broadcast", _fn_broadcast, (SPEC_2X3,), "stablehlo.broadcast_in_dim"),
    ("constant", _fn_constant, (etl.TensorSpec((3,), etl.float32),),
     "stablehlo.constant"),
]


@pytest.mark.parametrize(
    "fn,specs,mnemonic",
    [(fn, specs, mnemonic) for _, fn, specs, mnemonic in GOLDEN_CASES],
    ids=[case_id for case_id, _, _, _ in GOLDEN_CASES],
)
def test_golden_mnemonics(fn, specs, mnemonic):
    mlir = _export(fn, *specs)
    for substring in SHELL_SUBSTRINGS:
        assert substring in mlir
    assert mnemonic in mlir


# --- 2. dtype formatting ----------------------------------------------------

def test_f32_input_renders_as_tensor_2x3xf32():
    mlir = _export(_fn_add, SPEC_2X3)
    assert "tensor<2x3xf32>" in mlir


def test_i64_input_renders_as_tensor_2x3xi64():
    mlir = _export(_fn_add, etl.TensorSpec((2, 3), etl.int64))
    assert "tensor<2x3xi64>" in mlir


def test_bool_comparison_result_renders_as_i1():
    def f(a, b):
        return etl.less(a, b)

    mlir = _export(f, SPEC_2X3, SPEC_2X3)
    assert "-> tensor<2x3xi1>" in mlir


# --- 3. comparison_direction attribute rendering ----------------------------

_COMPARISONS = {
    "equal": (etl.equal, "EQ"),
    "not_equal": (etl.not_equal, "NE"),
    "less": (etl.less, "LT"),
    "less_equal": (etl.less_equal, "LE"),
    "greater": (etl.greater, "GT"),
    "greater_equal": (etl.greater_equal, "GE"),
}


@pytest.mark.parametrize(
    "op,direction",
    [(op, direction) for op, direction in _COMPARISONS.values()],
    ids=list(_COMPARISONS),
)
def test_comparison_direction_attribute(op, direction):
    def f(a, b):
        return op(a, b)

    mlir = _export(f, SPEC_2X3, SPEC_2X3)
    assert "stablehlo.compare" in mlir
    # The writer emits the generic op form with the exact attribute syntax:
    # {comparison_direction = #stablehlo<comparison_direction LT>} : (...)
    assert (
        f"comparison_direction = #stablehlo<comparison_direction {direction}>"
        in mlir
    )


# --- 4. multi-op graph: program order emission -----------------------------

def test_multiop_graph_emits_ops_in_program_order():
    def f(x):
        a = etl.add(x, x)
        b = etl.multiply(a, x)
        return etl.sum(b)

    mlir = _export(f, SPEC_2X3)
    i_add = mlir.index("stablehlo.add")
    i_multiply = mlir.index("stablehlo.multiply")
    i_reduce = mlir.index("stablehlo.reduce")
    assert i_add < i_multiply < i_reduce
    assert "func.return" in mlir


# --- 5. determinism ----------------------------------------------------------

def test_export_is_deterministic():
    def f(x):
        a = etl.add(x, x)
        return etl.multiply(a, x)

    graph = etl.trace(f, SPEC_2X3)
    assert (
        etl.backends.stablehlo.export(graph)
        == etl.backends.stablehlo.export(graph)
    )
    # A fresh trace of the same fn over the same specs emits identical text
    # (no caching, no nondeterministic naming/location leakage).
    fresh = etl.trace(f, SPEC_2X3)
    assert etl.backends.stablehlo.export(graph) == etl.backends.stablehlo.export(fresh)


# --- 6. symbolic dims ---------------------------------------------------------

def test_symbolic_dim_renders_as_unknown():
    mlir = _export(_fn_add, etl.TensorSpec((etl.dim("B"), 3), etl.float32))
    assert "tensor<?x3xf32>" in mlir


def test_none_dim_renders_as_unknown():
    mlir = _export(_fn_add, etl.TensorSpec((None, 3), etl.float32))
    assert "tensor<?x3xf32>" in mlir


# --- 7. deferred ops → BackendError naming the op -----------------------------

def _fn_gather(x, idx):
    return etl.gather(x, idx, axis=0)


def _fn_scatter(x, idx, updates):
    return etl.scatter(x, idx, updates, axis=0)


def _fn_scan(x):
    def step(carry, elem):
        return etl.add(carry, elem), etl.multiply(carry, elem)

    return etl.scan(step, 0.0, x)[1]


def _fn_runtime_call(x):
    return etl.runtime_call(np.add, x, x, result=etl.TensorSpec((2,), etl.float32))


# A bare block: declared, but with NO numpy impl and NO portable decomposition.
_BARE_BLOCK = etl.block(
    "stablehlo_export_bare_block",
    inputs=[etl.TensorSpec((2,), etl.float32)],
    outputs=[etl.TensorSpec((2,), etl.float32)],
)


def _fn_block_call(x):
    return _BARE_BLOCK(x)


def _fn_rank(x):
    return etl.dist.rank()


def _fn_world_size(x):
    return etl.dist.world_size()


def _fn_argmax(x):
    return etl.argmax(x)


def _fn_argmin(x):
    return etl.argmin(x)


def _fn_erf(x):
    return etl.erf(x)


def _fn_gelu(x):
    return etl.gelu(x)


def _fn_tril(x):
    return etl.tril(x)


def _fn_triu(x):
    return etl.triu(x)


def _fn_cumsum(x):
    return etl.cumsum(x)


def _fn_solve(a, b):
    return etl.solve(a, b)


DEFERRED_CASES = [
    ("gather", _fn_gather,
     (etl.TensorSpec((4, 3), etl.float32), etl.TensorSpec((2,), etl.int32)),
     "gather"),
    ("scatter", _fn_scatter,
     (etl.TensorSpec((4, 3), etl.float32), etl.TensorSpec((2,), etl.int32),
      etl.TensorSpec((2, 3), etl.float32)),
     "scatter"),
    # scan desugars at trace time into while + gather + scatter (see
    # etl/trace/control_flow.py); the writer walks op order and hits the
    # enclosing-level gather before the while op — hence 'gather' is named.
    ("scan", _fn_scan, (etl.TensorSpec((4,), etl.float32),), "gather"),
    ("runtime_call", _fn_runtime_call, (etl.TensorSpec((2,), etl.float32),),
     "runtime_call"),
    ("block_call", _fn_block_call, (etl.TensorSpec((2,), etl.float32),),
     "block_call"),
    ("rank", _fn_rank, (etl.TensorSpec((2,), etl.float32),), "rank"),
    ("world_size", _fn_world_size, (etl.TensorSpec((2,), etl.float32),),
     "world_size"),
    ("argmax", _fn_argmax, (SPEC_2X3,), "argmax"),
    ("argmin", _fn_argmin, (SPEC_2X3,), "argmin"),
    ("erf", _fn_erf, (SPEC_2X3,), "erf"),
    ("gelu", _fn_gelu, (SPEC_2X3,), "gelu"),
    ("tril", _fn_tril, (etl.TensorSpec((3, 3), etl.float32),), "tril"),
    ("triu", _fn_triu, (etl.TensorSpec((3, 3), etl.float32),), "triu"),
    ("cumsum", _fn_cumsum, (SPEC_2X3,), "cumsum"),
    ("solve", _fn_solve,
     (etl.TensorSpec((2, 2), etl.float32), etl.TensorSpec((2, 1), etl.float32)),
     "solve"),
]


@pytest.mark.parametrize(
    "fn,specs,expected_op",
    [(fn, specs, expected_op) for _, fn, specs, expected_op in DEFERRED_CASES],
    ids=[case_id for case_id, _, _, _ in DEFERRED_CASES],
)
def test_deferred_ops_raise_backend_error_naming_the_op(fn, specs, expected_op):
    msg = _export_error(fn, *specs)
    assert f"op '{expected_op}'" in msg
    assert "not supported in v1" in msg


def test_complex_reduction_is_deferred():
    # Complex-number computation beyond cast is deferred; the writer enforces
    # it on the reduce path (etl/backends/stablehlo/writer.py _emit_reduce).
    msg = _export_error(_fn_sum, etl.TensorSpec((2, 3), etl.complex64))
    assert "op 'reduce_sum'" in msg
    assert "complex" in msg


# --- 8. TypeError for non-graph inputs ----------------------------------------

@pytest.mark.parametrize(
    "bad_input",
    ["not a graph", 42, None, np.array([1.0, 2.0])],
    ids=["str", "int", "none", "ndarray"],
)
def test_export_rejects_non_graph_inputs(bad_input):
    with pytest.raises(TypeError):
        etl.backends.stablehlo.export(bad_input)


# --- 9. control flow -----------------------------------------------------------

def test_cond_exports_stablehlo_if():
    def f(x):
        def true_fn(y):
            return etl.add(y, 1.0)

        def false_fn(y):
            return etl.subtract(y, 1.0)

        return etl.cond(etl.less(x, 0.5), true_fn, false_fn, x)

    mlir = _export(f, etl.TensorSpec((), etl.float32))
    assert "stablehlo.if" in mlir
    assert "func.return" in mlir


def test_while_loop_exports_stablehlo_while():
    def f(x):
        def cond_fn(state):
            return etl.less(state, 10.0)

        def body_fn(state):
            return etl.add(state, 1.0)

        return etl.while_loop(cond_fn, body_fn, x)

    mlir = _export(f, etl.TensorSpec((), etl.float32))
    assert "stablehlo.while" in mlir
    assert "func.return" in mlir


# --- 10. constants and decompositions -------------------------------------------

def test_constant_embeds_dense_elements():
    mlir = _export(_fn_constant, etl.TensorSpec((3,), etl.float32))
    assert "stablehlo.constant" in mlir
    assert "dense<[1.0, 2.0, 3.0]>" in mlir


def test_square_decomposes_to_multiply():
    def f(x):
        return etl.square(x)

    mlir = _export(f, etl.TensorSpec((2,), etl.float32))
    assert "stablehlo.multiply" in mlir
    assert "stablehlo.square" not in mlir


def test_relu_decomposes_to_maximum():
    def f(x):
        return etl.relu(x)

    mlir = _export(f, etl.TensorSpec((2,), etl.float32))
    assert "stablehlo.maximum" in mlir
    assert "stablehlo.relu" not in mlir


def test_stop_gradient_passes_operand_through():
    def f(x):
        return etl.stop_gradient(x)

    mlir = _export(f, etl.TensorSpec((2,), etl.float32))
    assert "stablehlo.stop" not in mlir
    # The function body is just the passthrough return — no op is emitted.
    assert "func.return %arg0" in mlir


def test_reduce_mean_decomposes_to_reduce_then_divide():
    def f(x):
        return etl.mean(x)

    mlir = _export(f, SPEC_2X3)
    assert "stablehlo.reduce" in mlir
    assert "stablehlo.divide" in mlir


# --- 11. module input equivalence ------------------------------------------------

def test_export_accepts_module_directly():
    def f(x):
        a = etl.add(x, x)
        return etl.multiply(a, x)

    graph = etl.trace(f, SPEC_2X3)
    assert (
        etl.backends.stablehlo.export(graph.module)
        == etl.backends.stablehlo.export(graph)
    )


# --- collectives (direct 1:1 mnemonics) -------------------------------------------

COLLECTIVE_CASES = [
    ("all_reduce", lambda x: etl.dist.all_reduce(x, op="sum"),
     "stablehlo.all_reduce"),
    ("all_gather", lambda x: etl.dist.all_gather(x, axis=0),
     "stablehlo.all_gather"),
    ("reduce_scatter", lambda x: etl.dist.reduce_scatter(x, op="sum", axis=0),
     "stablehlo.reduce_scatter"),
    ("all_to_all", lambda x: etl.dist.all_to_all(x, split_axis=0, concat_axis=0),
     "stablehlo.all_to_all"),
    ("dist_broadcast", lambda x: etl.dist.broadcast(x, src_rank=0),
     "stablehlo.collective_broadcast"),
    ("collective_permute",
     lambda x: etl.dist.collective_permute(x, source_target_pairs=((0, 1),)),
     "stablehlo.collective_permute"),
]


@pytest.mark.parametrize(
    "fn,mnemonic",
    [(fn, mnemonic) for _, fn, mnemonic in COLLECTIVE_CASES],
    ids=[case_id for case_id, _, _ in COLLECTIVE_CASES],
)
def test_collectives_export_direct_mnemonics(fn, mnemonic):
    mlir = _export(fn, etl.TensorSpec((2,), etl.float32))
    assert mnemonic in mlir
