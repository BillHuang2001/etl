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


def _export(fn, *specs, options=None):
    """Trace ``fn`` over ``specs`` and export the graph as StableHLO MLIR text."""
    return etl.backends.stablehlo.export(etl.trace(fn, *specs), options=options)


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
    ("runtime_call", _fn_runtime_call, (etl.TensorSpec((2,), etl.float32),),
     "runtime_call"),
    ("block_call", _fn_block_call, (etl.TensorSpec((2,), etl.float32),),
     "block_call"),
    ("rank", _fn_rank, (etl.TensorSpec((2,), etl.float32),), "rank"),
    ("world_size", _fn_world_size, (etl.TensorSpec((2,), etl.float32),),
     "world_size"),
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


# --- 12. dynamic dims → BackendError at export time ----------------------------

# `Writer._reject_dynamic_dims` (etl/backends/stablehlo/writer.py) keeps
# invalid MLIR away from the compilers: reshape/conv/slice/pad with any
# symbolic dim, the keepdims reshapes inside the reduce emitters, and
# reduce_mean over a dynamic REDUCED dim all fail at export/lower time with a
# clear BackendError naming the op and the offending dims — never an obscure
# iree-compile parse error or a runtime abort (dynamic slice/pad were
# EMPIRICALLY verified through iree 3.11.0 to parse but ABORT at runtime:
# "hal.fence.await" failure).

_DYN = etl.dim("B")

DYNAMIC_DIM_REJECT_CASES = [
    ("reshape_operand",
     lambda x: etl.reshape(x, (3, 2)),
     (etl.TensorSpec((_DYN, 6), etl.float32),),
     "reshape", "operand shape has dynamic dims"),
    ("reshape_result",
     lambda x: etl.reshape(x, (_DYN, 3, 2)),
     (etl.TensorSpec((4, 6), etl.float32),),
     "reshape", "result shape has dynamic dims"),
    ("conv",
     lambda x, w: etl.conv(x, w),
     (etl.TensorSpec((_DYN, 3, 8, 8), etl.float32),
      etl.TensorSpec((4, 3, 3, 3), etl.float32)),
     "conv", "operand shape has dynamic dims"),
    ("slice",
     lambda x: etl.slice(x, (0, 0), (2, 4)),
     (etl.TensorSpec((_DYN, 4), etl.float32),),
     "slice", "operand shape has dynamic dims"),
    ("pad",
     lambda x: etl.pad(x, [[0, 0], [1, 1]]),
     (etl.TensorSpec((_DYN, 4), etl.float32),),
     "pad", "operand shape has dynamic dims"),
    ("reduce_mean_dynamic_reduced_dim",
     lambda x: etl.mean(x, axes=0),
     (etl.TensorSpec((_DYN, 4), etl.float32),),
     "reduce_mean", "reduces over dynamic dims"),
    ("keepdims_reduce_reshape",
     lambda x: etl.sum(x, axes=1, keepdims=True),
     (etl.TensorSpec((_DYN, 4), etl.float32),),
     "reshape", "keepdims reshape operand has dynamic dims"),
]


@pytest.mark.parametrize(
    "fn,specs,expected_op,fragment",
    [(fn, specs, expected_op, fragment)
     for _, fn, specs, expected_op, fragment in DYNAMIC_DIM_REJECT_CASES],
    ids=[case_id for case_id, _, _, _, _ in DYNAMIC_DIM_REJECT_CASES],
)
def test_dynamic_dims_rejected_at_export(fn, specs, expected_op, fragment):
    msg = _export_error(fn, *specs)
    assert f"op '{expected_op}'" in msg
    assert fragment in msg
    assert "dynamic" in msg


def test_dynamic_dims_rejection_message_format():
    # Exact wording of Writer._reject_dynamic_dims. The location sits
    # between the op name and the shape description, so only the trailing
    # part (which follows the location) is pinned verbatim.
    msg = _export_error(
        lambda x: etl.reshape(x, (3, 2)),
        etl.TensorSpec((_DYN, 6), etl.float32),
    )
    assert msg.endswith(
        "operand shape has dynamic dims (Dim('B'), 6) "
        "(offending dims (Dim('B'),)) — dynamic shapes are not supported "
        "by the StableHLO compiler backends in v1 (use concrete static "
        "shapes or the numpy backend)"
    )


def test_reduce_mean_dynamic_reduced_dim_message_format():
    # reduce_mean's rejection is its own message: dynamic REDUCED dims make
    # the element count uncomputable statically.
    msg = _export_error(
        lambda x: etl.mean(x, axes=0),
        etl.TensorSpec((_DYN, 4), etl.float32),
    )
    assert msg.endswith(
        "reduces over dynamic dims (Dim('B'),) of shape (Dim('B'), 4) — "
        "the element count cannot be computed statically; dynamic shapes "
        "are not supported by the StableHLO compiler backends in v1 "
        "(decompose it manually or use a future adapter)"
    )


def test_reduce_mean_dynamic_non_reduced_dim_still_exports():
    # Only the REDUCED dims must be static: a dynamic non-reduced dim keeps
    # working — the divide's scalar count broadcasts to the dynamic reduced
    # shape via dynamic_broadcast_in_dim + get_dimension_size.
    mlir = _export(
        lambda x: etl.mean(x, axes=1),
        etl.TensorSpec((_DYN, 4), etl.float32),
    )
    assert "-> tensor<?xf32>" in mlir
    assert "stablehlo.reduce" in mlir
    assert "stablehlo.get_dimension_size" in mlir
    assert "stablehlo.dynamic_broadcast_in_dim" in mlir
    assert "stablehlo.divide" in mlir


# --- 13. dot_general with mismatched batch ranks -------------------------------

# `Writer._emit_dot` must handle batched dots whose operands disagree on
# batch rank exactly like numpy matmul (right-aligned batch broadcast):
# rank-3@rank-2, rhs-higher-rank, size-1 batch squeeze, matched multi-batch,
# and symbolic-batch dynamic broadcast. Unbalanced batching dims (the pre-fix
# emission) produced invalid MLIR that iree-compile rejected; unprovable
# symbolic merges raise BackendError naming "dynamic".


def _fn_dot_batched(a, b):
    return etl.dot(a, b)


def test_dot_rank3_rank2_non_batched():
    # rhs is a plain matrix: non-batched dot_general whose lhs free dims ARE
    # a's batch dims — no broadcast materialization.
    mlir = _export(
        _fn_dot_batched,
        etl.TensorSpec((4, 512, 768), etl.float32),
        etl.TensorSpec((768, 2304), etl.float32),
    )
    assert "-> tensor<4x512x2304xf32>" in mlir
    assert (
        "lhs_batching_dimensions = [], rhs_batching_dimensions = [], "
        "lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [0]"
        in mlir
    )
    assert "stablehlo.broadcast_in_dim" not in mlir
    assert "stablehlo.reshape" not in mlir


def test_dot_rhs_higher_rank_lhs_broadcast():
    # (512,768) @ (4,768,2304): the lhs matrix broadcasts up to the batch
    # (4,512,768), then a matched batched dot_general runs.
    mlir = _export(
        _fn_dot_batched,
        etl.TensorSpec((512, 768), etl.float32),
        etl.TensorSpec((4, 768, 2304), etl.float32),
    )
    assert "-> tensor<4x512x2304xf32>" in mlir
    assert "stablehlo.broadcast_in_dim" in mlir
    assert (
        "lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], "
        "lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]"
        in mlir
    )


def test_dot_size1_rhs_batch_squeezed():
    # A provably size-1 rhs batch is reshaped to its plain matrix form —
    # non-batched dot_general, no broadcast materialized.
    mlir = _export(
        _fn_dot_batched,
        etl.TensorSpec((4, 512, 768), etl.float32),
        etl.TensorSpec((1, 768, 2304), etl.float32),
    )
    assert "-> tensor<4x512x2304xf32>" in mlir
    assert "stablehlo.reshape" in mlir
    assert "tensor<768x2304xf32>" in mlir
    assert (
        "lhs_batching_dimensions = [], rhs_batching_dimensions = [], "
        "lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [0]"
        in mlir
    )


def test_dot_size1_rhs_batch_dynamic_squeeze_skipped():
    # The size-1-batch squeeze path is gated on FULLY STATIC shapes (fix
    # 16ced05): with a dynamic lhs, (B,16,8) @ (1,8,4) must fall through to
    # the batched dynamic-broadcast path. The pre-fix emission — squeeze
    # reshape of the rhs plus a NON-batched dot_general whose lhs carries
    # the dynamic dim — made iree's import insert a `stablehlo.dynamic_
    # reshape` that its llvm-cpu pipeline cannot legalize
    # ("failed to legalize operation 'stablehlo.dynamic_reshape'").
    mlir = _export(
        _fn_dot_batched,
        etl.TensorSpec((_DYN, 16, 8), etl.float32),
        etl.TensorSpec((1, 8, 4), etl.float32),
    )
    assert "-> tensor<?x16x4xf32>" in mlir
    assert "stablehlo.dynamic_broadcast_in_dim" in mlir
    assert (
        "lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], "
        "lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]"
        in mlir
    )
    # The un-gated squeeze pattern is gone: no rhs matrix reshape
    # ((1,8,4) -> (8,4)) and no non-batched dot_general.
    assert ": (tensor<1x8x4xf32>) -> tensor<8x4xf32>" not in mlir
    assert "lhs_batching_dimensions = [], rhs_batching_dimensions = []," not in mlir


def test_dot_matched_multi_batch_direct():
    # Matched batch structure emits a direct batched dot_general with NO
    # broadcast/reshape ops — byte-identical to the pre-fix exporter output
    # for already-correct cases.
    mlir = _export(
        _fn_dot_batched,
        etl.TensorSpec((4, 8, 512, 768), etl.float32),
        etl.TensorSpec((4, 8, 768, 2304), etl.float32),
    )
    assert "-> tensor<4x8x512x2304xf32>" in mlir
    assert (
        "lhs_batching_dimensions = [0, 1], rhs_batching_dimensions = [0, 1], "
        "lhs_contracting_dimensions = [3], rhs_contracting_dimensions = [2]"
        in mlir
    )
    assert "stablehlo.broadcast_in_dim" not in mlir
    assert "stablehlo.reshape" not in mlir


def test_dot_symbolic_batch_dynamic_broadcast():
    # (B,512,768) @ (768,2304): the rhs matrix is broadcast up to the RUNTIME
    # batch via dynamic_broadcast_in_dim with output_dimensions built from
    # get_dimension_size on the lhs batch dim — one executable serves every
    # concrete B.
    mlir = _export(
        _fn_dot_batched,
        etl.TensorSpec((_DYN, 512, 768), etl.float32),
        etl.TensorSpec((768, 2304), etl.float32),
    )
    assert "-> tensor<?x512x2304xf32>" in mlir
    assert "stablehlo.get_dimension_size" in mlir
    assert "stablehlo.dynamic_broadcast_in_dim" in mlir
    assert (
        "lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], "
        "lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]"
        in mlir
    )


def test_dot_unprovable_symbolic_merge_rejected():
    # (4,512,768) @ (B,768,2304): the aligned batch pair (512, B) can be
    # neither proven equal nor size-1 at compile time — BackendError naming
    # the op and "dynamic", never invalid MLIR.
    msg = _export_error(
        _fn_dot_batched,
        etl.TensorSpec((4, 512, 768), etl.float32),
        etl.TensorSpec((_DYN, 768, 2304), etl.float32),
    )
    assert "op 'dot'" in msg
    assert "dynamic batch broadcast" in msg
    assert "symbolic dims that cannot be proven equal or size-1" in msg
    assert "dynamic" in msg


# --- 14. mixed-dtype operand equalization (regression for 6aec043) ----------

# etl IR allows mixed-dtype operands on binary elementwise ops / compare /
# select (the result dtype follows numpy promotion — e.g. a Python scalar
# like ``3`` becomes a scalar i64 weak constant at trace time). StableHLO
# requires every non-predicate operand AND the result to share one element
# type, so the writer equalizes dtypes by inserting ``stablehlo.convert``
# AFTER shape equalization (fix 6aec043): elementwise ops convert operands
# to the result dtype (skipped for ``cast``/unary — cast IS a convert),
# compare converts both operands to ``np.result_type(lhs, rhs)`` — the
# promoted NUMERIC dtype, never the i1 result — and select converts
# on_true/on_false to the promoted result dtype while the predicate stays
# i1. Matching-dtype graphs emit byte-identical MLIR (no converts).


def _line_containing(mlir, needle):
    """The first non-empty line of ``mlir`` containing ``needle``."""
    return next(line for line in mlir.splitlines() if needle in line)


def test_elementwise_mixed_dtype_operand_converted_to_result_dtype():
    # bitwise_and(cast(x, i32), 3) over f32: the cast emits its own
    # convert (f32 -> i32) while the Python scalar 3 promotes to a scalar
    # i64 constant, so the equalization convert is i32 -> i64 — the
    # promoted result dtype, distinct from the cast's convert.
    def f(x):
        return etl.bitwise_and(etl.cast(x, etl.int32), 3)

    mlir = _export(f, etl.TensorSpec((8, 16), etl.float32))
    assert "(tensor<8x16xf32>) -> tensor<8x16xi32>" in mlir  # the cast
    assert "(tensor<8x16xi32>) -> tensor<8x16xi64>" in mlir  # equalization
    and_line = _line_containing(mlir, "stablehlo.and")
    assert ": tensor<8x16xi64>" in and_line  # operands share the result type


def test_elementwise_mixed_dtype_only_mismatched_operand_converted():
    # add(x, 3) over i32: exactly ONE convert — only the i32 operand needs
    # equalization; the promoted i64 scalar constant is already the result
    # dtype. Pre-fix this graph emitted mixed-dtype stablehlo.add operands
    # that iree rejected at compile.
    def f(x):
        return etl.add(x, 3)

    mlir = _export(f, etl.TensorSpec((8, 16), etl.int32))
    assert mlir.count("stablehlo.convert") == 1
    assert "(tensor<8x16xi32>) -> tensor<8x16xi64>" in mlir
    add_line = _line_containing(mlir, "stablehlo.add")
    assert ": tensor<8x16xi64>" in add_line


def test_compare_mixed_dtype_operands_promoted_numeric_dtype():
    # equal(x, 3) over i32: both operands are converted to the promoted
    # NUMERIC dtype i64 (np.result_type(i32, i64)) — never to the i1
    # result type — so both operand slots render as tensor<8x16xi64>.
    def f(x):
        return etl.equal(x, 3)

    mlir = _export(f, etl.TensorSpec((8, 16), etl.int32))
    assert "(tensor<8x16xi64>, tensor<8x16xi64>) -> tensor<8x16xi1>" in mlir
    assert "(tensor<8x16xi32>) -> tensor<8x16xi64>" in mlir


def test_select_mixed_dtype_branches_converted_pred_stays_i1():
    # select(p, a, b) over bool/i32/i64: on_true is converted to the
    # promoted result dtype i64 while the predicate keeps its i1 type.
    def f(p, a, b):
        return etl.select(p, a, b)

    mlir = _export(
        f,
        etl.TensorSpec((8, 16), etl.bool_),
        etl.TensorSpec((8, 16), etl.int32),
        etl.TensorSpec((8, 16), etl.int64),
    )
    assert (
        "(tensor<8x16xi1>, tensor<8x16xi64>, tensor<8x16xi64>) -> tensor<8x16xi64>"
        in mlir
    )
    assert "(tensor<8x16xi32>) -> tensor<8x16xi64>" in mlir


NO_CONVERT_CASES = [
    ("add_f32", lambda x: etl.add(x, x),
     (etl.TensorSpec((8, 16), etl.float32),)),
    ("equal_i64_i64", lambda a, b: etl.equal(a, b),
     (etl.TensorSpec((8, 16), etl.int64), etl.TensorSpec((8, 16), etl.int64))),
    ("select_bool_i32_i32", lambda p, a, b: etl.select(p, a, b),
     (etl.TensorSpec((8, 16), etl.bool_),
      etl.TensorSpec((8, 16), etl.int32),
      etl.TensorSpec((8, 16), etl.int32))),
]


@pytest.mark.parametrize(
    "fn,specs",
    [(fn, specs) for _, fn, specs in NO_CONVERT_CASES],
    ids=[case_id for case_id, _, _ in NO_CONVERT_CASES],
)
def test_matching_dtype_graphs_emit_no_convert(fn, specs):
    # Negative guard: when every operand already shares the result dtype,
    # the exporter emits byte-identical MLIR — no stablehlo.convert is
    # inserted (these graphs contain no casts at all). Covers all three op
    # families: elementwise / compare / select.
    mlir = _export(fn, *specs)
    assert "stablehlo.convert" not in mlir


# --- 15. eigh / diag export compositions --------------------------------------

# eigh/diag have no mnemonic StableHLO op (StableHLO 1.0 removed
# eigh/qr/svd; there is no diag), so the writer emits multi-op compositions
# (SPECIAL_EMITTERS "eigh"/"diag"): eigh = adaptive unrolled cyclic-Jacobi
# sweeps (fp32: dim-based 3-7; f64: 8 — see writer.py `_eigh_sweeps`)
# (slice/iota/compare/select/elementwise rotations) + a stable pair-sort for
# the ascending eigenvalue order + a gather column reorder of V; diag =
# iota-EQ mask + select (rank-1 → diagonal matrix) or flatten +
# constant-index gather (rank-2 → main diagonal). These tests pin the
# composition markers, the dtype rules, and the dynamic-dims rejection;
# iree-llvm-cpu round-trip correctness lives in
# `test_iree_eigh_diag_parity.py`.


def _fn_eigh(x):
    w, v = etl.eigh(x)
    return w, v


def test_eigh_exports_jacobi_composition():
    mlir = _export(_fn_eigh, etl.TensorSpec((3, 3), etl.float32))
    # The ascending-order reorder is a stable pair-sort of the final
    # diagonal; the diagonal extract + the V column reorder are gathers;
    # the rotations are slice-based (no while-loops — unrolled sweeps).
    assert "stablehlo.sort" in mlir
    assert "stablehlo.gather" in mlir
    assert "stablehlo.slice" in mlir
    assert "stablehlo.iota" in mlir
    # (w, v) result types: w (..., n), v (..., n, n).
    assert "-> (tensor<3xf32>, tensor<3x3xf32>)" in mlir


def test_eigh_int_input_upcasts_to_f64():
    # numpy linalg upcast rule: int/bool → float64 (mirrors the numpy
    # kernel). The convert is emitted up front; the composition runs in f64.
    mlir = _export(_fn_eigh, etl.TensorSpec((3, 3), etl.int32))
    assert "stablehlo.convert" in mlir
    assert "-> (tensor<3xf64>, tensor<3x3xf64>)" in mlir


@pytest.mark.parametrize(
    "spec,fragment",
    [
        (etl.TensorSpec((3, 3), etl.float16), "float16"),
        (etl.TensorSpec((3, 3), etl.complex64), "complex"),
    ],
    ids=["f16", "complex"],
)
def test_eigh_unsupported_dtype_deferred_naming_op(spec, fragment):
    msg = _export_error(_fn_eigh, spec)
    assert "op 'eigh'" in msg
    assert "not supported in v1" in msg
    assert fragment in msg


def test_eigh_dynamic_dims_deferred_naming_op():
    msg = _export_error(_fn_eigh, etl.TensorSpec((_DYN, 3, 3), etl.float32))
    assert "op 'eigh'" in msg
    assert "dynamic" in msg
    assert "not supported" in msg


def test_diag_rank1_exports_iota_select_composition():
    # rank-1 (n,) → the (n, n) diagonal matrix: iota-EQ mask + select over
    # a broadcast of the input — no gather.
    mlir = _export(lambda x: etl.diag(x), etl.TensorSpec((3,), etl.float32))
    assert "stablehlo.select" in mlir
    assert "stablehlo.iota" in mlir
    assert "stablehlo.gather" not in mlir
    assert "-> tensor<3x3xf32>" in mlir


def test_diag_rank2_exports_gather_composition():
    # rank-2 (m, n) → the main diagonal (min(m, n),): flatten reshape +
    # constant-index gather — no select.
    mlir = _export(lambda x: etl.diag(x), etl.TensorSpec((2, 3), etl.float32))
    assert "stablehlo.gather" in mlir
    assert "stablehlo.reshape" in mlir
    assert "stablehlo.select" not in mlir
    assert "-> tensor<2xf32>" in mlir


def test_diag_complex_dtype_preserved():
    # diag preserves the input dtype incl. complex in both directions (the
    # mask compares iotas, never data) — no convert is inserted.
    mlir = _export(lambda x: etl.diag(x), etl.TensorSpec((3,), etl.complex64))
    assert "stablehlo.select" in mlir
    assert "stablehlo.convert" not in mlir
    assert "-> tensor<3x3xcomplex<f32>>" in mlir


def test_diag_dynamic_dims_deferred_naming_op():
    msg = _export_error(lambda x: etl.diag(x), etl.TensorSpec((_DYN,), etl.float32))
    assert "op 'diag'" in msg
    assert "dynamic" in msg
    assert "not supported" in msg


# --- 16. algorithm-aware random export --------------------------------------

# Multi-algorithm RNG lowering (workstream B, commit 119854b): the
# threefry2x32 / philox4x32_10 random ops carry a static ``algorithm``
# attribute; the exporter lowers them as INLINE bit-exact i32/ui32
# elementwise subgraphs by default, or — when the algorithm is in the
# writer's ``rng_bit_generator`` option (a bool — ``True`` → both ciphers,
# backward compat — or a collection of algorithm names; the native
# emission is selected PER-ALGORITHM iff the name is in the set, so a
# mixed module can keep one cipher inline while the other goes native) —
# as a NATIVE ``stablehlo.rng_bit_generator`` call with the verified
# state layout ``[key0, key1, ctr...]`` (key words FIRST, counter words
# zero). splitmix64 (the default, and the only pre-workstream-B layout)
# stays a pure i64 inline expansion regardless of the option. These tests
# pin the emission SHAPES (state layouts, enum spellings, salt folds,
# inline composition markers); the bit-exact behavior pins live in
# `test_iree_emitters_parity.py` / `tests/ops/test_random.py`. NOTE: the
# native PHILOX emission is export-only — iree's legalization of PHILOX
# fails, so that graph is never run on a compiler backend.

from etl.backends.stablehlo import random_export as _random_export

_THREEFRY_KEY_SPEC = etl.TensorSpec((2,), etl.int32)
_PHILOX_KEY_SPEC = etl.TensorSpec((4,), etl.int32)
_SPLITMIX_KEY_SPEC = etl.TensorSpec((), etl.int64)


def _fn_uniform_threefry(k):
    return etl.random.uniform(k, (4,))


def _fn_uniform_philox(k):
    return etl.random.uniform(k, (4,))


def _fn_uniform_splitmix(k):
    return etl.random.uniform(k, (4,))


def _signed_i32(word):
    """A 32-bit word as the exporter's two's-complement i32 literal."""
    return int(np.uint32(word).view(np.int32))


def test_threefry_native_rng_bit_generator_golden():
    mlir = _export(
        _fn_uniform_threefry, _THREEFRY_KEY_SPEC,
        options={"rng_bit_generator": True},
    )
    # Salt-folded key: uniform's pi-hex salt [slo, shi] as a 2-word i32
    # constant, xor'd with the (2,) i32 key (the key words come FIRST in
    # the state; the counter words are zero).
    slo, shi = _random_export._salt_words(_random_export._SALTS["uniform"])
    salt_text = f"dense<[{_signed_i32(slo)}, {_signed_i32(shi)}]>"
    salt_line = _line_containing(mlir, salt_text)
    assert ": tensor<2xi32>" in salt_line
    xor_line = _line_containing(mlir, "stablehlo.xor %arg0")
    assert ": tensor<2xi32>" in xor_line
    # The folded key converts bit-preservingly to ui32; the zero counter
    # words are a scalar-constant broadcast to (2,).
    assert "(tensor<2xi32>) -> tensor<2xui32>" in mlir
    assert "dense<0> : tensor<ui32>" in mlir
    assert "(tensor<ui32>) -> tensor<2xui32>" in mlir
    # State = concatenate(key' (2,), zeros (2,)) → tensor<4xui32>.
    concat_line = _line_containing(mlir, '"stablehlo.concatenate"')
    assert "dimension = 0 : i64" in concat_line
    assert "(tensor<2xui32>, tensor<2xui32>) -> tensor<4xui32>" in concat_line
    # The native call: TWO results (state_out, words), enum WITHOUT the
    # RNG_ALG_ prefix, no shape operand (the output length comes from the
    # result type).
    rng_line = _line_containing(mlir, "stablehlo.rng_bit_generator")
    assert "algorithm = THREE_FRY" in rng_line
    assert "(tensor<4xui32>) -> (tensor<4xui32>, tensor<4xui32>)" in rng_line
    assert "shape" not in rng_line
    assert "RNG_ALG_" not in mlir
    # Final bit-preserving out convert: ui32 → i32.
    assert "(tensor<4xui32>) -> tensor<4xi32>" in mlir


def test_philox_native_rng_bit_generator_golden():
    # EXPORT-ONLY golden: iree's legalization of PHILOX fails, so this
    # emission is never run on a compiler backend — it pins the verified
    # state layout [k0', k1', 0, 0, 0, 0] only.
    mlir = _export(
        _fn_uniform_philox, _PHILOX_KEY_SPEC,
        options={"rng_bit_generator": True},
    )
    # Salt fold: the [lo, hi, lo, hi] pattern on the (4,) i32 key.
    slo, shi = _random_export._salt_words(_random_export._SALTS["uniform"])
    salt_text = (
        f"dense<[{_signed_i32(slo)}, {_signed_i32(shi)}, "
        f"{_signed_i32(slo)}, {_signed_i32(shi)}]>"
    )
    salt_line = _line_containing(mlir, salt_text)
    assert ": tensor<4xi32>" in salt_line
    xor_line = _line_containing(mlir, "stablehlo.xor %arg0")
    assert ": tensor<4xi32>" in xor_line
    # Only the FIRST TWO folded key words feed the cipher: the 4-word
    # folded key is sliced down to (2,) before the state concat.
    assert "(tensor<4xi32>) -> tensor<2xi32>" in mlir
    assert "dense<0> : tensor<ui32>" in mlir
    assert "(tensor<ui32>) -> tensor<4xui32>" in mlir
    # State = concatenate(key' (2,), zeros (4,)) → tensor<6xui32>.
    concat_line = _line_containing(mlir, '"stablehlo.concatenate"')
    assert "(tensor<2xui32>, tensor<4xui32>) -> tensor<6xui32>" in concat_line
    rng_line = _line_containing(mlir, "stablehlo.rng_bit_generator")
    assert "algorithm = PHILOX" in rng_line
    assert "(tensor<6xui32>) -> (tensor<6xui32>, tensor<4xui32>)" in rng_line
    assert "shape" not in rng_line
    assert "RNG_ALG_" not in mlir
    assert "(tensor<4xui32>) -> tensor<4xi32>" in mlir


def _fn_uniform_mixed(tk, pk):
    """Threefry + philox uniform in ONE graph (two key inputs)."""
    a = etl.random.uniform(tk, (4,))
    b = etl.random.uniform(pk, (4,))
    return a, b


def test_mixed_module_set_option_native_threefry_inline_philox():
    # The per-algorithm SET form of the ``rng_bit_generator`` option: with
    # only ``threefry2x32`` in the set, the mixed module emits the native
    # call for the threefry op and the INLINE expansion for the philox op.
    mlir = _export(
        _fn_uniform_mixed, _THREEFRY_KEY_SPEC, _PHILOX_KEY_SPEC,
        options={"rng_bit_generator": {"threefry2x32"}},
    )
    assert mlir.count("stablehlo.rng_bit_generator") == 1
    rng_line = _line_containing(mlir, "stablehlo.rng_bit_generator")
    assert "algorithm = THREE_FRY" in rng_line
    # The native threefry state concat marker [k0', k1', 0, 0] → (4,).
    assert "(tensor<2xui32>, tensor<2xui32>) -> tensor<4xui32>" in mlir
    # The philox half stays inline: the i64 mulhilo chain marker.
    assert "(tensor<1xui32>) -> tensor<1xi64>" in mlir


def test_mixed_module_set_option_native_philox_inline_threefry():
    # The reverse selection: only ``philox4x32_10`` in the set — the philox
    # op goes native, the threefry op stays on the inline expansion.
    mlir = _export(
        _fn_uniform_mixed, _THREEFRY_KEY_SPEC, _PHILOX_KEY_SPEC,
        options={"rng_bit_generator": {"philox4x32_10"}},
    )
    assert mlir.count("stablehlo.rng_bit_generator") == 1
    rng_line = _line_containing(mlir, "stablehlo.rng_bit_generator")
    assert "algorithm = PHILOX" in rng_line
    # The native philox state concat marker [k0', k1', 0, 0, 0, 0] → (6,).
    assert "(tensor<2xui32>, tensor<4xui32>) -> tensor<6xui32>" in mlir
    # The threefry half stays inline: the rotl composition markers
    # (shift_left | shift_right_logical, or'd) — absent from philox.
    assert "stablehlo.shift_left" in mlir
    assert "stablehlo.or" in mlir


def test_threefry_inline_default_emission():
    # DEFAULT options (no options dict): no native call — the cipher is
    # expanded as the bit-exact inline i32 elementwise subgraph.
    mlir = _export(_fn_uniform_threefry, _THREEFRY_KEY_SPEC)
    assert "rng_bit_generator" not in mlir
    # Word-index counters: iota (M = ceil(4/2) = 2 blocks).
    assert "stablehlo.iota dim = 0 : tensor<2xi32>" in mlir
    # rotl = shift_left | shift_right_logical (bit-exact on the u32 bit
    # pattern) — the inline cipher's rotation composition.
    assert "stablehlo.shift_left" in mlir
    assert "stablehlo.shift_right_logical" in mlir
    assert "stablehlo.or" in mlir
    # A pure i32 expansion — no ui32 anywhere (unlike the native path).
    assert "ui32" not in mlir


def test_philox_inline_default_emission():
    mlir = _export(_fn_uniform_philox, _PHILOX_KEY_SPEC)
    assert "rng_bit_generator" not in mlir
    # iota (M = ceil(4/4) = 1 block), converted bit-preservingly to ui32.
    assert "stablehlo.iota dim = 0 : tensor<1xi32>" in mlir
    # The i64 mulhilo chain: ui32→i64 zero-extend convert, i64 multiply,
    # LOGICAL shift of the i64 product (the exact uint64 high half — the
    # emitter uses shift_right_logical, never shift_right_arithmetic),
    # truncating convert back to ui32.
    assert "(tensor<1xui32>) -> tensor<1xi64>" in mlir
    assert "stablehlo.multiply" in mlir
    assert "stablehlo.shift_right_logical" in mlir
    assert "stablehlo.shift_right_arithmetic" not in mlir
    # Key convert i32 → ui32 and the final word-stream convert ui32 → i32.
    assert "(tensor<4xi32>) -> tensor<4xui32>" in mlir
    assert "(tensor<4xui32>) -> tensor<4xi32>" in mlir


def test_inline_random_exports_are_deterministic():
    for fn, spec in [
        (_fn_uniform_threefry, _THREEFRY_KEY_SPEC),
        (_fn_uniform_philox, _PHILOX_KEY_SPEC),
    ]:
        graph = etl.trace(fn, spec)
        assert (
            etl.backends.stablehlo.export(graph)
            == etl.backends.stablehlo.export(graph)
        )


def test_splitmix64_inline_golden_and_option_insensitive():
    # splitmix64 (the default algorithm) is ALWAYS the classic i64 inline
    # expansion — the native rng_bit_generator option does not apply to
    # it (there is no stablehlo.rng_bit_generator mapping for splitmix64).
    default_mlir = _export(_fn_uniform_splitmix, _SPLITMIX_KEY_SPEC)
    assert "rng_bit_generator" not in default_mlir
    # word_i = mix3(seed + (i + 1) * GOLDEN): iota word-index counters.
    assert "stablehlo.iota dim = 0 : tensor<4xi64>" in default_mlir
    # The mix3 chain (documented as bit-exact on the uint64 bit pattern):
    # i64 multiply / xor / LOGICAL right shift — never arithmetic (the
    # uint64 logical shift is emitted directly as shift_right_logical).
    assert "stablehlo.multiply" in default_mlir
    assert "stablehlo.shift_right_logical" in default_mlir
    assert "stablehlo.shift_right_arithmetic" not in default_mlir
    # Salt fold: uniform's pi-hex salt (as a two's-complement i64)
    # xor'd with the rank-0 i64 key.
    salt = _random_export._SALTS["uniform"] & ((1 << 64) - 1)
    salt_signed = int(np.uint64(salt).view(np.int64))
    salt_line = _line_containing(default_mlir, f"dense<{salt_signed}>")
    assert ": tensor<i64>" in salt_line
    xor_line = _line_containing(default_mlir, "stablehlo.xor %arg0")
    assert ": tensor<i64>" in xor_line
    # The option changes nothing for splitmix64: byte-identical export —
    # in BOTH the legacy bool form and the per-algorithm SET form.
    optioned = _export(
        _fn_uniform_splitmix, _SPLITMIX_KEY_SPEC,
        options={"rng_bit_generator": True},
    )
    assert optioned == default_mlir
    assert "rng_bit_generator" not in optioned
    optioned_set = _export(
        _fn_uniform_splitmix, _SPLITMIX_KEY_SPEC,
        options={"rng_bit_generator": {"threefry2x32", "philox4x32_10"}},
    )
    assert optioned_set == default_mlir
    assert "rng_bit_generator" not in optioned_set


@pytest.mark.parametrize(
    "key_spec,options",
    [
        (_SPLITMIX_KEY_SPEC, None),
        (_THREEFRY_KEY_SPEC, None),
        (_PHILOX_KEY_SPEC, None),
        (_SPLITMIX_KEY_SPEC, {"rng_bit_generator": True}),
        (_THREEFRY_KEY_SPEC, {"rng_bit_generator": True}),
        (_PHILOX_KEY_SPEC, {"rng_bit_generator": True}),
    ],
    ids=[
        "splitmix64-default",
        "threefry2x32-default",
        "philox4x32_10-default",
        "splitmix64-native",
        "threefry2x32-native",
        "philox4x32_10-native",
    ],
)
def test_random_multinomial_deferred_for_all_algorithms(key_spec, options):
    # ``random_multinomial`` is the only random op still deferred in v1 —
    # for ALL algorithms (the existing splitmix64-only pin lives in
    # tests/ops/test_random.py; this covers the new key layouts AND the
    # native-emission option, which must not silently enable it).
    def f(key):
        probs = etl.constant(
            etl.tensor(np.array([0.25, 0.25, 0.5], dtype=np.float32))
        )
        return etl.random.multinomial(key, probs, 5)

    graph = etl.trace(f, key_spec)
    with pytest.raises(etl.BackendError) as excinfo:
        etl.backends.stablehlo.export(graph, options=options)
    msg = str(excinfo.value)
    assert "op 'random_multinomial'" in msg
    assert "not supported in v1" in msg
