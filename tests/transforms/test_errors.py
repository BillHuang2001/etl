"""Error paths of the graph transforms.

v1 restrictions and invalid arguments must fail explicitly with the
contract-mandated error types — `TransformError` (missing rules, axis
mismatches), `ShapeError` (grad of non-scalar output, bad None-cotangents),
`TypeError` (wrong argument kinds), `TraceError` (concrete tensors fed to a
`TransformCallable`) — never with silent fallbacks. Boolean/int-producing ops
(`argmax`, comparisons, ...) are the documented exception: their builtin
rules yield `ZeroTangent`, so gradients through them are zero, NOT an error
(see `./etl/transforms/CONTEXT.md` "AD semantics" and "v1 scope vs deferred").
"""

import numpy as np
import pytest

import etl

SPEC = etl.TensorSpec((4, 3), etl.float32)
BATCHED_SPEC = etl.TensorSpec((8, 4, 3), etl.float32)


def run_graph(graph, *args):
    """Explicit staging: trace→lower→compile→load→run."""
    return etl.run(etl.load(etl.compile(etl.lower(graph))), *args)


def f_scale(x):
    return x * 2.0


def f_two(x, y):
    return x + y


def f_static(x, c):
    return x * c


def f_rc(x):
    """Scalar output using a runtime callback (Python escape hatch — no AD or
    batching rules; transforms must reject it by name)."""
    return etl.sum(
        etl.runtime_call(lambda t: t * 2, x, result=etl.TensorSpec((4, 3), etl.float32))
        + x
    )


def f_all_reduce(x):
    return etl.sum(etl.dist.all_reduce(x, op="sum"))


def f_sq(x):
    return etl.sum(x * x)


def f_multi(x):
    return x * 2, x + 1


class TestVectorizeAxisErrors:
    """`vectorize(graph, axes)` validates axes against the graph inputs."""

    def test_nonzero_mapped_axis_rejected(self):
        graph = etl.trace(f_scale, SPEC)
        with pytest.raises(
            etl.TransformError, match=r"must be 0 \(the leading axis\) in v1"
        ):
            etl.vectorize(graph, 1)

    def test_mapped_axis_out_of_range(self):
        graph = etl.trace(f_scale, SPEC)  # rank 2
        with pytest.raises(etl.TransformError, match="out of range"):
            etl.vectorize(graph, 2)

    def test_axes_pytree_structure_mismatch(self):
        graph = etl.trace(f_two, SPEC, SPEC)
        with pytest.raises(
            etl.TransformError, match="does not match the graph's input structure"
        ):
            etl.vectorize(graph, (0, 0, 0))

    def test_static_leaf_mapped(self):
        graph = etl.trace(f_static, SPEC, 2.0)
        with pytest.raises(etl.TransformError, match="static leaf"):
            etl.vectorize(graph, (0, 0))

    def test_bare_int_with_multiple_tensor_inputs(self):
        graph = etl.trace(f_two, SPEC, SPEC)
        with pytest.raises(etl.TransformError, match="exactly ONE tensor input"):
            etl.vectorize(graph, 0)

    @pytest.mark.parametrize(
        "bad",
        [f_scale, etl.defn(f_scale)],
        ids=["callable", "defn"],
    )
    def test_non_graph_raises_typeerror(self, bad):
        with pytest.raises(TypeError, match="etl.vmap"):
            etl.vectorize(bad, 0)


class TestVmapInAxesErrors:
    """`vmap(f, in_axes)` mirrors vectorize's axes validation (sugar over the
    same machinery)."""

    def test_nonzero_in_axes_rejected(self):
        tf = etl.vmap(f_scale, in_axes=1)
        with pytest.raises(
            etl.TransformError, match=r"must be 0 \(the leading axis\) in v1"
        ):
            tf(BATCHED_SPEC)

    def test_in_axes_out_of_range(self):
        tf = etl.vmap(f_scale, in_axes=3)  # batched spec rank 3
        with pytest.raises(etl.TransformError, match="out of range"):
            tf(BATCHED_SPEC)

    def test_in_axes_structure_mismatch(self):
        tf = etl.vmap(f_two, in_axes=(0, 0, 0))
        with pytest.raises(
            etl.TransformError, match="does not match the argument structure"
        ):
            tf(BATCHED_SPEC, SPEC)

    def test_static_leaf_mapped(self):
        tf = etl.vmap(f_static, in_axes=(0, 0))
        with pytest.raises(etl.TransformError, match="static leaf"):
            tf(BATCHED_SPEC, 2.0)

    def test_bare_in_axes_with_multiple_tensor_specs(self):
        tf = etl.vmap(f_two)  # default in_axes=0 is a bare entry
        with pytest.raises(etl.TransformError, match="exactly ONE tensor spec"):
            tf(BATCHED_SPEC, SPEC)


class TestNonDifferentiableOps:
    """Ops with NO rule (`runtime_call`, collectives) raise `TransformError`
    naming the op — never a silent fallback."""

    def test_grad_runtime_call(self):
        with pytest.raises(
            etl.TransformError, match=r"no VJP rule for op 'runtime_call'"
        ):
            etl.grad(f_rc)(SPEC)

    def test_grad_runtime_call_graph_form(self):
        graph = etl.trace(f_rc, SPEC)
        with pytest.raises(
            etl.TransformError, match=r"no VJP rule for op 'runtime_call'"
        ):
            etl.grad(graph)

    def test_jvp_runtime_call(self):
        with pytest.raises(
            etl.TransformError, match=r"no JVP rule for op 'runtime_call'"
        ):
            etl.jvp(f_rc, SPEC)(SPEC)

    def test_vjp_runtime_call(self):
        with pytest.raises(
            etl.TransformError, match=r"no VJP rule for op 'runtime_call'"
        ):
            etl.vjp(f_rc)(SPEC)

    def test_grad_collective(self):
        with pytest.raises(
            etl.TransformError, match=r"no VJP rule for op 'all_reduce'"
        ):
            etl.grad(f_all_reduce)(SPEC)

    def test_vmap_runtime_call(self):
        with pytest.raises(
            etl.TransformError, match=r"no batching rule for op 'runtime_call'"
        ):
            etl.vmap(f_rc)(BATCHED_SPEC)

    def test_vectorize_runtime_call_graph(self):
        graph = etl.trace(f_rc, SPEC)
        with pytest.raises(
            etl.TransformError, match=r"no batching rule for op 'runtime_call'"
        ):
            etl.vectorize(graph, 0)


class TestArgValidation:
    """grad/jvp/vjp argument validation errors."""

    def test_grad_argnums_out_of_range(self):
        with pytest.raises(etl.TransformError, match="out of range"):
            etl.grad(f_sq, argnums=1)(SPEC)

    def test_grad_argnums_non_int(self):
        with pytest.raises(etl.TransformError, match="must be None, an int"):
            etl.grad(f_sq, argnums="x")(SPEC)

    @pytest.mark.parametrize("entry", [0.5, "a", True], ids=["float", "str", "bool"])
    def test_grad_argnums_bad_entries(self, entry):
        with pytest.raises(etl.TransformError, match="entries must be ints"):
            etl.grad(f_sq, argnums=(entry,))(SPEC)

    def test_grad_non_float_input(self):
        with pytest.raises(etl.TransformError, match="cannot be differentiated"):
            etl.grad(lambda x: etl.sum(x * 1))(etl.TensorSpec((4, 3), etl.int32))

    def test_jvp_tangent_shape_mismatch(self):
        with pytest.raises(etl.TransformError, match="does not match the primal shape"):
            etl.jvp(f_sq, etl.TensorSpec((5, 3), etl.float32))(SPEC)

    def test_jvp_tangent_dtype_mismatch(self):
        with pytest.raises(etl.TransformError, match="does not match the primal dtype"):
            etl.jvp(f_sq, etl.TensorSpec((4, 3), etl.int32))(SPEC)

    def test_jvp_tangent_structure_mismatch(self):
        with pytest.raises(
            etl.TransformError, match="the tangent structure does not match"
        ):
            etl.jvp(
                f_sq,
                (etl.TensorSpec((4, 3), etl.float32), etl.TensorSpec((4, 3), etl.float32)),
            )(SPEC)

    def test_jvp_tangent_invalid_leaf(self):
        with pytest.raises(
            etl.TransformError, match=r"entries must be a core.TensorSpec"
        ):
            etl.jvp(f_two, (etl.TensorSpec((4, 3), etl.float32), 42))(SPEC, SPEC)

    def test_vjp_cotangent_shape_mismatch(self):
        graph = etl.trace(f_sq, SPEC)
        with pytest.raises(etl.TransformError, match="does not match the primal shape"):
            etl.vjp(graph, etl.TensorSpec((2, 2), etl.float32))

    def test_vjp_cotangent_dtype_mismatch(self):
        graph = etl.trace(f_sq, SPEC)
        with pytest.raises(etl.TransformError, match="does not match the primal dtype"):
            etl.vjp(graph, etl.TensorSpec((), etl.int32))

    def test_vjp_cotangent_structure_mismatch(self):
        graph = etl.trace(f_sq, SPEC)
        with pytest.raises(
            etl.TransformError, match="the cotangent structure does not match"
        ):
            etl.vjp(
                graph,
                (etl.TensorSpec((), etl.float32), etl.TensorSpec((), etl.float32)),
            )

    def test_vjp_none_cotangent_on_nonscalar_output(self):
        with pytest.raises(etl.ShapeError, match="requires a scalar output"):
            etl.vjp(f_scale)(SPEC)

    def test_vjp_none_cotangent_on_multi_output(self):
        with pytest.raises(etl.ShapeError, match="requires exactly one tensor output"):
            etl.vjp(f_multi)(SPEC)

    def test_vjp_none_entry_cotangent_on_nonscalar_output(self):
        graph = etl.trace(f_scale, SPEC)
        with pytest.raises(etl.ShapeError, match="seeds a scalar-one"):
            etl.vjp(graph, (None,))

    def test_grad_nonscalar_output(self):
        with pytest.raises(etl.ShapeError, match="scalar"):
            etl.grad(f_scale)(SPEC)

    def test_grad_multi_output(self):
        with pytest.raises(etl.ShapeError, match="exactly one tensor output"):
            etl.grad(f_multi)(SPEC)


class TestTypeErrors:
    """Wrong argument kinds raise `TypeError`; concrete tensors fed to a
    `TransformCallable` raise `TraceError` (transforms never execute)."""

    @pytest.mark.parametrize(
        "transform,kwargs",
        [
            (etl.vmap, {}),
            (etl.grad, {}),
            (etl.jvp, {"tangents": None}),
            (etl.vjp, {}),
        ],
        ids=["vmap", "grad", "jvp", "vjp"],
    )
    def test_invalid_object_raises_typeerror(self, transform, kwargs):
        with pytest.raises(TypeError, match="expects an etl.Graph"):
            transform(42, **kwargs)

    @pytest.mark.parametrize(
        "tc",
        [
            etl.vmap(f_scale),
            etl.grad(f_sq),
            etl.jvp(f_sq, SPEC),
            etl.vjp(f_sq),
        ],
        ids=["vmap", "grad", "jvp", "vjp"],
    )
    def test_transform_callable_with_concrete_tensor(self, tc):
        concrete = etl.tensor(np.ones((4, 3), dtype=np.float32))
        with pytest.raises(etl.TraceError, match="never execute"):
            tc(concrete)


class TestZeroGradientContract:
    """Boolean/int-producing ops (argmax, comparisons, ...) have builtin rules
    yielding `ZeroTangent` — a zero gradient, documented, NOT an error."""

    def test_grad_through_argmax_is_zero(self):
        grad_fn = etl.grad(lambda x: etl.cast(etl.argmax(x), etl.float32), argnums=0)
        graph = grad_fn(etl.TensorSpec((3, 4), etl.float32))
        x = np.arange(12, dtype=np.float32).reshape(3, 4)
        result = run_graph(graph, x)
        assert result.dtype == etl.float32
        assert np.allclose(result.numpy(), 0.0)

    def test_grad_through_comparison_is_zero(self):
        grad_fn = etl.grad(
            lambda x: etl.sum(etl.cast(etl.equal(x, x), etl.float32)), argnums=0
        )
        graph = grad_fn(etl.TensorSpec((3, 4), etl.float32))
        x = np.arange(12, dtype=np.float32).reshape(3, 4)
        result = run_graph(graph, x)
        assert result.dtype == etl.float32
        assert np.allclose(result.numpy(), 0.0)


# ---------------------------------------------------------------------------
# external_call with no rule and no portable: the external:<name> key format
# ---------------------------------------------------------------------------


def _ext_none_graph():
    """A scalar-loss graph over an external_call with NO registered rule and
    NO portable decomposition (the kernel is never even registered)."""
    @etl.defn
    def f(x):
        return etl.sum(
            etl.external_call(
                "tx_ext_nerr", x, result=etl.TensorSpec((4, 3), etl.float32)
            )
        )

    return etl.trace(f, SPEC)


class TestExternalCallRuleErrors:
    """external_call with neither an explicit rule nor a portable
    decomposition raises `TransformError` naming the op AND the
    `external:<name>` key (the namespaced message format), hinting at the
    external-kernel handle — never a silent fallback."""

    def test_grad_external_call_names_op_and_key(self):
        with pytest.raises(
            etl.TransformError,
            match=(
                r"no VJP rule for op 'external_call' "
                r"\(key 'external:tx_ext_nerr'\)"
            ),
        ) as exc:
            etl.grad(_ext_none_graph())
        assert "etl.register_external_kernel('tx_ext_nerr', fn)" in str(exc.value)

    def test_vjp_external_call_names_op_and_key(self):
        with pytest.raises(
            etl.TransformError,
            match=(
                r"no VJP rule for op 'external_call' "
                r"\(key 'external:tx_ext_nerr'\)"
            ),
        ):
            etl.vjp(_ext_none_graph())

    def test_jvp_external_call_names_op_and_key(self):
        with pytest.raises(
            etl.TransformError,
            match=(
                r"no JVP rule for op 'external_call' "
                r"\(key 'external:tx_ext_nerr'\)"
            ),
        ):
            etl.jvp(
                _ext_none_graph(),
                tangents=(etl.TensorSpec((4, 3), etl.float32),),
            )

    def test_vmap_external_call_names_op_and_key(self):
        with pytest.raises(
            etl.TransformError,
            match=(
                r"no batching rule for op 'external_call' "
                r"\(key 'external:tx_ext_nerr'\)"
            ),
        ):
            etl.vmap(_ext_none_graph(), in_axes=0)

    def test_vectorize_external_call_names_op_and_key(self):
        with pytest.raises(
            etl.TransformError,
            match=(
                r"no batching rule for op 'external_call' "
                r"\(key 'external:tx_ext_nerr'\)"
            ),
        ):
            etl.vectorize(_ext_none_graph(), 0)
