"""Contract tests for the external-kernel RULE surface (``etl.external``
handle API + per-backend registry + portable-decomposition transform rules).

Companion to ``test_external_call.py`` (the trace-time / run-time / error
contract). This file covers the round-2 surface documented in
``etl/CONTEXT.md`` ("External kernels"):

- ``etl.register_external_kernel`` returns an ``ExternalKernel`` handle
  (``.name``, decorator-friendly ``.impl`` / ``.portable`` / ``.batching_rule``
  / ``.vjp_rule`` / ``.jvp_rule``); the registry is PER-BACKEND (``backend=None``
  = the default slot).
- Resolution: exact backend slot → default slot → ``None``
  (``get_external_kernel(name, backend)``); ``get_external_kernel(name)``
  returns the default slot (backward compatible). The numpy backend dispatches
  through this resolution with backend name ``"numpy"``.
- ``unregister_external_kernel`` removes ALL slots and the portable, but NOT
  the ``external:<name>`` transform rules (graph-level, they survive).
- ``register_portable`` requires an ``@etl.defn`` function and pre-registers
  the decomposition fallback rules: the vjp fallback ALWAYS, the batching
  fallback only when no explicit batching rule exists; explicit rules
  registered later overwrite the fallbacks (last-wins).
- Transforms compose through the portable: vmap/vectorize re-trace the
  decomposition over batched operands; grad/vjp/jvp inline it and run a local
  reverse sweep (nested external_calls resolve through their own
  ``external:<name>`` rules); all-zero cotangents short-circuit to zeros.
- Missing rule AND missing portable → ``TransformError`` naming the
  ``external_call`` op and the ``external:<name>`` key — never a silent
  fallback.
"""
import numpy as np
import pytest

import etl
from etl import transforms

from tests.ops.conftest import run_numpy


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _run(graph, *args):
    """Explicit pipeline: lower -> compile -> load -> run (returns structure)."""
    return etl.run(etl.load(etl.compile(etl.lower(graph))), *args)


def _as_np(value):
    """Recursively convert ``etl.Tensor`` / containers to numpy values."""
    if isinstance(value, etl.Tensor):
        return value.numpy()
    if isinstance(value, (tuple, list)):
        return tuple(_as_np(v) for v in value)
    return np.asarray(value)


def _first(value):
    """Unwrap the 1-tuple ``grad`` returns for a single input with
    ``argnums=None`` (bare tensor for an int argnum)."""
    if isinstance(value, tuple) and len(value) == 1:
        return value[0]
    return value


def _unregister(name):
    """Remove the kernel slots + portable AND the transform rules the test
    registered under ``external:<name>`` (rules are graph-level registrations
    and survive ``unregister_external_kernel`` by design)."""
    etl.unregister_external_kernel(name)
    transforms.batching_rules.pop(f"external:{name}", None)
    transforms.vjp_rules.pop(f"external:{name}", None)
    transforms.jvp_rules.pop(f"external:{name}", None)


# ---------------------------------------------------------------------------
# Handle API (register_external_kernel return value)
# ---------------------------------------------------------------------------


def test_register_returns_handle_with_name():
    handle = etl.register_external_kernel("er_handle", lambda x: x)
    try:
        assert isinstance(handle, etl.ExternalKernel)
        assert handle.name == "er_handle"
        assert repr(handle) == "ExternalKernel(name='er_handle')"
        # Every registration returns a (fresh) handle for the same name.
        handle2 = etl.register_external_kernel("er_handle", lambda x: x + 1)
        assert isinstance(handle2, etl.ExternalKernel)
        assert handle2.name == "er_handle"
        assert handle2 is not handle
    finally:
        _unregister("er_handle")


def test_handle_impl_registers_backend_slot_direct_and_decorator_forms():
    handle = etl.register_external_kernel("er_impl", lambda x: x)

    @handle.impl("numpy")
    def numpy_impl(x):
        return x * 3

    def tvm_impl(x):
        return x * 4

    returned = handle.impl("tvm", tvm_impl)  # direct form returns fn
    try:
        assert returned is tvm_impl
        assert etl.get_external_kernel("er_impl", "numpy") is numpy_impl
        assert etl.get_external_kernel("er_impl", "tvm") is tvm_impl
        assert etl.get_external_kernel("er_impl") is not numpy_impl
        # Keyword form is accepted too.
        def kw_impl(x):
            return x * 5

        assert handle.impl(backend="numpy", fn=kw_impl) is kw_impl
        assert etl.get_external_kernel("er_impl", "numpy") is kw_impl
    finally:
        _unregister("er_impl")


# ---------------------------------------------------------------------------
# Per-backend registry: resolution order (exact slot -> default -> None)
# ---------------------------------------------------------------------------


def test_per_backend_registration_and_resolution():
    def default_k(x):
        return x

    def iree_k(x):
        return x * 10

    etl.register_external_kernel("er_pb", default_k)
    etl.register_external_kernel("er_pb", iree_k, backend="iree")
    try:
        # Exact backend slot wins over the default slot.
        assert etl.get_external_kernel("er_pb", "iree") is iree_k
        # No-backend call returns the default slot (backward compatible).
        assert etl.get_external_kernel("er_pb") is default_k
        assert etl.get_external_kernel("er_pb", None) is default_k
        # A backend with no exact slot falls back to the default slot.
        assert etl.get_external_kernel("er_pb", "numpy") is default_k
        assert etl.get_external_kernel("er_pb", "tvm") is default_k
        assert etl.get_external_kernel("er_pb", "no_such_backend") is default_k
    finally:
        _unregister("er_pb")
    # Unregister clears every slot.
    assert etl.get_external_kernel("er_pb") is None
    assert etl.get_external_kernel("er_pb", "iree") is None
    assert etl.get_external_kernel("er_pb", "numpy") is None


def test_get_unknown_name_returns_none():
    assert etl.get_external_kernel("er_never_registered") is None
    assert etl.get_external_kernel("er_never_registered", "numpy") is None
    assert etl.get_external_kernel("er_never_registered", "iree") is None


def test_no_default_slot_means_no_fallback():
    """Without a default slot, an unknown backend resolves to None (not to
    some other backend's slot)."""
    etl.register_external_kernel("er_nodefault", lambda x: x, backend="iree")
    try:
        assert etl.get_external_kernel("er_nodefault", "iree") is not None
        assert etl.get_external_kernel("er_nodefault", "numpy") is None
        assert etl.get_external_kernel("er_nodefault") is None
    finally:
        _unregister("er_nodefault")


def test_register_rejects_non_str_backend():
    with pytest.raises(TypeError, match="backend must be None or a str"):
        etl.register_external_kernel("er_badbackend", lambda x: x, backend=42)
    with pytest.raises(TypeError, match="backend must be None or a str"):
        etl.register_external_kernel(
            "er_badbackend", lambda x: x, backend=["numpy"]
        )


# ---------------------------------------------------------------------------
# Numpy dispatch resolution (the run-time path through get_external_kernel)
# ---------------------------------------------------------------------------


def test_numpy_dispatch_uses_backend_slot_when_registered():
    """A ``"numpy"``-slot kernel is what the numpy backend dispatches, even
    when a default slot exists."""
    def default_k(x):
        return x + 100

    def numpy_k(x):
        return x + 1

    etl.register_external_kernel("er_dispatch", default_k)
    etl.register_external_kernel("er_dispatch", numpy_k, backend="numpy")
    try:
        out = run_numpy(
            lambda a: etl.external_call(
                "er_dispatch", a, result=etl.TensorSpec((3,), etl.int64)
            ),
            np.array([1, 2, 3], dtype=np.int64),
        )
        np.testing.assert_array_equal(out, np.array([2, 3, 4], dtype=np.int64))
    finally:
        _unregister("er_dispatch")


def test_numpy_dispatch_default_slot_only_unchanged():
    """Only a default slot registered -> the numpy backend uses it (the
    original single-slot behavior)."""
    def default_k(x):
        return x * 2

    etl.register_external_kernel("er_dispatch_default", default_k)
    try:
        out = run_numpy(
            lambda a: etl.external_call(
                "er_dispatch_default", a, result=etl.TensorSpec((3,), etl.int64)
            ),
            np.array([1, 2, 3], dtype=np.int64),
        )
        np.testing.assert_array_equal(out, np.array([2, 4, 6], dtype=np.int64))
    finally:
        _unregister("er_dispatch_default")


# ---------------------------------------------------------------------------
# Unregister: all slots + portable removed; transform rules survive
# ---------------------------------------------------------------------------


def test_unregister_removes_all_slots_and_portable_but_keeps_rules():
    def k(x):
        return x

    @etl.defn
    def port(x):
        return x + 1.0

    handle = etl.register_external_kernel("er_unreg", k)
    etl.register_external_kernel("er_unreg", k, backend="iree")
    etl.register_external_kernel("er_unreg", k, backend="numpy")
    handle.portable(port)
    assert f"external:er_unreg" in transforms.vjp_rules
    assert f"external:er_unreg" in transforms.batching_rules
    vjp_fallback = transforms.vjp_rules["external:er_unreg"]

    etl.unregister_external_kernel("er_unreg")
    try:
        # Every kernel slot is gone...
        assert etl.get_external_kernel("er_unreg") is None
        assert etl.get_external_kernel("er_unreg", "numpy") is None
        assert etl.get_external_kernel("er_unreg", "iree") is None
        # ...and the portable is gone too.
        assert etl.get_portable("er_unreg") is None
        # The transform rules are graph-level registrations: they survive.
        assert transforms.vjp_rules["external:er_unreg"] is vjp_fallback
        assert "external:er_unreg" in transforms.batching_rules
        # The surviving rules still resolve (a re-registered portable would
        # replace them on the next register_portable call).
        etl.register_external_kernel("er_unreg", k)
        handle2 = etl.register_external_kernel("er_unreg", k)
        handle2.portable(port)
        assert transforms.vjp_rules["external:er_unreg"] is not vjp_fallback
    finally:
        _unregister("er_unreg")


def test_unregister_with_only_a_portable_registered():
    @etl.defn
    def port(x):
        return x + 1.0

    etl.register_portable("er_portonly", port)
    try:
        assert etl.get_portable("er_portonly") is port
        # No kernel slots -> resolution returns None.
        assert etl.get_external_kernel("er_portonly") is None
    finally:
        etl.unregister_external_kernel("er_portonly")
    assert etl.get_portable("er_portonly") is None


# ---------------------------------------------------------------------------
# Portable validation + fallback-rule installation
# ---------------------------------------------------------------------------


def test_portable_requires_defn_marker():
    with pytest.raises(TypeError, match="etl.defn"):
        etl.register_portable("er_badport", lambda x: x)

    handle = etl.register_external_kernel("er_badport", lambda x: x)
    try:
        with pytest.raises(TypeError, match="etl.defn"):
            handle.portable(lambda x: x)
    finally:
        _unregister("er_badport")

    with pytest.raises(TypeError, match="non-empty str"):
        etl.register_portable("", lambda x: x)


def test_portable_pre_registers_vjp_and_batching_fallbacks():
    @etl.defn
    def port(x):
        return x + 1.0

    handle = etl.register_external_kernel("er_fb", lambda x: x)
    handle.portable(port)
    try:
        assert "external:er_fb" in transforms.vjp_rules
        assert "external:er_fb" in transforms.batching_rules
        assert etl.get_portable("er_fb") is port
    finally:
        _unregister("er_fb")


def test_explicit_batching_rule_before_portable_blocks_the_fallback():
    """The batching fallback fills only a slot no explicit rule claimed: an
    explicit batching_rule registered BEFORE the portable survives."""
    handle = etl.register_external_kernel("er_fb_before", lambda x: x)
    explicit = lambda op, operands, axes: (operands, axes)
    handle.batching_rule(explicit)
    @etl.defn
    def port(x):
        return x + 1.0

    handle.portable(port)
    try:
        assert transforms.batching_rules["external:er_fb_before"] is explicit
        # The vjp fallback is ALWAYS pre-registered (no such guard there).
        assert "external:er_fb_before" in transforms.vjp_rules
    finally:
        _unregister("er_fb_before")


def test_explicit_rule_after_portable_overwrites_the_fallback():
    """Last-wins: explicit rules registered after portable() replace the
    pre-registered fallbacks in the public registries."""
    handle = etl.register_external_kernel("er_fb_after", lambda x: x)
    @etl.defn
    def port(x):
        return x + 1.0

    handle.portable(port)
    vjp_fallback = transforms.vjp_rules["external:er_fb_after"]
    batching_fallback = transforms.batching_rules["external:er_fb_after"]

    explicit_vjp = lambda op, cotangents, primals: (None,)
    explicit_batching = lambda op, operands, axes: (operands, axes)
    handle.vjp_rule(explicit_vjp)
    handle.batching_rule(explicit_batching)
    try:
        assert transforms.vjp_rules["external:er_fb_after"] is explicit_vjp
        assert transforms.batching_rules["external:er_fb_after"] is explicit_batching
        assert transforms.vjp_rules["external:er_fb_after"] is not vjp_fallback
        assert (
            transforms.batching_rules["external:er_fb_after"]
            is not batching_fallback
        )
    finally:
        _unregister("er_fb_after")


def test_explicit_vjp_rule_before_portable_is_overwritten_by_fallback():
    """The vjp fallback is ALWAYS pre-registered at portable() time — even an
    explicit vjp rule registered earlier is replaced (last-wins, documented)."""
    handle = etl.register_external_kernel("er_fb_vjpfirst", lambda x: x)
    explicit_vjp = lambda op, cotangents, primals: (None,)
    handle.vjp_rule(explicit_vjp)
    @etl.defn
    def port(x):
        return x + 1.0

    handle.portable(port)
    try:
        assert (
            transforms.vjp_rules["external:er_fb_vjpfirst"] is not explicit_vjp
        )
    finally:
        _unregister("er_fb_vjpfirst")


# ---------------------------------------------------------------------------
# End-to-end transforms through portable decompositions
# ---------------------------------------------------------------------------


@etl.defn
def _square_plus_portable(x):
    """Portable decomposition of ``k(x) = x*x + 1.5`` — includes a constant."""
    c = etl.constant(etl.tensor(np.float32(1.5)))
    return etl.multiply(x, x) + c


def _square_plus_ref(x):
    return x * x + 1.5


def test_vmap_through_portable_with_constant_matches_reference():
    """vmap of an external_call with only a portable == per-row run of the
    decomposition (the batching fallback re-traces it over batched operands;
    the constant is unmapped and threads through add)."""
    handle = etl.register_external_kernel("er_vmap_k", _square_plus_ref)
    handle.portable(_square_plus_portable)
    try:
        @etl.defn
        def f(x):
            return etl.external_call(
                "er_vmap_k", x, result=etl.TensorSpec((3,), etl.float32)
            )

        graph = etl.vmap(f)(etl.TensorSpec((4, 3), etl.float32))
        x = np.arange(12, dtype=np.float32).reshape(4, 3)
        out = _as_np(_run(graph, x))
        np.testing.assert_allclose(out, _square_plus_ref(x), rtol=1e-6)

        # vectorize (the non-sugar entry point) behaves identically.
        plain = etl.vectorize(etl.trace(f, etl.TensorSpec((3,), etl.float32)), 0)
        out_plain = _as_np(_run(plain, x))
        np.testing.assert_allclose(out_plain, _square_plus_ref(x), rtol=1e-6)
    finally:
        _unregister("er_vmap_k")


def test_grad_through_portable_matches_reference():
    handle = etl.register_external_kernel("er_grad_k", _square_plus_ref)
    handle.portable(_square_plus_portable)
    try:
        @etl.defn
        def loss(x):
            y = etl.external_call(
                "er_grad_k", x, result=etl.TensorSpec((3,), etl.float32)
            )
            return etl.sum(y)

        graph = etl.grad(loss)(etl.TensorSpec((3,), etl.float32))
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        out = _first(_as_np(_run(graph, x)))
        np.testing.assert_allclose(out, 2.0 * x, rtol=1e-6)  # d/dx sum(x²+1.5)
    finally:
        _unregister("er_grad_k")


def test_vjp_and_jvp_through_portable_match_reference():
    handle = etl.register_external_kernel("er_ad_k", _square_plus_ref)
    handle.portable(_square_plus_portable)
    try:
        @etl.defn
        def f(x):
            return etl.external_call(
                "er_ad_k", x, result=etl.TensorSpec((3,), etl.float32)
            )

        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        # vjp: primal outputs stay on the registered kernel path, the input
        # cotangents come from the portable's local reverse sweep.
        vj_graph = etl.vjp(
            f, cotangents=(etl.TensorSpec((3,), etl.float32),)
        )(etl.TensorSpec((3,), etl.float32))
        primal, (grad_x,) = _as_np(
            _run(
                vj_graph,
                (x,),
                (np.array([1.0, 1.0, 1.0], dtype=np.float32),),
            )
        )
        assert np.allclose(primal, _square_plus_ref(x), rtol=1e-6)
        np.testing.assert_allclose(grad_x, 2.0 * x, rtol=1e-6)

        # jvp: derived from the portable's vjp fallback via the adjoint trick.
        jv_graph = etl.jvp(
            f, tangents=(etl.TensorSpec((3,), etl.float32),)
        )(etl.TensorSpec((3,), etl.float32))
        primal_j, (tangent_out,) = _as_np(
            _run(jv_graph, (x,), (np.array([1.0, 0.0, 0.0], dtype=np.float32),))
        )
        assert np.allclose(primal_j, _square_plus_ref(x), rtol=1e-6)
        # tangent = J·[1,0,0] where J = diag(2x) -> [2x[0], 0, 0]
        np.testing.assert_allclose(
            tangent_out, np.array([2.0 * x[0], 0.0, 0.0], dtype=np.float32),
            rtol=1e-6,
        )
    finally:
        _unregister("er_ad_k")


def test_grad_through_portable_with_nested_external_call():
    """A portable may itself contain an external_call: its own
    ``external:<name>`` vjp rule (also a portable fallback) resolves during
    the local reverse sweep."""
    @etl.defn
    def inner_port(x):
        return etl.multiply(x, 2.0)

    inner = etl.register_external_kernel(
        "er_inner_k", lambda x: x * 2.0
    )
    inner.portable(inner_port)

    @etl.defn
    def outer_port(x):
        y = etl.external_call(
            "er_inner_k", x, result=etl.TensorSpec((3,), etl.float32)
        )
        return y + etl.multiply(x, x)

    outer = etl.register_external_kernel(
        "er_outer_k", lambda x: x * 2.0 + x * x
    )
    outer.portable(outer_port)
    try:
        @etl.defn
        def loss(x):
            y = etl.external_call(
                "er_outer_k", x, result=etl.TensorSpec((3,), etl.float32)
            )
            return etl.sum(y)

        graph = etl.grad(loss)(etl.TensorSpec((3,), etl.float32))
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        out = _first(_as_np(_run(graph, x)))
        # d/dx (2x + x²) = 2 + 2x
        np.testing.assert_allclose(out, 2.0 + 2.0 * x, rtol=1e-6)
    finally:
        _unregister("er_inner_k")
        _unregister("er_outer_k")


def test_zero_cotangent_short_circuit_through_portable():
    """grad of an output the external call does not influence: the portable's
    vjp fallback short-circuits to zero cotangents without inlining — the
    call's operands get zeros, the live path keeps its true gradient."""
    handle = etl.register_external_kernel("er_zc_k", lambda x: x * 2.0)
    handle.portable(_square_plus_portable)
    try:
        @etl.defn
        def loss(x, w):
            etl.external_call(
                "er_zc_k", x, result=etl.TensorSpec((3,), etl.float32)
            )
            return etl.sum(w * 5.0)

        graph = etl.grad(loss)(
            etl.TensorSpec((3,), etl.float32), etl.TensorSpec((3,), etl.float32)
        )
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        w = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        grad_x, grad_w = _as_np(_run(graph, x, w))
        np.testing.assert_array_equal(grad_x, np.zeros_like(x))
        np.testing.assert_allclose(grad_w, 5.0 * np.ones_like(w), rtol=1e-6)
    finally:
        _unregister("er_zc_k")


# ---------------------------------------------------------------------------
# TransformError paths: no rule and no portable
# ---------------------------------------------------------------------------


def _rule_less_graph(kernel_name):
    @etl.defn
    def f(x):
        y = etl.external_call(
            kernel_name, x, result=etl.TensorSpec((3,), etl.float32)
        )
        return etl.sum(y)

    return etl.trace(f, etl.TensorSpec((3,), etl.float32))


@pytest.mark.parametrize(
    "transform",
    ["vmap", "grad", "vjp", "jvp"],
    ids=["vmap", "grad", "vjp", "jvp"],
)
def test_transforms_without_rule_or_portable_raise_transform_error(transform):
    """No explicit rule AND no portable -> every transform raises
    TransformError naming the ``external_call`` op and the ``external:<name>``
    key, hinting the handle API — never a silent fallback."""
    name = "er_ruless"
    graph = _rule_less_graph(name)
    if transform == "vmap":
        with pytest.raises(etl.TransformError) as exc:
            etl.vmap(graph, in_axes=0)
    elif transform == "grad":
        with pytest.raises(etl.TransformError) as exc:
            etl.grad(graph)
    elif transform == "vjp":
        with pytest.raises(etl.TransformError) as exc:
            etl.vjp(graph)
    else:
        with pytest.raises(etl.TransformError) as exc:
            etl.jvp(
                graph, tangents=(etl.TensorSpec((3,), etl.float32),)
            )
    message = str(exc.value)
    assert "external_call" in message
    assert f"external:{name}" in message


def test_kernel_registration_alone_does_not_create_rules():
    """A registered kernel (no rules, no portable) must NOT make transforms
    work: registration is a run-time concern only."""
    def k(x):
        return x * 2.0

    etl.register_external_kernel("er_norule", k)
    try:
        assert f"external:er_norule" not in transforms.batching_rules
        assert f"external:er_norule" not in transforms.vjp_rules
        assert f"external:er_norule" not in transforms.jvp_rules
        with pytest.raises(etl.TransformError) as exc:
            etl.vmap(_rule_less_graph("er_norule"), in_axes=0)
        message = str(exc.value)
        assert "external_call" in message
        assert "external:er_norule" in message
    finally:
        _unregister("er_norule")


# ---------------------------------------------------------------------------
# Runtime registry still dispatches end-to-end (per-backend resolution)
# ---------------------------------------------------------------------------


def test_runtime_dispatch_unchanged_with_backend_slot_plus_downstream_ops():
    """The numpy run path (kernel output flows into downstream ops bit-exact)
    is unchanged by the per-backend registry: the ``"numpy"`` slot wins."""
    def numpy_k(x, w):
        return np.dot(x, w) * 2.0

    etl.register_external_kernel("er_e2e", numpy_k)
    etl.register_external_kernel("er_e2e", numpy_k, backend="numpy")
    try:
        @etl.defn
        def f(x, w):
            y = etl.external_call(
                "er_e2e", x, w, result=etl.TensorSpec((2, 3), etl.float32)
            )
            return etl.add(y, 1.0)

        x = np.arange(4, dtype=np.float32).reshape(2, 2)
        w = np.ones((2, 3), dtype=np.float32)
        out = run_numpy(f, x, w)
        np.testing.assert_array_equal(out, np.dot(x, w) * 2.0 + 1.0)
    finally:
        _unregister("er_e2e")
