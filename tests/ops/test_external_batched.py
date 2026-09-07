"""Contract tests for the batched-variant sugar on external kernels
(``ExternalKernel.batch_invariant`` / ``ExternalKernel.batch_variant``).

Assertions on the declarative surface documented in ``etl/external.py`` and
``etl/external_rules.py`` (the "External kernels" contract in
``etl/CONTEXT.md``):

- RESERVED derived-name suffix: ``_ETL_BATCH_SUFFIX == "__etl_batched"``.
  User-facing registrations — ``etl.register_external_kernel`` AND
  ``etl.register_portable`` — reject names containing it (``TypeError``
  naming the suffix); the internal derived registration path used by
  ``batch_variant`` is the only way into that namespace.
- ``batch_variant(fn=None, *, backend=None, device_resident=False)``: direct /
  decorator / decorator-with-args forms. Registers ``fn`` under the DERIVED
  name ``f"{name}__etl_batched"`` (same backend-slot + ``device_resident``
  semantics as ``impl`` — ``device_resident=True`` without an explicit
  backend still raises ``TypeError``). Installs the shared pass-through
  batching rule under BOTH ``external:<name>`` AND ``external:<derived>``.
- ``batch_invariant()``: no callable; returns ``self``; installs the rule
  under ``external:<name>`` ONLY, rebuilding mapped ``external_call`` ops with
  their name attribute UNCHANGED (the base kernel handles the batched stack
  in place). Creates no derived kernel slot/portable.
- The shared rule (``etl.external_rules``) rebuilds mapped external_call ops
  WITHOUT operand alignment reshapes: mapped operands keep their batched
  shapes, unmapped operands arrive unbatched, and the kernel is called ONCE
  per run with the full batched operand stack as-is (invocation-count pins on
  ``vmap`` AND ``vectorize`` end-to-end runs). Results gain the batch dims
  (the leading ``k`` dims of the first most-mapped operand).
- Nested vmap (2 levels, callable composition ``vmap(vmap(f))``) works for
  BOTH variants: the second pass re-resolves rebuilt derived-named ops via
  the ``external:<derived>`` key.
- Nothing declared (no rule, no portable) → ``TransformError`` naming the op
  and the ``external:<name>`` key with the no-Python-loop-fallback wording;
  declaring the batched variant afterwards makes vmap of the same graph
  succeed.
- Explicit rule vs portable: the pass-through rule wins in BOTH registration
  orders — registered before ``portable()`` it blocks the batching fallback
  (``portable()`` pre-registers the fallback only when no explicit batching
  rule exists); registered after ``portable()`` it overwrites the fallback
  (last-wins). A live batch-variant rule is evidenced by the derived-named
  external_call surviving in the IR (the portable fallback would inline the
  decomposition away, leaving NO external_call op).
- Save/load: a vmapped graph round-trips through ``Graph.save``/``load``
  carrying the derived name; running the loaded graph with the kernels
  unregistered fails with ``BackendError`` naming the derived name.
- Unregister: ``unregister_external_kernel`` removes the base AND derived
  kernel slots and portables; the ``external:<name>`` / ``external:<derived>``
  transform rules survive; vmap works after re-registration + re-declaration;
  unregistering a never-registered name raises ``KeyError``.
- Gap-1 (numpy dispatch is mode-aware): a resolved ``device_resident=True``
  slot on the numpy backend — here the DERIVED batched kernel registered
  ``backend="numpy", device_resident=True`` — raises ``core.BackendError``
  naming the kernel + ``device_resident`` at run (the numpy backend passes
  host arrays and never silently hands them to a device-resident kernel);
  registering a HOST-mode numpy slot for the derived name makes the SAME
  vmapped graph run bit-exact.
"""
import numpy as np
import pytest

import etl
from etl import transforms

from tests.ops.conftest import ops_of

_BATCH_SUFFIX = "__etl_batched"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _run(graph, *args):
    """Explicit pipeline: lower -> compile -> load -> run (returns structure)."""
    return etl.run(etl.load(etl.compile(etl.lower(graph))), *args)


def _derived(name):
    return f"{name}{_BATCH_SUFFIX}"


def _ext_calls(graph):
    """The ``external_call`` ops of a graph's main function."""
    return ops_of(graph, "external_call")


def _ext_names(graph):
    return [op.attributes["name"] for op in _ext_calls(graph)]


def _single_result_type(graph):
    return _ext_calls(graph)[0].results[0].type


def _rule_keys(name):
    return (f"external:{name}", f"external:{_derived(name)}")


def _cleanup(name):
    """Remove the kernel slots + portables AND every transform rule the test
    may have installed (rules are graph-level and survive
    ``unregister_external_kernel`` by design — tests must clean them up so
    later tests start from a blank registry)."""
    try:
        etl.unregister_external_kernel(name)
    except KeyError:
        pass  # the test may already have unregistered mid-test
    for key in _rule_keys(name):
        transforms.batching_rules.pop(key, None)
    transforms.vjp_rules.pop(f"external:{name}", None)
    transforms.jvp_rules.pop(f"external:{name}", None)


# ---------------------------------------------------------------------------
# Reserved derived-name suffix: user-facing registrations reject it
# ---------------------------------------------------------------------------


def test_register_external_kernel_rejects_reserved_suffix():
    """User-facing kernel registrations must not use names containing the
    reserved derived-name suffix ``__etl_batched`` (that namespace belongs to
    kernels registered internally by ``batch_variant``)."""
    for bad in ("eb_res_a__etl_batched", "eb_res_b__etl_batched_sub"):
        with pytest.raises(TypeError, match=_BATCH_SUFFIX) as exc:
            etl.register_external_kernel(bad, lambda x: x)
        assert "register_external_kernel" in str(exc.value)
        # The rejection happens before any mutation.
        assert etl.get_external_kernel(bad) is None

    # The same name without the suffix is fine (positive control).
    handle = etl.register_external_kernel("eb_res_ok", lambda x: x)
    try:
        assert handle.name == "eb_res_ok"
    finally:
        _cleanup("eb_res_ok")


def test_register_portable_rejects_reserved_suffix():
    @etl.defn
    def port(x):
        return x + 1.0

    bad = "eb_resp_a__etl_batched"
    with pytest.raises(TypeError, match=_BATCH_SUFFIX) as exc:
        etl.register_portable(bad, port)
    assert "register_portable" in str(exc.value)
    # The rejection happens before any mutation.
    assert etl.get_portable(bad) is None

    # register_portable on a good base name still works (positive control).
    handle = etl.register_external_kernel("eb_resp_ok", lambda x: x)
    try:
        etl.register_portable("eb_resp_ok", port)
        assert etl.get_portable("eb_resp_ok") is port
    finally:
        _cleanup("eb_resp_ok")


# ---------------------------------------------------------------------------
# batch_variant registration forms + registry/rule-key assertions
# ---------------------------------------------------------------------------


def test_batch_variant_direct_form_registers_derived_slot_and_rules():
    """The DIRECT form (``handle.batch_variant(fn)``) registers ``fn`` under
    the derived name (default slot, host mode) and installs the pass-through
    batching rule under BOTH ``external:<name>`` and ``external:<derived>``.
    Returns ``fn``."""

    def k(x):
        return x * 2 + 1

    def bv(x):
        return x * 2 + 1

    handle = etl.register_external_kernel("eb_form_direct", k)
    try:
        returned = handle.batch_variant(bv)
        assert returned is bv

        derived = _derived("eb_form_direct")
        # Derived slot registered with host-mode default-slot semantics.
        assert etl.external.get_external_kernel_entry(derived) == (bv, False)
        assert etl.get_external_kernel(derived) is bv
        # The base slot is untouched.
        assert etl.get_external_kernel("eb_form_direct") is k
        # Rules under BOTH keys.
        for key in _rule_keys("eb_form_direct"):
            assert key in transforms.batching_rules
    finally:
        _cleanup("eb_form_direct")
    # Cleanup removed the derived slot too.
    assert etl.get_external_kernel(_derived("eb_form_direct")) is None


def test_batch_variant_decorator_and_args_forms():
    """The decorator forms: ``@handle.batch_variant`` (default slot) and
    ``@handle.batch_variant(backend=..., device_resident=True)`` (explicit
    per-backend device slot); both return ``fn``."""

    def k(x):
        return x

    handle = etl.register_external_kernel("eb_form_deco", k)
    try:
        @handle.batch_variant
        def bv_plain(x):
            return x * 3

        @handle.batch_variant(backend="iree", device_resident=True)
        def bv_device(x):
            return x * 4

        derived = _derived("eb_form_deco")
        # Plain decorator -> default slot.
        assert etl.external.get_external_kernel_entry(derived) == (bv_plain, False)
        # Args decorator -> the explicit iree device slot; the default slot
        # still holds the plain variant.
        assert etl.external.get_external_kernel_entry(derived, "iree") == (
            bv_device,
            True,
        )
        assert etl.get_external_kernel(derived) is bv_plain
        # Rules under both keys; the base slot is untouched.
        for key in _rule_keys("eb_form_deco"):
            assert key in transforms.batching_rules
        assert etl.get_external_kernel("eb_form_deco") is k
    finally:
        _cleanup("eb_form_deco")


def test_batch_variant_device_resident_requires_explicit_backend():
    """Mirrors the register_external_kernel rule: ``device_resident=True``
    without an explicit backend raises TypeError (a device kernel in the
    default slot would receive host numpy arrays from the numpy backend)."""
    handle = etl.register_external_kernel("eb_form_dr", lambda x: x)
    try:
        with pytest.raises(TypeError, match="requires an explicit backend"):
            handle.batch_variant(lambda x: x, device_resident=True)

        with pytest.raises(TypeError, match="requires an explicit backend"):

            @handle.batch_variant(device_resident=True)
            def bv_bad(x):
                return x

        # Nothing was registered by the failed attempts.
        assert etl.get_external_kernel(_derived("eb_form_dr")) is None
        for key in _rule_keys("eb_form_dr"):
            assert key not in transforms.batching_rules
    finally:
        _cleanup("eb_form_dr")


# ---------------------------------------------------------------------------
# batch_invariant: declaration + registry/rule keys + unchanged op names
# ---------------------------------------------------------------------------


def test_batch_invariant_returns_self_and_installs_base_rule_only():
    """``batch_invariant()`` takes no callable, returns the handle, and
    installs the pass-through rule under ``external:<name>`` ONLY — no
    derived rule key, no derived kernel slot/portable (the kernel itself is
    the batched implementation)."""

    def k(x):
        return x * 2 + 1

    handle = etl.register_external_kernel("eb_bi", k)
    try:
        assert handle.batch_invariant() is handle
        # Rule under the base key only.
        assert f"external:eb_bi" in transforms.batching_rules
        assert f"external:{_derived('eb_bi')}" not in transforms.batching_rules
        # No derived slot is auto-registered.
        assert etl.get_external_kernel(_derived("eb_bi")) is None
        assert etl.get_portable(_derived("eb_bi")) is None
        assert etl.get_portable("eb_bi") is None
    finally:
        _cleanup("eb_bi")


def test_batch_invariant_end_to_end_vmap_and_vectorize():
    """vmap AND vectorize of a batch-invariant kernel: bit-exact vs the numpy
    reference, the rebuilt op keeps the BASE name attribute, the declared
    result spec gains the batch dim, and the kernel is invoked once per run
    with the full batched stack (no per-row loop)."""
    name = "eb_bi_e2e"
    calls = []

    def k(*arrays):
        calls.append([np.shape(a) for a in arrays])
        return [a * 2 + 1 for a in arrays]

    handle = etl.register_external_kernel(name, k)
    handle.batch_invariant()

    @etl.defn
    def f(x):
        return etl.external_call(name, x, result=etl.TensorSpec((2, 3), etl.int64))

    x = np.arange(24, dtype=np.int64).reshape(4, 2, 3)
    expected = x * 2 + 1

    try:
        # vmap (function-side sugar).
        graph = etl.vmap(f)(etl.TensorSpec((4, 2, 3), etl.int64))
        assert _ext_names(graph) == [name]  # name UNCHANGED
        result_type = _single_result_type(graph)
        assert len(result_type.shape) == 3  # batch dims + declared (2, 3)
        assert tuple(result_type.shape[1:]) == (2, 3)
        calls.clear()
        out = _run(graph, x)
        assert isinstance(out, etl.Tensor)
        np.testing.assert_array_equal(out.numpy(), expected)
        assert calls == [[(4, 2, 3)]]  # kernel ONCE, full batched stack

        # vectorize (the primitive graph-to-graph entry point).
        plain = etl.vectorize(
            etl.trace(f, etl.TensorSpec((2, 3), etl.int64)), 0
        )
        assert _ext_names(plain) == [name]
        calls.clear()
        out = _run(plain, x)
        np.testing.assert_array_equal(out.numpy(), expected)
        assert calls == [[(4, 2, 3)]]
    finally:
        _cleanup(name)


# ---------------------------------------------------------------------------
# batch_variant end-to-end: full-stack semantics via vmap + vectorize
# ---------------------------------------------------------------------------


def test_batch_variant_end_to_end_vmap_and_vectorize():
    """vmap AND vectorize of a batched-variant kernel: the rebuilt op carries
    the DERIVED name, results gain the batch dims, and the derived kernel is
    invoked once per run with the FULL batched operand stack as-is."""
    name = "eb_bv_e2e"
    calls = []

    def base_k(x):
        return x * 2 + 1

    def bv(x):
        calls.append(np.shape(x))
        return x * 2 + 1

    handle = etl.register_external_kernel(name, base_k)
    handle.batch_variant(bv)

    @etl.defn
    def f(x):
        return etl.external_call(name, x, result=etl.TensorSpec((2, 3), etl.int64))

    x = np.arange(24, dtype=np.int64).reshape(4, 2, 3)
    expected = x * 2 + 1

    try:
        graph = etl.vmap(f)(etl.TensorSpec((4, 2, 3), etl.int64))
        derived = _derived(name)
        assert _ext_names(graph) == [derived]
        result_type = _single_result_type(graph)
        assert len(result_type.shape) == 3
        assert tuple(result_type.shape[1:]) == (2, 3)

        calls.clear()
        out = _run(graph, x)
        np.testing.assert_array_equal(out.numpy(), expected)
        assert calls == [(4, 2, 3)]  # ONCE, the full stack

        # vectorize path: identical semantics.
        plain = etl.vectorize(
            etl.trace(f, etl.TensorSpec((2, 3), etl.int64)), 0
        )
        assert _ext_names(plain) == [derived]
        calls.clear()
        out = _run(plain, x)
        np.testing.assert_array_equal(out.numpy(), expected)
        assert calls == [(4, 2, 3)]

        # Without vmap the BASE name still dispatches the BASE kernel (the
        # derived kernel never runs on the unbatched path).
        calls.clear()
        out = _run(etl.trace(f, etl.TensorSpec((2, 3), etl.int64)), x[0])
        np.testing.assert_array_equal(out.numpy(), expected[0])
        assert calls == []
    finally:
        _cleanup(name)


def test_batch_variant_multi_output_full_stack():
    """Multi-output batched kernels flow through the same rebuild (each
    result gains the batch dims from the operand stack) into downstream
    ops."""
    name = "eb_bv_multi"
    calls = []

    def bv(a):
        calls.append(np.shape(a))
        return a + 1, a * 2

    handle = etl.register_external_kernel(name, lambda a: (a + 1, a * 2))
    handle.batch_variant(bv)

    @etl.defn
    def f(a):
        b, c = etl.external_call(
            name,
            a,
            result=(
                etl.TensorSpec((3,), etl.int64),
                etl.TensorSpec((3,), etl.int64),
            ),
        )
        return b + c

    x = np.arange(12, dtype=np.int64).reshape(4, 3)
    expected = (x + 1) + (x * 2)
    try:
        graph = etl.vmap(f)(etl.TensorSpec((4, 3), etl.int64))
        assert _ext_names(graph) == [_derived(name)]
        for result in _ext_calls(graph)[0].results:
            assert len(result.type.shape) == 2
            assert tuple(result.type.shape[1:]) == (3,)
        calls.clear()
        out = _run(graph, x)
        np.testing.assert_array_equal(out.numpy(), expected)
        assert calls == [(4, 3)]
    finally:
        _cleanup(name)


# ---------------------------------------------------------------------------
# Nested vmap (2 levels, both variants)
# ---------------------------------------------------------------------------


def test_nested_vmap_batch_variant():
    """Nested vmap through ``vmap(vmap(f))`` with a batched-variant kernel:
    the inner pass rebuilds the op with the derived name, the outer pass
    re-resolves it through the ``external:<derived>`` key, and the derived
    kernel is called ONCE with the full 3-d batched stack."""
    name = "eb_nest_bv"
    calls = []

    def bv(x):
        calls.append(np.shape(x))
        return x * 2 + 1

    handle = etl.register_external_kernel(name, lambda x: x * 2 + 1)
    handle.batch_variant(bv)

    @etl.defn
    def f(x):
        return etl.external_call(name, x, result=etl.TensorSpec((2,), etl.int64))

    x = np.arange(24, dtype=np.int64).reshape(3, 4, 2)
    expected = x * 2 + 1
    try:
        graph = etl.vmap(etl.vmap(f))(etl.TensorSpec((3, 4, 2), etl.int64))
        calls_op = _ext_calls(graph)
        assert len(calls_op) == 1
        assert calls_op[0].attributes["name"] == _derived(name)
        # both batch dims are mapped: result shape = (batch, batch_1) + (2,)
        assert tuple(calls_op[0].results[0].type.shape[2:]) == (2,)

        calls.clear()
        out = _run(graph, x)
        np.testing.assert_array_equal(out.numpy(), expected)
        assert calls == [(3, 4, 2)]
    finally:
        _cleanup(name)


def test_nested_vmap_batch_invariant():
    """Nested vmap through ``vmap(vmap(f))`` with a batch-INVARIANT kernel:
    the rebuilt op keeps the base name through BOTH passes, and the base
    kernel receives the full 3-d stack once."""
    name = "eb_nest_bi"
    calls = []

    def k(*arrays):
        calls.append([np.shape(a) for a in arrays])
        return [a * 2 + 1 for a in arrays]

    handle = etl.register_external_kernel(name, k)
    handle.batch_invariant()

    @etl.defn
    def f(x):
        return etl.external_call(name, x, result=etl.TensorSpec((2,), etl.int64))

    x = np.arange(24, dtype=np.int64).reshape(3, 4, 2)
    expected = x * 2 + 1
    try:
        graph = etl.vmap(etl.vmap(f))(etl.TensorSpec((3, 4, 2), etl.int64))
        assert _ext_names(graph) == [name]
        calls.clear()
        out = _run(graph, x)
        np.testing.assert_array_equal(out.numpy(), expected)
        assert calls == [[(3, 4, 2)]]
    finally:
        _cleanup(name)


# ---------------------------------------------------------------------------
# Mixed mapped counts: unmapped operands arrive unbatched (no reshapes)
# ---------------------------------------------------------------------------


def test_batch_variant_mixed_mapped_counts():
    """No operand alignment reshapes: with one mapped and one unmapped
    operand, the mapped operand is passed through with its leading batch dim
    and the unmapped operand arrives with its original shape — the kernel
    decides how to combine them."""
    name = "eb_mix"
    calls = []

    def bv(a, b):
        calls.append((np.shape(a), np.shape(b)))
        return a + b  # numpy broadcasts the (4, 2, 3) with the (3,)

    handle = etl.register_external_kernel(name, bv)
    handle.batch_variant(bv)

    @etl.defn
    def f(a, b):
        return etl.external_call(
            name, a, b, result=etl.TensorSpec((2, 3), etl.int64)
        )

    graph = etl.trace(
        f,
        etl.TensorSpec((2, 3), etl.int64),
        etl.TensorSpec((3,), etl.int64),
    )
    x = np.arange(24, dtype=np.int64).reshape(4, 2, 3)
    w = np.array([1, 2, 3], dtype=np.int64)
    expected = x + w
    try:
        batched = etl.vectorize(graph, (0, None))
        assert _ext_names(batched) == [_derived(name)]
        # Result shape = batch dims (from the mapped operand) + declared.
        result_type = _single_result_type(batched)
        assert len(result_type.shape) == 3
        assert tuple(result_type.shape[1:]) == (2, 3)

        calls.clear()
        out = _run(batched, x, w)
        np.testing.assert_array_equal(out.numpy(), expected)
        assert calls == [((4, 2, 3), (3,))]  # unmapped operand as-is

        # Same via vmap's in_axes pytree form.
        calls.clear()
        batched2 = etl.vmap(f, in_axes=(0, None))(
            etl.TensorSpec((4, 2, 3), etl.int64),
            etl.TensorSpec((3,), etl.int64),
        )
        out = _run(batched2, x, w)
        np.testing.assert_array_equal(out.numpy(), expected)
        assert calls == [((4, 2, 3), (3,))]
    finally:
        _cleanup(name)


# ---------------------------------------------------------------------------
# Nothing declared -> TransformError (no Python-loop fallback); declaration
# afterwards enables batching of the SAME pre-traced graph
# ---------------------------------------------------------------------------


def test_nothing_declared_raises_then_success_after_batch_variant():
    """A registered kernel alone installs no batching rule: vmap raises the
    canonical TransformError naming the op and the ``external:<name>`` key
    (no per-element Python-loop fallback exists). Declaring the batched
    variant afterwards enables vmap of the SAME pre-traced graph."""
    name = "eb_nothing"

    @etl.defn
    def f(x):
        return etl.external_call(name, x, result=etl.TensorSpec((2,), etl.int64))

    graph = etl.trace(f, etl.TensorSpec((2,), etl.int64))
    handle = etl.register_external_kernel(name, lambda x: x * 2 + 1)
    try:
        assert f"external:{name}" not in transforms.batching_rules
        with pytest.raises(etl.TransformError) as exc:
            etl.vmap(graph, in_axes=0)
        message = str(exc.value)
        assert "external_call" in message
        assert f"external:{name}" in message
        assert "Python-loop fallback" in message  # never a silent loop

        handle.batch_variant(lambda x: x * 2 + 1)
        batched = etl.vmap(graph, in_axes=0)
        x = np.arange(8, dtype=np.int64).reshape(4, 2)
        out = _run(batched, x)
        np.testing.assert_array_equal(out.numpy(), x * 2 + 1)
    finally:
        _cleanup(name)


# ---------------------------------------------------------------------------
# Explicit rule vs portable: the pass-through rule wins in BOTH orders
# ---------------------------------------------------------------------------


def _decl_f(name):
    @etl.defn
    def f(x):
        return etl.external_call(name, x, result=etl.TensorSpec((2,), etl.int64))

    return f


def _assert_batch_variant_rule_live(name, x, expected):
    """Shared end-to-end check: the batch-variant rule is what vmap uses —
    evidenced by the derived-named external_call surviving in the IR (a
    portable fallback would inline the decomposition away, leaving NO
    external_call op) — with a bit-exact run."""
    graph = etl.vmap(
        etl.trace(_decl_f(name), etl.TensorSpec((2,), etl.int64)), 0
    )
    assert _ext_names(graph) == [_derived(name)]
    out = _run(graph, x)
    np.testing.assert_array_equal(out.numpy(), expected)


def test_batch_variant_registered_before_portable_wins():
    """Order 1: ``batch_variant`` first, ``portable()`` second. The
    portable's batching fallback fills only a slot no explicit rule claimed
    — the pass-through rule stays, and vmap uses the batched variant
    (derived name in the IR, no inlined decomposition)."""
    name = "eb_ord_before"

    @etl.defn
    def port(x):
        return x + 1.0

    handle = etl.register_external_kernel(name, lambda x: x * 2 + 1)
    handle.batch_variant(lambda x: x * 2 + 1)
    pass_through = transforms.batching_rules.get(f"external:{name}")
    try:
        assert pass_through is not None
        handle.portable(port)
        # The explicit rule survived portable() (fallback not installed).
        assert transforms.batching_rules.get(f"external:{name}") is pass_through
        x = np.arange(8, dtype=np.int64).reshape(4, 2)
        _assert_batch_variant_rule_live(name, x, x * 2 + 1)
    finally:
        _cleanup(name)


def test_batch_variant_registered_after_portable_wins():
    """Order 2: ``portable()`` first, ``batch_variant`` second. The portable
    pre-registers its batching fallback; the later batch_variant overwrites
    it (last-wins ordinary registry assignment)."""
    name = "eb_ord_after"

    @etl.defn
    def port(x):
        return x + 1.0

    handle = etl.register_external_kernel(name, lambda x: x * 2 + 1)
    handle.portable(port)
    fallback = transforms.batching_rules.get(f"external:{name}")
    try:
        assert fallback is not None  # the pre-registered portable fallback
        handle.batch_variant(lambda x: x * 2 + 1)
        assert transforms.batching_rules.get(f"external:{name}") is not fallback
        x = np.arange(8, dtype=np.int64).reshape(4, 2)
        _assert_batch_variant_rule_live(name, x, x * 2 + 1)
    finally:
        _cleanup(name)


# ---------------------------------------------------------------------------
# Save/load round-trip with derived names
# ---------------------------------------------------------------------------


def test_graph_save_load_roundtrip_with_derived_names(tmp_path):
    """A vmapped graph (whose external_call carries the DERIVED name)
    round-trips through ``Graph.save``/``load``; with the kernels registered
    the loaded graph runs bit-exact, and after unregistering (the registry is
    never serialized) the run fails with a BackendError NAMING the derived
    kernel."""
    name = "eb_persist"
    handle = etl.register_external_kernel(name, lambda x: x * 2 + 1)
    handle.batch_variant(lambda x: x * 2 + 1)

    @etl.defn
    def f(x):
        return etl.external_call(name, x, result=etl.TensorSpec((2, 3), etl.int64))

    x = np.arange(24, dtype=np.int64).reshape(4, 2, 3)
    expected = x * 2 + 1
    loaded = None
    try:
        graph = etl.vmap(f)(etl.TensorSpec((4, 2, 3), etl.int64))
        assert _ext_names(graph) == [_derived(name)]
        path = tmp_path / "batched_ext.etl"
        graph.save(str(path))
        loaded = etl.Graph.load(str(path))
        assert _ext_names(loaded) == [_derived(name)]
        out = _run(loaded, x)
        np.testing.assert_array_equal(out.numpy(), expected)
    finally:
        _cleanup(name)

    # Kernels unregistered -> the loaded graph fails loudly, naming the
    # derived name the op now carries.
    assert loaded is not None
    with pytest.raises(etl.BackendError) as exc:
        _run(loaded, x)
    assert _derived(name) in str(exc.value)
    assert "register_external_kernel" in str(exc.value)


# ---------------------------------------------------------------------------
# Unregister: base + derived slots/portables gone; rules survive
# ---------------------------------------------------------------------------


def test_unregister_removes_derived_slots_and_portables_and_rules_survive():
    """unregister_external_kernel cleans up the base AND the derived
    (batched-variant) namespace: kernel slots and portables for both are
    gone. Transform rules under ``external:<name>`` AND ``external:<derived>``
    are graph-level registrations and survive with their identity."""
    name = "eb_unreg"
    base_rule_key, derived_rule_key = _rule_keys(name)

    def k(x):
        return x

    @etl.defn
    def port(x):
        return x + 1.0

    handle = etl.register_external_kernel(name, k)
    handle.impl("iree", k, device_resident=True)
    handle.portable(port)
    handle.batch_variant(lambda x: x)
    base_rule = transforms.batching_rules.get(base_rule_key)
    derived_rule = transforms.batching_rules.get(derived_rule_key)
    try:
        assert base_rule is not None and derived_rule is not None
        assert etl.get_external_kernel(name) is not None
        assert etl.get_external_kernel(name, "iree") is not None
        assert etl.get_external_kernel(_derived(name)) is not None
        assert etl.get_portable(name) is port

        etl.unregister_external_kernel(name)
        # Base AND derived slots gone, portables gone (derived never had one).
        assert etl.get_external_kernel(name) is None
        assert etl.get_external_kernel(name, "iree") is None
        assert etl.get_external_kernel(_derived(name)) is None
        assert etl.get_portable(name) is None
        assert etl.get_portable(_derived(name)) is None
        # Transform rules survive with their original identity.
        assert transforms.batching_rules.get(base_rule_key) is base_rule
        assert transforms.batching_rules.get(derived_rule_key) is derived_rule
    finally:
        _cleanup(name)


def test_unregister_then_reregister_restores_vmap():
    """After unregister, re-registering the kernel and RE-declaring the
    batched variant restores vmap end-to-end (registry keys are fresh)."""
    name = "eb_rereg"

    def k(x):
        return x * 2 + 1

    handle = etl.register_external_kernel(name, k)
    handle.batch_variant(k)
    etl.unregister_external_kernel(name)
    assert etl.get_external_kernel(name) is None

    handle2 = etl.register_external_kernel(name, k)
    handle2.batch_variant(k)
    try:
        @etl.defn
        def f(x):
            return etl.external_call(name, x, result=etl.TensorSpec((2,), etl.int64))

        x = np.arange(8, dtype=np.int64).reshape(4, 2)
        graph = etl.vmap(
            etl.trace(f, etl.TensorSpec((2,), etl.int64)), 0
        )
        assert _ext_names(graph) == [_derived(name)]
        out = _run(graph, x)
        np.testing.assert_array_equal(out.numpy(), x * 2 + 1)
    finally:
        _cleanup(name)


def test_unregister_never_registered_raises_key_error():
    with pytest.raises(
        KeyError, match="no external kernel registered under 'eb_ghost'"
    ):
        etl.unregister_external_kernel("eb_ghost")


# ---------------------------------------------------------------------------
# Gap-1: numpy dispatch is mode-aware — a device_resident numpy slot on the
# DERIVED (batched) kernel raises at run; a host-mode numpy slot recovers
# ---------------------------------------------------------------------------


def test_device_resident_numpy_batched_slot_raises_and_host_recovery():
    """A batched variant registered ``backend="numpy", device_resident=True``
    makes the numpy run of the vmapped graph raise ``BackendError`` naming
    the derived kernel + ``device_resident`` (the numpy backend passes host
    arrays and cannot honor device semantics — never a silent hand-off).
    Re-registering a HOST-mode numpy slot for the derived name makes the SAME
    graph run bit-exact."""
    name = "eb_gap1"

    def k(x):
        return x * 2 + 1

    handle = etl.register_external_kernel(name, k)
    handle.batch_variant(k, backend="numpy", device_resident=True)

    @etl.defn
    def f(x):
        return etl.external_call(name, x, result=etl.TensorSpec((2,), etl.int64))

    graph = etl.vmap(etl.trace(f, etl.TensorSpec((2,), etl.int64)), 0)
    x = np.arange(8, dtype=np.int64).reshape(4, 2)
    try:
        assert etl.external.get_external_kernel_entry(
            _derived(name), "numpy"
        ) == (k, True)
        with pytest.raises(etl.BackendError) as exc:
            _run(graph, x)
        message = str(exc.value)
        assert "device_resident" in message
        assert _derived(name) in message  # the derived kernel is named

        # Host-mode recovery: a host numpy slot for the derived name.
        handle.batch_variant(k, backend="numpy")
        out = _run(graph, x)
        np.testing.assert_array_equal(out.numpy(), x * 2 + 1)
    finally:
        _cleanup(name)
