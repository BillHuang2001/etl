"""Declaration forms of `etl.block` — factory + decorator + validation errors.

Covers ONLY declaration (no tracing, no execution): factory basics, attribute
schema normalization, batching-policy resolution, decorator-form spec
derivation, duplicate/unknown registration errors, and the pending-decorator
form.

IMPORTANT: the block registry is GLOBAL and process-wide — a name can never be
re-declared. Every block declared here has a unique `decl_`-prefixed name so
this file never collides with itself or with other test files.

NOTE: deliberately NO `from __future__ import annotations` in this file —
`etl.block` derives TensorSpecs from parameter/return annotations via
`inspect.signature`, which on Python 3.11 leaves stringified annotations
unevaluated; eager annotations are required for the decorator-form tests.
"""

import pytest

import etl
from etl.block import AttributeField, BatchingPolicy, BlockError

# Shared small spec for factory-form declarations.
_SPEC = etl.TensorSpec((3, 4), etl.float32)
_SCALAR = etl.TensorSpec((), etl.float32)


# ---------------------------------------------------------------------------
# 1. Factory form basics
# ---------------------------------------------------------------------------


def test_factory_form_basics():
    blk = etl.block(
        "decl_double",
        inputs=[_SPEC],
        outputs=[etl.TensorSpec((3, 4), etl.float32)],
        attributes={"scale": float, "causal": bool},
        effects="pure",
        batching="opaque_batched",
    )

    assert callable(blk)
    assert isinstance(blk, etl.BlockOp)

    assert blk.name == "decl_double"

    assert isinstance(blk.input_specs, tuple)
    assert len(blk.input_specs) == 1
    assert blk.input_specs[0] == _SPEC
    assert blk.input_specs[0].shape == (3, 4)
    assert blk.input_specs[0].dtype == etl.float32

    assert isinstance(blk.output_specs, tuple)
    assert len(blk.output_specs) == 1
    assert blk.output_specs[0] == etl.TensorSpec((3, 4), etl.float32)

    schema = blk.attribute_schema
    assert isinstance(schema, dict)
    assert set(schema) == {"scale", "causal"}
    assert schema["scale"].type is float
    assert schema["scale"].required is True
    assert schema["causal"].type is bool
    assert schema["causal"].required is True
    # Contract name vs objective name — documented alias, same mapping.
    assert blk.attributes is blk.attribute_schema

    assert blk.effects == "pure"
    assert isinstance(blk.batching_policy, BatchingPolicy)
    assert blk.batching_policy == "opaque_batched"
    assert blk.batching_policy == BatchingPolicy.OPAQUE_BATCHED

    assert blk.has_portable is False
    assert blk.get_impl("numpy") is None


# ---------------------------------------------------------------------------
# 2. Attribute schema normalization (bare type = required, default = optional)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, attributes, key, expected_type, expected_required, expected_default",
    [
        ("decl_attr_required", {"scale": float}, "scale", float, True, None),
        ("decl_attr_optional", {"eps": 1e-5}, "eps", float, False, 1e-5),
    ],
    ids=["bare-type-required", "default-value-optional"],
)
def test_attribute_schema_normalization(
    name, attributes, key, expected_type, expected_required, expected_default
):
    blk = etl.block(name, inputs=[_SPEC], outputs=[_SPEC], attributes=attributes)

    field = blk.attribute_schema[key]
    assert isinstance(field, AttributeField)
    assert field.name == key
    assert field.type is expected_type
    assert field.required is expected_required
    if expected_required:
        # Required fields carry the dataclass MISSING sentinel as their
        # "default" — there is no usable default. A fresh AttributeField
        # without a default holds the very same sentinel instance.
        missing = AttributeField(name="_sentinel", type=int).default
        assert field.default is missing
    else:
        assert field.default == expected_default


# ---------------------------------------------------------------------------
# 3. Batching policy resolution
# ---------------------------------------------------------------------------


def test_default_batching_without_portable_is_unsupported():
    blk = etl.block("decl_unsupported", inputs=[_SPEC], outputs=[_SPEC])

    assert blk.has_portable is False
    assert blk.batching_policy == "unsupported"
    assert blk.batching_policy == BatchingPolicy.UNSUPPORTED


def test_default_batching_with_portable_is_batching_rule():
    @etl.defn
    def _decl_portable_identity(
        x: etl.TensorSpec((), etl.float32),
    ) -> etl.TensorSpec((), etl.float32):
        return x

    blk = etl.block(
        "decl_batch_portable",
        inputs=[_SCALAR],
        outputs=[_SCALAR],
        portable=_decl_portable_identity,
    )

    assert blk.has_portable is True
    assert blk.batching_policy == "batching_rule"
    assert blk.batching_policy is BatchingPolicy.BATCHING_RULE


@pytest.mark.parametrize(
    "name, batching, expected",
    [
        ("decl_batch_string", "elementwise", BatchingPolicy.ELEMENTWISE),
        (
            "decl_batch_enum",
            BatchingPolicy.MAP_OVER_BATCH,
            BatchingPolicy.MAP_OVER_BATCH,
        ),
    ],
    ids=["string", "enum"],
)
def test_explicit_batching_policy(name, batching, expected):
    blk = etl.block(name, inputs=[_SPEC], outputs=[_SPEC], batching=batching)

    assert blk.batching_policy is expected
    assert blk.batching_policy == expected
    assert blk.batching_policy == expected.value


def test_invalid_batching_policy_raises():
    with pytest.raises(BlockError, match="batching must be one of"):
        etl.block(
            "decl_bad_batching", inputs=[_SPEC], outputs=[_SPEC], batching="nonsense"
        )


# ---------------------------------------------------------------------------
# 4. Decorator form (over @etl.defn)
# ---------------------------------------------------------------------------


def test_decorator_form_derives_specs_from_annotations():
    @etl.block
    @etl.defn
    def decl_swish(
        x: etl.TensorSpec((), etl.float32),
    ) -> etl.TensorSpec((), etl.float32):
        return etl.sigmoid(x) * x

    assert isinstance(decl_swish, etl.BlockOp)
    assert decl_swish.name == "decl_swish"
    assert decl_swish.input_specs == (_SCALAR,)
    assert decl_swish.output_specs == (_SCALAR,)
    assert decl_swish.has_portable is True
    assert decl_swish.batching_policy == "batching_rule"
    assert decl_swish.batching_policy is BatchingPolicy.BATCHING_RULE


def test_decorator_form_derives_specs_from_defaults():
    @etl.block
    @etl.defn
    def decl_default_specs(
        x=etl.TensorSpec((2, 3), etl.float32),
    ) -> etl.TensorSpec((2, 3), etl.float32):
        return x

    assert decl_default_specs.name == "decl_default_specs"
    assert decl_default_specs.input_specs == (etl.TensorSpec((2, 3), etl.float32),)
    assert decl_default_specs.output_specs == (etl.TensorSpec((2, 3), etl.float32),)
    assert decl_default_specs.has_portable is True


def test_decorator_form_explicit_outputs_kwarg():
    @etl.block(outputs=[etl.TensorSpec((4,), etl.float32)])
    @etl.defn
    def decl_explicit_outputs(x: etl.TensorSpec((4,), etl.float32)):
        return x

    assert decl_explicit_outputs.name == "decl_explicit_outputs"
    assert decl_explicit_outputs.input_specs == (etl.TensorSpec((4,), etl.float32),)
    assert decl_explicit_outputs.output_specs == (etl.TensorSpec((4,), etl.float32),)
    assert decl_explicit_outputs.has_portable is True


def test_decorator_form_explicit_result_kwarg():
    @etl.block(result=[etl.TensorSpec((4,), etl.float32)])
    @etl.defn
    def decl_explicit_result(x: etl.TensorSpec((4,), etl.float32)):
        return x

    assert decl_explicit_result.name == "decl_explicit_result"
    assert decl_explicit_result.input_specs == (etl.TensorSpec((4,), etl.float32),)
    assert decl_explicit_result.output_specs == (etl.TensorSpec((4,), etl.float32),)
    assert decl_explicit_result.has_portable is True


@pytest.mark.parametrize(
    "make_decorator, match",
    [
        # Bare @etl.block over a plain (non-defn) function.
        (lambda: etl.block, "must be an etl.defn"),
        # @etl.block(outputs=...) over a plain function.
        (
            lambda: etl.block(outputs=[_SCALAR]),
            "must decorate an etl.defn",
        ),
        # Pending decorator (etl.block()) applied to a plain function.
        (lambda: etl.block(), "must decorate an etl.defn"),
    ],
    ids=["bare", "with-kwargs", "pending"],
)
def test_decorator_rejects_non_defn(make_decorator, match):
    def plain(x):
        return x

    with pytest.raises(BlockError, match=match):
        make_decorator()(plain)


# ---------------------------------------------------------------------------
# 5. Duplicate registration
# ---------------------------------------------------------------------------


def test_duplicate_registration_raises():
    etl.block("decl_dup", inputs=[_SPEC], outputs=[_SPEC])
    with pytest.raises(BlockError, match="already registered"):
        etl.block("decl_dup", inputs=[_SPEC], outputs=[_SPEC])


# ---------------------------------------------------------------------------
# 6. get_block registry accessor
# ---------------------------------------------------------------------------


def test_get_block_returns_registered_op():
    blk = etl.block("decl_getblock", inputs=[_SPEC], outputs=[_SPEC])
    assert etl.get_block("decl_getblock") is blk


def test_get_block_unknown_raises():
    with pytest.raises(BlockError, match="unknown block"):
        etl.get_block("decl_never_declared")


# ---------------------------------------------------------------------------
# 7. Declaration validation errors + pending decorator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, bad_inputs",
    [
        ("decl_bad_inputs_list", [3]),
        ("decl_bad_inputs_scalar", 5),
    ],
    ids=["list-of-non-spec", "scalar"],
)
def test_inputs_must_be_tensor_specs(name, bad_inputs):
    with pytest.raises(BlockError, match="inputs must be a TensorSpec"):
        etl.block(name, inputs=bad_inputs, outputs=[_SPEC])


def test_invalid_effects_raise():
    with pytest.raises(BlockError, match="effects must be one of"):
        etl.block("decl_bad_effects", inputs=[_SPEC], effects="magic")


@pytest.mark.parametrize(
    "effect", ["callback", "collective", "pure", "read", "write"]
)
def test_valid_effects_accepted(effect):
    blk = etl.block(f"decl_effect_{effect}", inputs=[_SPEC], effects=effect)
    assert blk.effects == effect


@pytest.mark.parametrize("bad_name", ["has space", "1abc"])
def test_invalid_block_name_raises(bad_name):
    with pytest.raises(BlockError, match="invalid block name"):
        etl.block(bad_name, inputs=[_SPEC])


def test_factory_requires_inputs_or_portable():
    with pytest.raises(BlockError, match="no tensor inputs and no portable"):
        etl.block("decl_no_inputs")


def test_bare_block_returns_pending_decorator():
    pending = etl.block()
    assert repr(pending) == "<pending etl.block declaration (no name and no function yet)>"
    assert "pending etl.block declaration" in repr(pending)

    @etl.defn
    def decl_pending(
        x: etl.TensorSpec((), etl.float32),
    ) -> etl.TensorSpec((), etl.float32):
        return x

    # Applying the pending decorator completes the declaration.
    blk = pending(decl_pending)
    assert isinstance(blk, etl.BlockOp)
    assert blk.name == "decl_pending"
    assert blk.input_specs == (_SCALAR,)
    assert blk.output_specs == (_SCALAR,)
    assert blk.has_portable is True
