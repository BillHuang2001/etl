"""EvoXIR — the compiler-neutral, region-based SSA IR of the EvoX Tensor Library.

What lives here:

* the SSA data model: ``Module``, ``Function``, ``Region``, ``Block``, ``Op``,
  ``Value`` (with ``Use``), typed by ``ValueType`` (dtype + symbolic shape);
* the op-definition registry (``opdef``/``OpDef``/``AttrSpec``) declaring the
  full canonical v1 op set (~75 ops, split by category under ``op_defs/``);
* the ``Builder`` — the only op-construction API;
* ``verify`` — structural/type/attribute validation raising
  ``VerificationError``;
* serialization — ``IR_FORMAT_VERSION``, ``serialize_module``,
  ``deserialize_module`` (the self-describing payload at the core of the
  ``.etlgraph`` format);
* ``pretty_print`` — readable SSA text.

Import rule (binding): this package imports ONLY ``etl.core`` (Dim/DimExpr,
errors), stdlib, and numpy. Lower layers (``ops``, ``trace``, ``backends``,
...) import from here — never the reverse. See ``./CONTEXT.md`` for design
decisions, invariants, and the effect model.

Note: Phase 2 is in progress — the data structures, the registry, shape
inference (`inference`), `pretty_print`, the `Builder`, and `verify` are real;
the remaining behavioral bodies (`serialize_module`/`deserialize_module`)
raise ``NotImplementedError`` until Phase 2 fills them.
"""

from etl.core import VerificationError  # owned by core; re-exported here

from . import inference
from .block import Block
from .builder import Builder
from .effects import (
    EFFECT_CALLBACK,
    EFFECT_COLLECTIVE,
    EFFECT_KINDS,
    EFFECT_PURE,
    EFFECT_READ,
    EFFECT_WRITE,
)
from .function import Function
from .location import Location
from .module import Module
from .op import Op
from .op_defs import (
    ATTR_ANY,
    ATTR_BOOL,
    ATTR_DTYPE,
    ATTR_FLOAT,
    ATTR_FLOATS,
    ATTR_INT,
    ATTR_INTS,
    ATTR_NDARRAY,
    ATTR_NESTED_INTS,
    ATTR_SHAPE,
    ATTR_STR,
    ATTR_STRS,
    ATTR_TYPE_NAMES,
    AttrSpec,
    OpDef,
    all_opdefs,
    has_opdef,
    op_names,
    opdef,
    register_opdef,
)
from .printer import pretty_print
from .region import Region
from .serialize import deserialize_module, serialize_module
from .types import Shape, ShapeDim, ValueType
from .value import Use, Value
from .verify import verify
from .version import IR_FORMAT_VERSION

__all__ = [
    # SSA structures
    "Module",
    "Function",
    "Region",
    "Block",
    "Op",
    "Value",
    "Use",
    # types & location
    "ValueType",
    "Shape",
    "ShapeDim",
    "Location",
    # effects
    "EFFECT_PURE",
    "EFFECT_WRITE",
    "EFFECT_READ",
    "EFFECT_COLLECTIVE",
    "EFFECT_CALLBACK",
    "EFFECT_KINDS",
    # op registry
    "OpDef",
    "AttrSpec",
    "opdef",
    "has_opdef",
    "op_names",
    "all_opdefs",
    "register_opdef",
    "ATTR_BOOL",
    "ATTR_INT",
    "ATTR_FLOAT",
    "ATTR_STR",
    "ATTR_DTYPE",
    "ATTR_INTS",
    "ATTR_FLOATS",
    "ATTR_STRS",
    "ATTR_NESTED_INTS",
    "ATTR_SHAPE",
    "ATTR_NDARRAY",
    "ATTR_ANY",
    "ATTR_TYPE_NAMES",
    # builder
    "Builder",
    # verification
    "verify",
    "VerificationError",
    # serialization
    "IR_FORMAT_VERSION",
    "serialize_module",
    "deserialize_module",
    # printing
    "pretty_print",
    # submodule (shape-inference hooks)
    "inference",
]
