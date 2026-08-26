"""IR text printing (debugging, logging, and human review).

Example output (note: block arguments carry no locations in the data model,
so ``loc(...)`` appears on ops only; the original format spec illustrated
argument locations which ``Value`` does not store)::

    module @"main" version 1 {
      func @main(%arg0: tensor<BxNxf32>,
                 %arg1: tensor<Nxf32>) -> tensor<Bxf32> {
        %0 = etl.add(%arg0, %arg1) : tensor<BxNxf32> loc("model.py":12:8)
        %1 = etl.reduce_sum(%0) attributes {axes = [1]} : tensor<Bxf32> loc("model.py":13:4)
        etl.return(%1) loc("model.py":13:4)
      }
    }

Format rules: block arguments print as ``%argN`` (N = the argument index within
its block); op results print as ``%N`` (numbered sequentially per function,
starting at 0; only ops *with* results consume numbers); each op line is
``%r = name(operands) [attributes {...}] : type(s) loc(...)`` — the ``: type``
part is omitted for ops with zero results and ``attributes {...}`` only for
ops with non-empty attributes; nested regions render inline under the op,
indented two spaces deeper, with each nested block labeled ``^bbN``.
"""
from __future__ import annotations

import json
from itertools import count
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np

from etl.core import Dim, DimExpr  # attribute values may be symbolic dims

from .location import Location
from .module import Module
from .types import _dim_str, _dtype_str

if TYPE_CHECKING:
    from .block import Block
    from .function import Function
    from .op import Op

_INDENT = "  "  # one indentation level


def pretty_print(module: Module) -> str:
    """Render ``module`` as readable SSA text (format above).

    The output is deterministic: functions, ops, and block arguments print in
    program order, attribute keys print sorted, and op results are renumbered
    ``%0, %1, ...`` per function (SSA ids are not reused verbatim).

    Raises:
        KeyError: If the module references an unregistered op name.
        ValueError: If a function has no ``return`` terminator (its output
            signature cannot be read).
    """
    lines = [f'module @{json.dumps(module.name)} version {module.version} {{']
    for function in module.functions:
        lines.extend(_print_function(function, 1))
    lines.append("}")
    return "\n".join(lines) + "\n"


def _print_function(function: Function, level: int) -> list[str]:
    """Render one function: header line(s) + entry block + closing brace."""
    pad = _INDENT * level
    names = _collect_names(function)
    lines = _function_header(function, pad)
    lines.extend(
        _print_block(function.entry_block, pad + _INDENT, names, count(), entry=True)
    )
    lines.append(f"{pad}}}")
    return lines


# --- value naming ------------------------------------------------------------
#
# Printed names are per-function renumberings, not raw SSA ids: op results
# become %0, %1, ... in program order (ops without results consume no number);
# block arguments become %argN by argument index within their block. A single
# pre-pass assigns every name before any line is rendered, so operands resolve
# even if a (verification-failing) module references a value "out of order".


def _collect_names(function: Function) -> dict[int, str]:
    """Map ``id(value)`` -> printed SSA name for every value in the function."""
    names: dict[int, str] = {}
    counter = count()
    entry = function.entry_block
    for i, argument in enumerate(entry.arguments):
        names[id(argument)] = f"%arg{i}"
    _collect_block_names(entry, names, counter)
    return names


def _collect_block_names(
    block: Block, names: dict[int, str], counter: Iterator[int]
) -> None:
    for op in block.ops:
        op.opdef  # raises KeyError for unregistered op names
        for result in op.results:
            names[id(result)] = f"%{next(counter)}"
        for region in op.regions:
            for nested in region.blocks:
                for i, argument in enumerate(nested.arguments):
                    names[id(argument)] = f"%arg{i}"
                _collect_block_names(nested, names, counter)


def _value_name(names: dict[int, str], value: Any) -> str:
    """The printed name of a value (``%<unknown N>`` for unnamed references)."""
    return names.get(id(value), f"%<unknown {value.id}>")


# --- rendering ---------------------------------------------------------------


def _function_header(function: Function, pad: str) -> list[str]:
    """The ``func @name(...) -> ... {`` header (one or more lines).

    Zero or one block argument renders on a single line; two or more render
    one argument per line, continued aligned right after the ``(``. Result
    types are comma-separated after ``->`` (omitted for zero outputs).
    """
    arguments = [
        f"%arg{i}: {argument.type}" for i, argument in enumerate(function.entry_block.arguments)
    ]
    result_types = [str(t) for t in function.output_types]
    results = f" -> {', '.join(result_types)}" if result_types else ""
    if len(arguments) <= 1:
        return [f"{pad}func @{function.name}({', '.join(arguments)}){results} {{"]
    continuation = " " * len(pad + f"func @{function.name}(")
    lines = [f"{pad}func @{function.name}({arguments[0]},"]
    for argument in arguments[1:-1]:
        lines.append(f"{continuation}{argument},")
    lines.append(f"{continuation}{arguments[-1]}){results} {{")
    return lines


def _print_block(
    block: Block,
    label_pad: str,
    names: dict[int, str],
    block_labels: Iterator[int],
    entry: bool = False,
) -> list[str]:
    """Render a block: optional ``^bbN`` label, then its ops in order.

    ``label_pad`` indents the ``^bbN`` label (unused for the function's
    entry block, which has no label); ops render one level deeper.
    """
    lines: list[str] = []
    op_pad = label_pad if entry else label_pad + _INDENT
    if not entry:
        arguments = ", ".join(
            f"%arg{i}: {argument.type}" for i, argument in enumerate(block.arguments)
        )
        lines.append(f"{label_pad}^bb{next(block_labels)}({arguments}):")
    for op in block.ops:
        lines.extend(_print_op(op, op_pad, names, block_labels))
    return lines


def _print_op(
    op: Op, pad: str, names: dict[int, str], block_labels: Iterator[int]
) -> list[str]:
    """Render one op line, with nested regions inline underneath."""
    op.opdef  # raises KeyError for unregistered op names
    line = ""
    if op.results:
        line += f"{', '.join(_value_name(names, r) for r in op.results)} = "
    line += f"etl.{op.name}("
    line += ", ".join(_value_name(names, operand) for operand in op.operands)
    line += ")"
    if op.attributes:
        pairs = ", ".join(
            f"{key} = {_attr_repr(op.attributes[key])}" for key in sorted(op.attributes)
        )
        line += f" attributes {{{pairs}}}"
    if op.results:
        line += f" : {', '.join(str(result.type) for result in op.results)}"
    if op.location is not None:
        line += f" {_loc_str(op.location)}"
    lines = [pad + line]
    if op.regions:
        lines[-1] += " {"
        for i, region in enumerate(op.regions):
            for nested in region.blocks:
                lines.extend(_print_block(nested, pad + _INDENT, names, block_labels))
            closer = "}" if i == len(op.regions) - 1 else "} {"
            lines.append(pad + closer)
    return lines


def _loc_str(location: Location) -> str:
    """``loc("file":line:col)`` — the column is omitted when None."""
    col = f":{location.col}" if location.col is not None else ""
    return f"loc({json.dumps(location.file)}:{location.line}{col})"


# --- attribute values --------------------------------------------------------


def _attr_repr(value: Any) -> str:
    """Deterministic, readable spelling of one attribute value.

    Conventions: ``None`` -> ``?``; strings JSON-quoted; sequences ->
    ``[a, b]``; dicts -> ``{k = v}`` with sorted keys; numpy arrays -> a
    compact ``ndarray<dtype[shape]>`` summary (payloads can be large — the
    text printer never dumps them; ``serialize_module`` carries the data).
    """
    if value is None:
        return "?"
    if isinstance(value, (Dim, DimExpr)):
        return _dim_str(value)
    if isinstance(value, bool):
        return repr(value)
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, np.ndarray):
        dims = ",".join(str(dim) for dim in value.shape)
        return f"ndarray<{_dtype_str(value.dtype)}[{dims}]>"
    if isinstance(value, np.generic):  # numpy scalar -> Python scalar
        return _attr_repr(value.item())
    if isinstance(value, (tuple, list)):
        return "[" + ", ".join(_attr_repr(item) for item in value) + "]"
    if isinstance(value, dict):
        pairs = ", ".join(f"{key} = {_attr_repr(value[key])}" for key in sorted(value))
        return f"{{{pairs}}}"
    return repr(value)
