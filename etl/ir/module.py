"""Modules: versioned containers of functions — the unit of compilation.

A ``Module`` is what ``etl.trace`` produces, what transforms rewrite, and what
backends consume. v1 ``Graph``s produce exactly one function (conventionally
named "main"); multi-function modules exist for the `call` op and future
expansion (see CONTEXT.md, "call + multi-function modules").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
from typing import TYPE_CHECKING, Any

from .version import IR_FORMAT_VERSION

if TYPE_CHECKING:
    from .function import Function


@dataclass
class Module:
    """The top-level IR unit: a versioned container of functions.

    Attributes:
        name: Module name (display/identification; not a serialization key).
        functions: The module's functions (at least one; unique names).
        metadata: Free-form JSON-able annotations.
        version: ``IR_FORMAT_VERSION`` the module was built against.
    """

    name: str = "main"
    functions: list[Function] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: int = IR_FORMAT_VERSION
    _op_ids: Any = field(default_factory=count, repr=False)
    _value_ids: Any = field(default_factory=count, repr=False)

    def get_function(self, name: str) -> "Function":
        """Look up a function by name.

        Raises:
            KeyError: If no function with this name exists.
        """
        for function in self.functions:
            if function.name == name:
                return function
        raise KeyError(f"no function named '{name}' in module '{self.name}'")

    @property
    def main(self) -> "Function":
        """The single function of a v1 module.

        Raises:
            ValueError: If the module does not have exactly one function.
        """
        if len(self.functions) != 1:
            raise ValueError(
                f"module '{self.name}' has {len(self.functions)} functions; "
                "'.main' requires exactly one"
            )
        return self.functions[0]

    def add_function(self, function: "Function") -> "Function":
        """Append ``function`` and wire its parent pointer. Returns function."""
        function.parent = self
        self.functions.append(function)
        return function

    def new_op_id(self) -> int:
        """Allocate a fresh module-unique op id (used by the ``Builder``)."""
        return next(self._op_ids)

    def new_value_id(self) -> int:
        """Allocate a fresh module-unique value id (used by the ``Builder``)."""
        return next(self._value_ids)
