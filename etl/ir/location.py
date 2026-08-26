"""Source locations attached to IR operations.

Locations record where in user code an op was created. They are carried on
``Op.location`` and never affect semantics — they exist for error messages,
debugging, and ``pretty_print`` output. Error messages include a location
(e.g. ``model.py:83``) whenever a graph location exists.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Location:
    """A source-code position.

    Attributes:
        file: Path of the source file (as written at trace time).
        line: 1-based line number.
        col: 1-based column number.
        code_snippet: Optional single-line snippet of the source at that
            position (None when unavailable).
    """

    file: str
    line: int
    col: int
    code_snippet: str | None = None

    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.col}"

    @staticmethod
    def unknown() -> "Location":
        """A placeholder location for IR built without source info."""
        return Location("<unknown>", 0, 0, None)
