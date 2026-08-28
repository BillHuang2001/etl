"""Curated example registry for etl.bench (package form).

Each :class:`Example` bundles:

- ``graph``: an ``@etl.defn`` graph taking fixed static ``TensorSpec`` inputs,
- ``specs``: the input specs (tuple of ``etl.TensorSpec``),
- ``numpy_ref``: a pure-numpy reference (same inputs → numpy output(s)),
- ``torch_ref``: an OPTIONAL factory that imports torch INSIDE the function
  body and raises the clear ``pip install etl[bench]`` hint error when torch
  is absent — merely listing examples (``import etl.bench``) never imports
  torch.

References must match the graph's output structure exactly (single ndarray or
a tuple of ndarrays for multi-output graphs).

Shapes are deliberately small so a full conformance run stays well under a
couple of seconds on the default numpy backend.

Examples are grouped by ``category`` (``"micro"``, ``"grad"``,
``"vectorize"``, ``"large"``); the CLI ``--examples`` accepts either example
names or category names (expanded via :func:`expand_names`). Registry order
follows module import order (``micro`` first, then ``grad``, ``vectorize``,
``large`` — later modules self-register via ``register_all`` at import time).
"""
from . import base  # noqa: F401  (registry + shared infra; import first)
from . import micro, grad, vectorize, large  # noqa: F401  (self-register on import)

from .base import Example, UnknownExampleError, generate_inputs

__all__ = [
    "Example",
    "UnknownExampleError",
    "list_examples",
    "get_example",
    "generate_inputs",
    "list_categories",
    "expand_names",
]


def list_examples() -> list:
    """Return the registered example names (registry order)."""
    return list(base._REGISTRY)


def get_example(name: str) -> Example:
    """Return the :class:`Example` registered under ``name``.

    Raises:
        UnknownExampleError: unknown name — the message lists all available
            names.
    """
    try:
        return base._REGISTRY[name]
    except KeyError:
        available = ", ".join(list_examples())
        raise UnknownExampleError(
            f"unknown example {name!r}; available examples: {available}"
        ) from None


def list_categories() -> list:
    """Return the distinct example categories in order of first appearance in
    the registry."""
    seen = []
    for example in base._REGISTRY.values():
        if example.category not in seen:
            seen.append(example.category)
    return seen


def expand_names(entries) -> list:
    """Expand ``entries`` into a flat list of example names.

    Each entry that equals a category name (see :func:`list_categories`)
    expands to all example names of that category (registry order); any
    other entry is validated via :func:`get_example` and kept as-is. Unknown
    names raise :class:`UnknownExampleError`.
    """
    categories = list_categories()
    names = []
    for entry in entries:
        if entry in categories:
            names.extend(
                example.name
                for example in base._REGISTRY.values()
                if example.category == entry
            )
        else:
            get_example(entry)  # validates; raises UnknownExampleError
            names.append(entry)
    return names
