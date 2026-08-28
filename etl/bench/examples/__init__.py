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

Examples are grouped by ``category`` into THREE groups — ``"op"`` (single
ops and op-level compositions: micro, grad, vectorize, op_large, and the
later-phase ``op_*`` modules), ``"block"`` (whole-block compositions: the
``block_*`` modules), and ``"e2e"`` (multi-run end-to-end procedures: the
``e2e_*`` modules, driven through the optional ``Example.runner`` path). Each
example also carries a tuple of ``tags`` (subgroup selectors such as
``"micro"``, ``"grad"``, ``"vectorize"``, ``"large"``, ``"control-flow"``,
``"vmap"``, ``"custom"``, ``"xla"``).

The CLI ``--examples`` (and :func:`expand_names`) accepts example names,
category names, or tag names — resolution precedence per entry: (1) category
name → all examples of that category; (2) exact example name via
:func:`get_example` (an exact name wins over a same-named tag); (3) tag name
→ all examples carrying that tag.
Registry order follows module import order: ``op`` category first (``micro``,
``grad``, ``vectorize``, ``op_large``, then the later-phase ``op_*``
modules), then ``block`` (``block_transformer`` + the later-phase ``block_*``
modules), then ``e2e`` (``e2e_train``, ``e2e_infer``) — later modules
self-register via ``register_all`` at import time.
"""
from . import base  # noqa: F401  (registry + shared infra; import first)
from . import micro, grad, vectorize, op_large  # noqa: F401  (op category)
from . import (  # noqa: F401  (op category, filled by a later phase)
    op_basic, op_matmul, op_control_flow, op_grad2,
    op_vmap2, op_custom, op_xla,
)
from . import (  # noqa: F401  (block category)
    block_transformer, block_rnn, block_conv, block_mlp, block_opt,
)
from . import e2e_train, e2e_infer  # noqa: F401  (e2e category)

from .base import Example, UnknownExampleError, generate_inputs

__all__ = [
    "Example",
    "UnknownExampleError",
    "list_examples",
    "get_example",
    "generate_inputs",
    "list_categories",
    "list_tags",
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


def list_tags() -> list:
    """Return the distinct example tags in order of first appearance in the
    registry (like :func:`list_categories`)."""
    seen = []
    for example in base._REGISTRY.values():
        for tag in example.tags:
            if tag not in seen:
                seen.append(tag)
    return seen


def expand_names(entries) -> list:
    """Expand ``entries`` into a flat list of example names.

    Resolution precedence per entry (documented):
    (1) a category name (see :func:`list_categories`) expands to all example
        names of that category (registry order);
    (2) otherwise an exact example name (see :func:`get_example`) is kept
        as-is — an exact name wins over a tag of the same string;
    (3) otherwise a tag name (see :func:`list_tags`) expands to all example
        names carrying that tag (registry order).

    Unknown names raise :class:`UnknownExampleError`.
    """
    categories = list_categories()
    tags = list_tags()
    names = []
    for entry in entries:
        if entry in categories:
            names.extend(
                example.name
                for example in base._REGISTRY.values()
                if example.category == entry
            )
        elif entry in base._REGISTRY:
            # Exact example name — wins over a same-named tag.
            names.append(entry)
        elif entry in tags:
            names.extend(
                example.name
                for example in base._REGISTRY.values()
                if entry in example.tags
            )
        else:
            get_example(entry)  # validates; raises UnknownExampleError
    return names
