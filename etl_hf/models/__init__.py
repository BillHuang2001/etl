"""Per-family graph builders for ``etl_hf``.

This package holds one module per model family (``llama``, ``gpt2``) plus the
shared block assembly helpers in ``_common``. Nothing else belongs here.

Family modules are imported directly by their qualified paths
(``etl_hf.models.llama`` / ``etl_hf.models.gpt2``); this package deliberately
does **not** re-export them, to keep imports acyclic.
"""

__all__: list[str] = []
