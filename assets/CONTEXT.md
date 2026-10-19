# assets — README-only SVG assets

Hand-written SVG used by `README.md` only: no code imports them, there is no build step, no generated output and no dependency — edit the files directly.

`etl-mark.svg` — 128×128 (viewBox `0 0 128 128`) project mark, an SSA dataflow glyph (two inputs → one op node → one output), displayed at 96×96.
`etl-stack.svg` — 1200×560 (viewBox `0 0 1200 560`) hero diagram of the explicit staging pipeline, displayed full width.

Palette contract (both files): card `#0d1117`, inner card `#161b22`, border `#30363d`, text `#e6edf3`, muted `#8b949e` (opacity variants allowed); accents violet `#7c5cff` (frontend/language), cyan `#22d3ee` (graph/IR), green `#34d399` (runtime/execution), amber `#f0b429` (compilers). System sans for prose labels, `ui-monospace` for stage and function names.

**Rule:** the diagram text MUST stay in sync with the public stage names in `./etl/CONTEXT.md` — `@etl.defn`, `Graph`, `LoweredProgram`, `CompiledArtifact`, `Executable`, `Tensor` — the verbs between them (`trace`, `lower`, `compile`, `load`, `run`) and the backends (`numpy interpreter`, `IREE · XLA (PJRT) · TVM`). Renaming a stage in the library updates `etl-stack.svg` in the same change.
