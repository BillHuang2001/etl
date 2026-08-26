# tests/core — value-model tests

## Intent

pytest tests for `etl/core` (sibling package — see `../../etl/core/CONTEXT.md` for the contract under test): errors, dtypes, symbolic dims, specs, concrete tensors, symbolic tensors, devices, pytrees.

## Structure

| File | Covers |
|---|---|
| `test_tensor.py` | `Tensor` (attrs, zero-copy `.numpy()`, structural eq, unhashable), creators (`tensor`/`zeros`/`ones`/`full`/`empty`/`from_numpy`), DLPack round-trips (`__dlpack__`, `from_dlpack`, numpy + optional torch interop via `pytest.importorskip`) |

## Notes for agents

- Test files import from `etl` directly (root `conftest.py` puts the repo root on `sys.path`).
- Torch interop tests must call `pytest.importorskip("torch")` inside the test (torch is optional; not installed in the default env → those tests skip).
- DLPack gotcha (numpy >= 2.0): `np.from_dlpack` and etl `from_dlpack` take an *object exposing* `__dlpack__` — raw PyCapsules are rejected (capsules have no `__dlpack__` method).
- Keep tests fast (<2s per file), CPU only, small shapes (<= ~16 elements).
