"""Shared fixtures for tests/trace/.

Executing a traced Graph on the numpy backend requires the explicit staging
pipeline (trace → lower → compile → load → run; `etl.build` accepts a defn/
callable, NOT an already-traced Graph). The `run_graph` fixture wraps that;
`as_numpy` recursively converts `etl.Tensor` results into numpy arrays for
assertions. Fixtures are injected by name — see the test modules.
"""

import pytest

import etl


@pytest.fixture
def run_graph():
    """Explicitly stage + execute a traced `etl.Graph` on the numpy backend.

    Usage: ``result = run_graph(graph, np_array_a, np_array_b)``. Returns the
    structured outputs (rebuilt per the graph's output tree).
    """

    def _run(graph, *args):
        lowered = etl.lower(graph)
        artifact = etl.compile(lowered)
        executable = etl.load(artifact, device="cpu")
        return etl.run(executable, *args)

    return _run


@pytest.fixture
def as_numpy():
    """Recursively convert `etl.Tensor` leaves of a (structured) result to
    numpy arrays; non-tensor values pass through unchanged."""

    def _as_numpy(value):
        if isinstance(value, etl.Tensor):
            return value.numpy()
        if isinstance(value, (tuple, list)):
            return type(value)(_as_numpy(v) for v in value)
        if isinstance(value, dict):
            return {key: _as_numpy(v) for key, v in value.items()}
        return value

    return _as_numpy
