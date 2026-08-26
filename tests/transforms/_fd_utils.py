"""Shared numerical helpers for the transforms test suite.

The leading underscore keeps pytest from collecting this module. Both
`test_jvp.py` and `test_autodiff_rules.py` import from it.

All execution goes through the explicit staging pipeline
(lower -> compile -> load -> run) — no hidden staging anywhere, per the
design contract. Numerical references are numpy central finite differences,
computed in float64 (float32 inputs are upcast so the reference itself is
never polluted by float32 rounding).
"""

import numpy as np

import etl


def run_graph(graph, *args):
    """Run a graph through the explicit staging pipeline: lower, compile,
    load, run."""
    return etl.run(etl.load(etl.compile(etl.lower(graph))), *args)


def to_np(value):
    """Recursively replace etl.Tensor leaves with numpy arrays."""
    if isinstance(value, etl.Tensor):
        return value.numpy()
    if isinstance(value, (tuple, list)):
        return type(value)(to_np(v) for v in value)
    return value


def central_directional(fn, xs, tangents, eps):
    """Numeric directional derivative of `fn` at `xs` along `tangents`.

    Computes ``(fn(x + eps*t) - fn(x - eps*t)) / (2*eps)`` for each input
    (`None` tangents act as zero directions). Returns one array per fn
    output — a bare array for single-output fns, a tuple for tuple outputs.
    """
    ts = tuple(np.zeros_like(x) if t is None else t for x, t in zip(xs, tangents))
    plus = fn(*(x + eps * t for x, t in zip(xs, ts)))
    minus = fn(*(x - eps * t for x, t in zip(xs, ts)))
    if isinstance(plus, tuple):
        return tuple((p - m) / (2 * eps) for p, m in zip(plus, minus))
    return (plus - minus) / (2 * eps)


def central_jacobian(fn, xs, arg, eps):
    """Numeric Jacobian of `fn` w.r.t. `xs[arg]` (rows = output elements).

    Perturbs one input element at a time with the central difference; the
    result has shape ``(num_output_elems, num_input_elems)``.
    """
    x = xs[arg]
    cols = []
    for idx in np.ndindex(x.shape):
        xp = x.copy()
        xm = x.copy()
        xp[idx] += eps
        xm[idx] -= eps
        args_p = list(xs)
        args_m = list(xs)
        args_p[arg] = xp
        args_m[arg] = xm
        col = (np.asarray(fn(*args_p)) - np.asarray(fn(*args_m))) / (2 * eps)
        cols.append(np.ravel(col))
    return np.stack(cols, axis=1)


def central_grad(fn, xs, arg, eps):
    """Numeric gradient of a scalar-output `fn` w.r.t. `xs[arg]`."""
    return central_jacobian(fn, xs, arg, eps)[0].reshape(xs[arg].shape)


def run_jvp(fn, tangents, specs, xs, ts):
    """Transform via the spec callable, verify, run; returns
    ``(graph, primal_outputs, tangent_outputs)`` (numpy trees).

    `ts` entries may be `None` (zero tangent); runtime args mirror the
    transformed graph's input tree ``(primal tree, flat tangent tuple)``.
    """
    graph = etl.jvp(fn, tangents)(*specs)
    graph.verify()
    out = to_np(run_graph(graph, tuple(xs), tuple(ts)))
    return graph, out[0], out[1]


def run_vjp(fn, cotangents, specs, xs, cs):
    """`vjp(fn, cotangents)` via the spec callable, verify, run; returns
    ``(graph, primal_outputs, input_cotangents)`` (numpy trees)."""
    graph = etl.vjp(fn, cotangents)(*specs)
    graph.verify()
    out = to_np(run_graph(graph, tuple(xs), tuple(cs)))
    return graph, out[0], out[1]


def run_grad(fn, argnums, specs, *xs):
    """`grad(fn, argnums)` via the spec callable, verify, run; returns the
    gradient(s): a bare array for an int argnum, a tuple otherwise."""
    graph = etl.grad(fn, argnums)(*specs)
    graph.verify()
    return to_np(run_graph(graph, *xs))
