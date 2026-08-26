"""v1 control-flow restriction of the transforms.

Region-bearing control-flow ops (`etl.cond` → `if`, `etl.while_loop` → `while`,
`etl.scan` → `while`) are NOT vectorizable in v1 and have no AD rules:
`vectorize`/`vmap` raise `TransformError` naming the op ("region-bearing ...
not vectorizable in v1"), and `grad`/`jvp`/`vjp` raise `TransformError` for the
missing rule. The restriction is TRANSFORM-only: plainly tracing the same
functions still works (see `./etl/transforms/CONTEXT.md` "v1 scope vs
deferred").
"""

import pytest

import etl

SPEC = etl.TensorSpec((4, 3), etl.float32)
BATCHED_SPEC = etl.TensorSpec((8, 4, 3), etl.float32)


def f_cond(x):
    # Rank-2 input: a rank-0 (scalar) input would fail earlier with an
    # in_axes out-of-range error (a mapped entry on a rank-0 spec), before the
    # region check.
    return etl.sum(etl.cond(etl.sum(x) > 0, lambda: x * 2, lambda: -x))


def f_while(x):
    def body(s):
        return s + 1

    init = etl.sum(x) * 0
    return etl.while_loop(lambda s: s < 4, body, init)


def f_scan(x, y):
    # `x` is the mapped input; `y` (unmapped) feeds the scan. Scanning over a
    # MAPPED xs would fail one op earlier (gather batching with mapped data +
    # unmapped indices is a separate v1 deferral), before the region check.
    def step(c, y_step):
        return c + y_step, y_step * 2

    return etl.sum(etl.scan(step, y[0] * 0, y)[1]) * etl.sum(x)


def _op_names(graph):
    names = set()

    def walk(block):
        for op in block.ops:
            names.add(op.name)
            for region in op.regions:
                for nested_block in region.blocks:
                    walk(nested_block)

    walk(graph.module.main.entry_block)
    return names


class TestVmapControlFlow:
    def test_vmap_over_cond(self):
        with pytest.raises(
            etl.TransformError,
            match=r"cannot batch op 'if'.*not vectorizable in v1",
        ):
            etl.vmap(f_cond)(BATCHED_SPEC)

    def test_vmap_over_while_loop(self):
        with pytest.raises(
            etl.TransformError,
            match=r"cannot batch op 'while'.*not vectorizable in v1",
        ):
            etl.vmap(f_while)(BATCHED_SPEC)

    def test_vmap_over_scan(self):
        # `etl.scan` lowers to the same region-bearing 'while' op as
        # `etl.while_loop`, so the error names 'while'.
        with pytest.raises(
            etl.TransformError,
            match=r"cannot batch op 'while'.*not vectorizable in v1",
        ):
            etl.vmap(f_scan, in_axes=(0, None))(BATCHED_SPEC, SPEC)


class TestVectorizeControlFlowTrace:
    @pytest.mark.parametrize(
        "fn,args,axes,opname",
        [
            (f_cond, (SPEC,), 0, "if"),
            (f_while, (SPEC,), 0, "while"),
            (f_scan, (SPEC, SPEC), (0, None), "while"),
        ],
        ids=["cond", "while_loop", "scan"],
    )
    def test_vectorize_trace_directly(self, fn, args, axes, opname):
        graph = etl.trace(fn, *args)
        with pytest.raises(
            etl.TransformError,
            match=rf"cannot batch op '{opname}'.*not vectorizable in v1",
        ):
            etl.vectorize(graph, axes)


class TestAdControlFlow:
    @pytest.mark.parametrize(
        "kind,build,expected",
        [
            ("grad", lambda: etl.grad(f_cond)(SPEC), r"no VJP rule for op 'if'"),
            ("jvp", lambda: etl.jvp(f_cond, SPEC)(SPEC), r"no JVP rule for op 'if'"),
            ("vjp", lambda: etl.vjp(f_cond)(SPEC), r"no VJP rule for op 'if'"),
        ],
        ids=["grad", "jvp", "vjp"],
    )
    def test_ad_over_cond(self, kind, build, expected):
        with pytest.raises(etl.TransformError, match=expected):
            build()


class TestPlainTraceStillWorks:
    """The control-flow restriction is transform-only: plain tracing of the
    same functions succeeds and the region-bearing op really is present."""

    @pytest.mark.parametrize(
        "fn,args,opname",
        [
            (f_cond, (SPEC,), "if"),
            (f_while, (SPEC,), "while"),
            (f_scan, (SPEC, SPEC), "while"),
        ],
        ids=["cond", "while_loop", "scan"],
    )
    def test_plain_trace_succeeds(self, fn, args, opname):
        graph = etl.trace(fn, *args)
        graph.verify()
        assert opname in _op_names(graph)
