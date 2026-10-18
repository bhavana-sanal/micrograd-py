import math, random
from micrograd import Value, MLP


def numeric_grad(f, x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2 * h)


def test_gradients_match_finite_differences():
    f = lambda v: ((v * 3 + 2) ** 2 / (v * v + 1) + v.tanh() - (v * 0.5).exp()) if isinstance(v, Value) else None
    def g(x):
        v = Value(x)
        return f(v).data
    x = Value(0.7)
    f(x).backward()
    assert math.isclose(x.grad, numeric_grad(g, 0.7), rel_tol=1e-5)


def test_reused_node_accumulates():
    a = Value(3.0)
    (a * a + a).backward()
    assert a.grad == 7.0


def test_relu_and_log():
    a = Value(-2.0); b = Value(4.0)
    (a.relu() + b.log()).backward()
    assert a.grad == 0.0 and math.isclose(b.grad, 0.25)


def test_mlp_learns_xor():
    random.seed(1)
    xs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    ys = [-1, 1, 1, -1]
    net = MLP(2, [6, 6, 1])
    for _ in range(300):
        loss = sum((net(x) - y) ** 2 for x, y in zip(xs, ys))
        net.zero_grad()
        loss.backward()
        for p in net.parameters():
            p.data -= 0.05 * p.grad
    assert all((net(x).data > 0) == (y > 0) for x, y in zip(xs, ys))
