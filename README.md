# micrograd-py

[![CI](https://github.com/bhavana-sanal/micrograd-py/actions/workflows/ci.yml/badge.svg)](https://github.com/bhavana-sanal/micrograd-py/actions) ![License](https://img.shields.io/badge/license-MIT-blue)

A tiny scalar **autograd engine** (reverse-mode automatic differentiation) and a small
neural-network library on top of it — about 150 lines of pure Python, no dependencies.
Inspired by Andrej Karpathy's *micrograd*, re-implemented with extra ops (`exp`, `log`, `relu`, `tanh`, powers, division).

```python
from micrograd import Value
a, b = Value(2.0), Value(-3.0)
c = (a * b + b ** 2).tanh()
c.backward()
print(a.grad, b.grad)
```

## Train a network
```bash
python demo.py          # trains an MLP on two moons, prints loss/accuracy + ASCII decision boundary
```

## How it works
Each `Value` remembers its parents and a closure that pushes gradients back via the chain rule.
`backward()` topologically sorts the graph and runs those closures in reverse.

## Tests
```bash
pip install pytest && pytest
```
Gradients are verified against finite differences, and an MLP is trained to solve XOR.
