"""Train a small MLP on the two-moons dataset and print the decision boundary as ASCII."""
import math, random
from micrograd import MLP, Value

random.seed(0)


def moons(n=100, noise=0.1):
    pts = []
    for i in range(n):
        if i < n // 2:
            t = math.pi * i / (n // 2)
            pts.append(([math.cos(t) + random.gauss(0, noise), math.sin(t) + random.gauss(0, noise)], 1))
        else:
            t = math.pi * (i - n // 2) / (n // 2)
            pts.append(([1 - math.cos(t) + random.gauss(0, noise), 0.5 - math.sin(t) + random.gauss(0, noise)], -1))
    return pts


data = moons()
net = MLP(2, [16, 16, 1])
for epoch in range(100):
    losses = [(1 - y * net(x)).relu() for x, y in data]  # hinge loss
    reg = 1e-4 * sum(p * p for p in net.parameters())
    loss = sum(losses, Value(0)) * (1 / len(data)) + reg
    acc = sum((net(x).data > 0) == (y > 0) for x, y in data) / len(data)
    net.zero_grad()
    loss.backward()
    lr = 0.1 - 0.09 * epoch / 100
    for p in net.parameters():
        p.data -= lr * p.grad
    if epoch % 10 == 0 or epoch == 99:
        print(f"epoch {epoch:3d}  loss {loss.data:.4f}  acc {acc:.0%}")

for yy in range(10, -6, -1):
    print("".join("#" if net([xx / 10, yy / 5]).data > 0 else "." for xx in range(-15, 26)))
