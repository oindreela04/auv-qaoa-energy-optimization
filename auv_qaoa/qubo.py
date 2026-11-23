"""ATSP -> QUBO / Ising compilation with DOcplex + Qiskit Optimization."""
import numpy as np
from docplex.mp.model import Model
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.converters import QuadraticProgramToQubo


def build_full(C: np.ndarray):
    """Position encoding with N^2 binaries x_{i,t}: node i visited at step t."""
    n = len(C)
    mdl = Model("auv_atsp_full")
    x = {(i, t): mdl.binary_var(f"x_{i}_{t}") for i in range(n) for t in range(n)}
    mdl.minimize(mdl.sum(C[i, j] * x[i, t] * x[j, (t + 1) % n]
                         for i in range(n) for j in range(n) if i != j for t in range(n)))
    for t in range(n):
        mdl.add_constraint(mdl.sum(x[i, t] for i in range(n)) == 1, f"step_{t}")
    for i in range(n):
        mdl.add_constraint(mdl.sum(x[i, t] for t in range(n)) == 1, f"node_{i}")
    return from_docplex_mp(mdl)


def build_reduced(C: np.ndarray, base: int = 0):
    """Base fixed at step 0: (N-1)^2 binaries over the remaining nodes and steps 1..N-1."""
    n = len(C)
    others = [k for k in range(n) if k != base]
    steps = range(1, n)
    mdl = Model("auv_atsp_reduced")
    x = {(i, t): mdl.binary_var(f"x_{i}_{t}") for i in others for t in steps}
    obj = mdl.sum(C[base, i] * x[i, 1] for i in others)
    obj += mdl.sum(C[i, base] * x[i, n - 1] for i in others)
    obj += mdl.sum(C[i, j] * x[i, t] * x[j, t + 1]
                   for i in others for j in others if i != j for t in range(1, n - 1))
    mdl.minimize(obj)
    for t in steps:
        mdl.add_constraint(mdl.sum(x[i, t] for i in others) == 1, f"step_{t}")
    for i in others:
        mdl.add_constraint(mdl.sum(x[i, t] for t in steps) == 1, f"node_{i}")
    return from_docplex_mp(mdl)


def to_ising(qp):
    qubo = QuadraticProgramToQubo().convert(qp)
    op, offset = qubo.to_ising()
    return op, offset


def decode(x, n: int, base: int = 0, reduced: bool = False):
    """Bitstring -> tour starting at base, or None if it violates the constraints."""
    x = np.round(np.asarray(x)).astype(int)
    m = n - 1 if reduced else n
    if x.size != m * m:
        return None
    X = x.reshape(m, m)                       # rows = nodes, cols = steps
    if not (np.all(X.sum(0) == 1) and np.all(X.sum(1) == 1)):
        return None
    if reduced:
        others = [k for k in range(n) if k != base]
        return tuple([base] + [others[int(np.argmax(X[:, t]))] for t in range(m)])
    seq = [int(np.argmax(X[:, t])) for t in range(n)]
    k = seq.index(base)
    return tuple(seq[k:] + seq[:k])
