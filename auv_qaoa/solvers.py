"""Solvers: brute force (ground truth), exact eigensolver, and QAOA."""
import itertools, time
import numpy as np
from qiskit.primitives import StatevectorSampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.minimum_eigensolvers import QAOA, NumPyMinimumEigensolver
from qiskit_optimization.optimizers import COBYLA

from .physics import tour_cost
from .qubo import build_full, build_reduced, decode


def brute_force(C, base=0):
    n = len(C)
    others = [k for k in range(n) if k != base]
    tours = [(base,) + p for p in itertools.permutations(others)]
    return sorted(((tour_cost(t, C), t) for t in tours))


def exact(C, base=0, reduced=False):
    qp = build_reduced(C, base) if reduced else build_full(C)
    t0 = time.time()
    r = MinimumEigenOptimizer(NumPyMinimumEigensolver()).solve(qp)
    tour = decode(r.x, len(C), base, reduced)
    return dict(solver="exact", tour=tour, energy=tour_cost(tour, C) if tour else None,
                seconds=time.time() - t0, qubits=qp.get_num_vars())


def qaoa_isolated(C, **kwargs):
    """Run qaoa() in a fresh process.

    Qiskit's native extension can crash when called from a web server's worker thread
    (as in Streamlit), so the dashboard runs each QAOA job in its own process.
    """
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as ex:
        return ex.submit(qaoa, np.asarray(C), **kwargs).result()


def qaoa(C, base=0, reduced=True, reps=3, maxiter=150, shots=4096, seed=1, optimum=None):
    qp = build_reduced(C, base) if reduced else build_full(C)
    n = len(C)
    history = []
    rng = np.random.default_rng(seed)
    solver = QAOA(StatevectorSampler(seed=seed, default_shots=shots), COBYLA(maxiter=maxiter),
                  reps=reps, initial_point=rng.uniform(0, np.pi, 2 * reps),
                  callback=lambda k, p, v, m: history.append(float(v)))
    t0 = time.time()
    r = MinimumEigenOptimizer(solver).solve(qp)
    secs = time.time() - t0
    tour = decode(r.x, n, base, reduced)
    feas = sum(s.probability for s in r.samples if decode(s.x, n, base, reduced))
    opt_p = (sum(s.probability for s in r.samples if decode(s.x, n, base, reduced) == optimum)
             if optimum else None)
    return dict(solver=f"qaoa-{'reduced' if reduced else 'full'}", tour=tour,
                energy=tour_cost(tour, C) if tour else None, seconds=secs,
                qubits=qp.get_num_vars(), reps=reps, seed=seed,
                feasible_prob=feas, optimal_prob=opt_p, history=history)
