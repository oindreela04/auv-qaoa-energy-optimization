import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auv_qaoa import Mission, cost_matrix, tour_cost
from auv_qaoa.qubo import build_full, build_reduced, decode
from auv_qaoa import solvers

CFG = os.path.join(os.path.dirname(__file__), "..", "config", "mission.json")


def mission():
    return Mission.from_json(CFG)


def test_cost_matrix_is_asymmetric_and_positive():
    C = cost_matrix(mission())
    off = ~np.eye(len(C), dtype=bool)
    assert np.all(C[off] > 0)
    assert not np.allclose(C, C.T)
    assert np.isclose(C[3, 0] / C[0, 3], 2.755, atol=1e-2)


def test_current_too_strong_raises():
    m = mission(); m.current = np.array([3.0, 0.0])
    try:
        cost_matrix(m); assert False
    except ValueError:
        pass


def test_brute_force_optimum():
    cost, tour = solvers.brute_force(cost_matrix(mission()))[0]
    assert tour == (0, 1, 3, 2)
    assert np.isclose(cost, 10.4668, atol=1e-3)


def test_qubo_sizes():
    C = cost_matrix(mission())
    assert build_full(C).get_num_vars() == 16
    assert build_reduced(C).get_num_vars() == 9


def test_decode():
    # node i at step t; tour 0->1->3->2
    X = np.zeros((4, 4)); X[0, 0] = X[1, 1] = X[3, 2] = X[2, 3] = 1
    assert decode(X.flatten(), 4) == (0, 1, 3, 2)
    assert decode(np.zeros(16), 4) is None


def test_exact_matches_brute_force():
    C = cost_matrix(mission())
    assert solvers.exact(C)["tour"] == solvers.brute_force(C)[0][1]
    assert solvers.exact(C, reduced=True)["tour"] == solvers.brute_force(C)[0][1]
