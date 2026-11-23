"""Run the full pipeline: physics -> QUBO -> solvers -> report + interactive map.

Usage:
    python main.py                         # exact + reduced 9-qubit QAOA (fast)
    python main.py --full                  # also run the 16-qubit QAOA (slower)
    python main.py --seeds 1 2 3 --reps 3  # choose seeds / layers
    python main.py --exact-only            # CI mode: exact eigensolver only
"""
import argparse, json, os, warnings
warnings.filterwarnings("ignore")
import numpy as np
from auv_qaoa import Mission, cost_matrix, still_water_matrix, tour_cost
from auv_qaoa import solvers
from auv_qaoa.visualize import route_map, convergence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/mission.json")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--maxiter", type=int, default=150)
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--full", action="store_true", help="also run 16-qubit full-encoding QAOA")
    ap.add_argument("--exact-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    m = Mission.from_json(a.config)
    C, S = cost_matrix(m), still_water_matrix(m)
    np.set_printoptions(precision=3, suppress=True)
    print("Asymmetric cost matrix C[i,j]:\n", C)

    ranked = solvers.brute_force(C, m.base_node)
    best_cost, best = ranked[0]
    print("\nAll tours (brute force):")
    for c, t in ranked:
        print(f"  {'-'.join(map(str, t))}-{t[0]}  energy {c:7.3f}  still-water {tour_cost(t, S):6.3f}"
              f"  +{100 * (c / best_cost - 1):5.1f}%")

    results = {"cost_matrix": C.tolist(), "brute_force": [(c, list(t)) for c, t in ranked]}
    ex = solvers.exact(C, m.base_node)
    print(f"\nExact eigensolver: {ex['tour']}  energy {ex['energy']:.3f}  ({ex['seconds']:.2f}s)")
    results["exact"] = ex

    histories = {}
    if not a.exact_only:
        configs = [("reduced", True)] + ([("full", False)] if a.full else [])
        for label, reduced in configs:
            runs = []
            for s in a.seeds:
                r = solvers.qaoa(C, m.base_node, reduced=reduced, reps=a.reps, maxiter=a.maxiter,
                                 shots=a.shots, seed=s, optimum=best)
                histories[f"{label} seed {s}"] = r.pop("history")
                runs.append(r)
                ok = "OPTIMAL" if r["tour"] == best else ("feasible" if r["tour"] else "infeasible")
                print(f"QAOA {label:7s} ({r['qubits']} qubits) seed {s}: {r['tour']}  [{ok}]  "
                      f"P(feasible)={r['feasible_prob']:.4f}  P(optimal)={r['optimal_prob']:.4f}  "
                      f"{r['seconds']:.1f}s")
            hits = sum(r["tour"] == best for r in runs)
            print(f"  -> {label}: optimum found in {hits}/{len(runs)} runs")
            results[f"qaoa_{label}"] = runs

    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(results, f, indent=1, default=lambda o: list(o) if isinstance(o, tuple) else str(o))
    route_map(m, best, C, f"Energy-optimal tour {'→'.join(map(str, best))}→{best[0]}  "
                          f"(energy {best_cost:.3f})").write_html(os.path.join(a.out, "route_map.html"))
    if histories:
        convergence(histories).write_html(os.path.join(a.out, "convergence.html"))
    print(f"\nSaved results.json, route_map.html{', convergence.html' if histories else ''} to {a.out}/")


if __name__ == "__main__":
    main()
