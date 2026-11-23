"""Ocean-current physics: asymmetric leg costs for an AUV."""
from dataclasses import dataclass, field
import json
import numpy as np


@dataclass
class Mission:
    names: list
    coords: np.ndarray            # (N, 2) waypoint coordinates
    current: np.ndarray           # (2,) uniform current vector (u, v)
    base_speed: float = 2.0       # AUV still-water speed
    base_node: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.coords)

    @classmethod
    def from_json(cls, path: str) -> "Mission":
        with open(path) as f:
            d = json.load(f)
        return cls(names=[w["name"] for w in d["waypoints"]],
                   coords=np.array([w["xy"] for w in d["waypoints"]], float),
                   current=np.array(d["current"], float),
                   base_speed=float(d.get("base_speed", 2.0)),
                   base_node=int(d.get("base_node", 0)), meta=d)


def effective_speed(mission: Mission, i: int, j: int) -> float:
    """V_eff = V_b + V_c . d_hat_ij (projection model)."""
    d = mission.coords[j] - mission.coords[i]
    return mission.base_speed + float(mission.current @ (d / np.linalg.norm(d)))


def cost_matrix(mission: Mission) -> np.ndarray:
    """C_ij = distance_ij / V_eff_ij (travel time, proportional to energy at constant power)."""
    if np.linalg.norm(mission.current) >= mission.base_speed:
        raise ValueError("Current speed must be below AUV speed, or some legs are impossible.")
    n = mission.n
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = np.linalg.norm(mission.coords[j] - mission.coords[i])
                C[i, j] = dist / effective_speed(mission, i, j)
    return C


def still_water_matrix(mission: Mission) -> np.ndarray:
    """Symmetric baseline: same distances, no current."""
    diff = mission.coords[:, None, :] - mission.coords[None, :, :]
    return np.linalg.norm(diff, axis=-1) / mission.base_speed


def tour_cost(tour, C) -> float:
    return float(sum(C[tour[k], tour[(k + 1) % len(tour)]] for k in range(len(tour))))
