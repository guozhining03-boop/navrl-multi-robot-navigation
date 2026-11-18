"""
Scenario builders for the course project.
Requirement #1: three different scenarios, multi-robot navigation,
6 robots and 15 static / dynamic / mixed obstacles.
"""
import numpy as np

ROBOT_RADIUS = 0.3
REGION_MIN, REGION_MAX = -15.0, 15.0
DYN_BOUND = 18.0
MIN_RADIUS, MAX_RADIUS = 0.3, 0.5
N_ROBOTS = 6
N_OBSTACLES = 15
DYN_RADIUS = 0.5
DYN_VEL_RANGE = (0.5, 1.5)


def sample_static_obstacles(rng, n, existing=None, clearance=1.0, tries=5000):
    existing = list(existing or [])
    out = []
    for _ in range(n):
        placed = False
        for _ in range(tries):
            r = rng.uniform(MIN_RADIUS, MAX_RADIUS)
            x = rng.uniform(REGION_MIN + r, REGION_MAX - r)
            y = rng.uniform(REGION_MIN + r, REGION_MAX - r)
            ok = all(np.hypot(x - ox, y - oy) >= r + orr + clearance
                     for ox, oy, orr in existing + out)
            if ok:
                out.append((float(x), float(y), float(r)))
                placed = True
                break
        if not placed:
            raise RuntimeError("failed to place a static obstacle after %d tries" % tries)
    return out


def _centers(obstacles):
    """Accept both (x, y, r) tuples and dict obstacles and return (x, y, r) tuples."""
    out = []
    for o in obstacles:
        if isinstance(o, dict):
            out.append((o["pos"][0], o["pos"][1], o["radius"]))
        else:
            out.append((o[0], o[1], o[2]))
    return out


def sample_dynamic_obstacles(rng, n, existing=None, clearance=1.0):
    existing = _centers(existing or [])
    out = []
    for _ in range(n):
        for _ in range(5000):
            x = rng.uniform(REGION_MIN, REGION_MAX)
            y = rng.uniform(REGION_MIN, REGION_MAX)
            if all(np.hypot(x - ox, y - oy) >= DYN_RADIUS + orr + clearance
                   for ox, oy, orr in existing + _centers(out)):
                speed = rng.uniform(*DYN_VEL_RANGE)
                heading = rng.uniform(0, 2 * np.pi)
                out.append({
                    "pos": np.array([x, y], dtype=float),
                    "vel": np.array([speed * np.cos(heading), speed * np.sin(heading)]),
                    "radius": DYN_RADIUS,
                })
                break
    return out


def robot_start_goal(seed_shift=0.0):
    xs = np.linspace(-10.0, 10.0, N_ROBOTS) + seed_shift
    starts = [np.array([x, -18.0]) for x in xs]
    goals = [np.array([x, 18.0]) for x in xs]
    return starts, goals


def build_scenario(kind, seed=0):
    """kind in {'static', 'dynamic', 'mixed'}"""
    rng = np.random.default_rng(seed)
    starts, goals = robot_start_goal(seed_shift=rng.uniform(-1.0, 1.0))
    static, dynamic = [], []

    if kind == "static":
        static = sample_static_obstacles(rng, N_OBSTACLES)
    elif kind == "dynamic":
        dynamic = sample_dynamic_obstacles(rng, N_OBSTACLES)
    elif kind == "mixed":
        n_static = N_OBSTACLES // 2
        static = sample_static_obstacles(rng, n_static)
        dynamic = sample_dynamic_obstacles(rng, N_OBSTACLES - n_static)
    else:
        raise ValueError(kind)

    return {
        "kind": kind,
        "seed": seed,
        "starts": starts,
        "goals": goals,
        "static": static,
        "dynamic": dynamic,
        "n_obstacles": len(static) + len(dynamic),
    }
