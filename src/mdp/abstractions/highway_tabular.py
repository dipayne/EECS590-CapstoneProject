import numpy as np

def discretize_speed(v: float) -> int:
    # 3 bins: slow/medium/fast (tune later)
    if v < 20:
        return 0
    if v < 30:
        return 1
    return 2

def discretize_dist(d: float) -> int:
    # 3 bins: close/medium/far (tune later)
    if d < 10:
        return 0
    if d < 30:
        return 1
    return 2

def obs_to_tabular_state(obs: np.ndarray) -> int:
    """
    highway-env obs often comes as a (N, features) array.
    We'll use a simple heuristic:
      - ego is row 0
      - columns include presence/x/y/vx/vy depending on config
    This function is intentionally conservative and can be refined.
    """
    # If obs is (5,5) like you saw, use row0 as ego.
    ego = obs[0]

    # Heuristic indices (common highway-env format: presence, x, y, vx, vy)
    # If your config differs, we’ll adjust later.
    x = float(ego[1])
    y = float(ego[2])
    vx = float(ego[3])
    lane = int(round(y))  # coarse lane index proxy

    speed_bin = discretize_speed(vx)

    # crude front/rear distances using other vehicles' x positions
    others = obs[1:]
    front_d = 999.0
    rear_d = 999.0
    for veh in others:
        if float(veh[0]) < 0.5:  # presence flag
            continue
        ox = float(veh[1])
        dx = ox - x
        if dx >= 0:
            front_d = min(front_d, dx)
        else:
            rear_d = min(rear_d, abs(dx))

    front_bin = discretize_dist(front_d if front_d < 999 else 1000.0)
    rear_bin = discretize_dist(rear_d if rear_d < 999 else 1000.0)

    # Pack into a single integer state ID
    # lane in [0..4] (safe cap), speed 0..2, front 0..2, rear 0..2
    lane = max(0, min(4, lane))
    s = (((lane * 3) + speed_bin) * 3 + front_bin) * 3 + rear_bin
    return int(s)

def tabular_state_space_size(max_lanes: int = 5) -> int:
    return max_lanes * 3 * 3 * 3
