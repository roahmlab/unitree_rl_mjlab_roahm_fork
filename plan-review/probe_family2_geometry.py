"""Adversarial geometry probe for the planned `sequential_two_barrier` family.

EE-space proxy, deliberately OPTIMISTIC for the plan:
  * protected sweeps = end-effector polylines only (the real builder protects
    whole-arm capsule-axis sweeps; FINDINGS.md #4 measured the arm-to-arm gap
    at ~55% of the hand-path gap on accepted scenes, so real yields are lower);
  * deviation directions drawn orthogonal to the travel direction (kindest
    case; the plan samples joint-space directions with less clean EE effect);
  * grow_barrier, KEEPOUT, CAP_R, min_half, max_half, step ported verbatim
    from dfed/inverse.py.

Families probed:
  fam1  : the repo's existing 2-route family (dev 0.45-0.70) - calibration.
          Real whole-arm barrier-fit acceptance is ~12-13%; the EE-only proxy
          must come out HIGHER for the proxy to be certified optimistic.
  fam2  : the plan's 4-route family exactly as specced (dev 0.25-0.45 per
          stage, t1=1/3, t2=2/3, travel 0.30-0.75, 2 styles/mode, jitter).
  fam2big: fam2 with the repo-scale deviations 0.45-0.70 (what the geom sweep
          says barriers actually need).
  fam2long: fam2 with travel 0.60-1.00 (longer arc between the barriers).
"""
import numpy as np

CAP_R = 0.085
SOLVER_BUF = 0.02
MARGIN = 0.06
KEEPOUT = CAP_R + SOLVER_BUF + MARGIN          # 0.165, verbatim
N = 20


def _box_point_dist(lo, hi, P):
    d = np.maximum(np.maximum(lo - P, P - hi), 0.0)
    return float(np.min(np.linalg.norm(d, axis=1)))


def grow_barrier(seed, keep_clear, blocks, step=0.015,
                 max_half=(0.30, 0.30, 0.60), min_half=0.035):
    lo = np.asarray(seed, float) - 1e-3
    hi = np.asarray(seed, float) + 1e-3
    if _box_point_dist(lo, hi, keep_clear) < KEEPOUT:
        return None, "seed_too_close"
    dirs = [(0, -1), (0, 1), (1, -1), (1, 1), (2, -1), (2, 1)]
    live = [True] * 6
    while any(live):
        for t, (ax, sgn) in enumerate(dirs):
            if not live[t]:
                continue
            nlo, nhi = lo.copy(), hi.copy()
            if sgn < 0:
                nlo[ax] -= step
            else:
                nhi[ax] += step
            if (nhi[ax] - nlo[ax]) / 2 > max_half[ax] or \
               _box_point_dist(nlo, nhi, keep_clear) < KEEPOUT:
                live[t] = False
                continue
            lo, hi = nlo, nhi
    half = (hi - lo) / 2
    if np.any(half < min_half):
        return None, "too_small"
    box = np.concatenate([(lo + hi) / 2, half])
    if _box_point_dist(lo, hi, blocks) > CAP_R:
        return None, "missed_path"
    return box, "ok"


def polyline_points(knots, n_nodes=N + 1, sub=3):
    """Arc-length resample of an EE polyline + sub-sampling, like sweep_points."""
    K = np.asarray(knots, float)
    seg = np.linalg.norm(np.diff(K, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    s = s / max(s[-1], 1e-9)
    u = np.linspace(0.0, 1.0, n_nodes)
    P = np.stack([np.interp(u, s, K[:, j]) for j in range(3)], axis=1)
    out = [P]
    for f in np.linspace(0, 1, sub, endpoint=False)[1:]:
        out.append(P[:-1] * (1 - f) + P[1:] * f)
    return np.vstack(out)


def orth_units(rng, d):
    """Two random unit vectors orthogonal to travel direction d (kindest case)."""
    while True:
        a = rng.normal(size=3)
        a -= (a @ d) * d
        n = np.linalg.norm(a)
        if n > 1e-6:
            break
    a /= n
    while True:
        b = rng.normal(size=3)
        b -= (b @ d) * d
        n = np.linalg.norm(b)
        if n > 1e-6:
            break
    b /= n
    return a, b


def boxes_overlap(b1, b2):
    return bool(np.all(np.abs(b1[:3] - b2[:3]) < b1[3:6] + b2[3:6]))


def try_fam2(rng, dev_rng=(0.25, 0.45), travel=(0.30, 0.75), styles=True,
             seed_search=8):
    L = rng.uniform(*travel)
    d = rng.normal(size=3); d /= np.linalg.norm(d)
    p0 = np.zeros(3)
    pg = d * L
    u1, u2 = orth_units(rng, d)
    d1 = rng.uniform(*dev_rng)
    d2 = rng.uniform(*dev_rng)
    naive = polyline_points([p0, pg])
    a1 = p0 + d * (L / 3.0)
    a2 = p0 + d * (2.0 * L / 3.0)

    routes = []
    scales = ([0.85, 1.15] if styles else [1.0])
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            for sc in scales:
                jit = rng.normal(scale=0.015, size=(2, 3)) if styles else 0.0
                k1 = a1 + s1 * sc * d1 * u1 + (jit[0] if styles else 0)
                k2 = a2 + s2 * sc * d2 * u2 + (jit[1] if styles else 0)
                routes.append(polyline_points([p0, k1, k2, pg]))
    protected = np.vstack(routes)

    # barrier 1 near t1, barrier 2 near t2; several nearby seeds as the plan says
    node = np.linspace(0, 1, N + 1)
    k_t1 = int(np.argmin(np.abs(node - 1 / 3)))
    k_t2 = int(np.argmin(np.abs(node - 2 / 3)))
    naive_nodes = np.array([p0 + d * (L * t) for t in node])

    def grow_near(k_star):
        order = sorted(range(1, N), key=lambda k: abs(k - k_star))
        last = "no_seed"
        for k in order[:seed_search]:
            box, why = grow_barrier(naive_nodes[k], protected, naive)
            if box is not None:
                return box, "ok"
            last = why
        return None, last

    b1, why1 = grow_near(k_t1)
    if b1 is None:
        return "b1_" + why1
    b2, why2 = grow_near(k_t2)
    if b2 is None:
        return "b2_" + why2
    if boxes_overlap(b1, b2):
        return "overlap"
    return "ok"


def try_fam1(rng, dev_rng=(0.45, 0.70), travel=(0.35, 0.90)):
    L = rng.uniform(*travel)
    d = rng.normal(size=3); d /= np.linalg.norm(d)
    p0 = np.zeros(3); pg = d * L
    u, _ = orth_units(rng, d)
    dv = rng.uniform(*dev_rng)
    mid = p0 + d * (L / 2)
    A = polyline_points([p0, mid + dv * u, pg])
    B = polyline_points([p0, mid - dv * u, pg])
    protected = np.vstack([A, B])
    naive = polyline_points([p0, pg])
    node = np.linspace(0, 1, N + 1)
    naive_nodes = np.array([p0 + d * (L * t) for t in node])
    order = sorted(range(1, N), key=lambda k: abs(k - N // 2))
    for k in order[:8]:
        box, why = grow_barrier(naive_nodes[k], protected, naive)
        if box is not None:
            return "ok"
    return "b_" + why


def run(name, fn, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    from collections import Counter
    c = Counter(fn(rng) for _ in range(n))
    ok = c.get("ok", 0)
    print(f"{name:10s}  yield {ok/n*100:6.2f}%   " +
          "  ".join(f"{k}:{v/n*100:.1f}%" for k, v in sorted(c.items(), key=lambda kv: -kv[1]) if k != "ok"))
    return ok / n


print(f"EE-only proxy (optimistic: real builder protects whole-arm sweeps, "
      f"arm gap ~= 0.55 x hand gap per FINDINGS #4); KEEPOUT={KEEPOUT}")
run("fam1", try_fam1)
run("fam2", lambda rng: try_fam2(rng))
run("fam2-nostyle", lambda rng: try_fam2(rng, styles=False))
run("fam2big", lambda rng: try_fam2(rng, dev_rng=(0.45, 0.70)))
run("fam2long", lambda rng: try_fam2(rng, travel=(0.60, 1.00)))
run("fam2big+long", lambda rng: try_fam2(rng, dev_rng=(0.45, 0.70), travel=(0.60, 1.00)))
