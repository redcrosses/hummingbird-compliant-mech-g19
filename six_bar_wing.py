r"""
six_bar_wing.py
===============

Six-bar flapping-wing synthesis in two steps.

    Step 1  Four-bar O2-A-B-O4 with coupler point P, optimised for the PATH of P
            only (four_bar_path_synthesis.py, path-only mode).
    Step 2  Wing dyad added to it:
                wing   = link P-C, pivots on the coupler at P (P is always on the wing)
                link 6 = link C-D, D the point that drives the wing. D is either
                         "ground": a fixed pivot O6 on the body
                         "crank" : a point on the crank (rotates with the crank)
                         "rocker": a point on the rocker O4-B (swings with it)
            The dyad is optimised so that the wing angle (direction P->C, or C->P)
            follows the target angle over the flap cycle. The dyad hangs on P and
            does not change P's path, so step 1 is unaffected.

                      D  (fixed O6, or a point on the crank / rocker)
                       \   link 6 (|DC|)
                        \
                         C
                         |  wing (|PC|)
                         P  ---- coupler of the four-bar

    The target angle follows the MOTION, not the crank angle: 0-50 % of the
    target function is the downstroke, 50-100 % the upstroke (stroke_phase).

    Why D may need to move: P passes the crossing of the figure-eight twice, once
    on the downstroke and once on the upstroke. With D fixed on the body the wing
    angle depends only on where P is, so it is the same both times. With D on the
    crank or rocker it also depends on how far the cycle has progressed.

Needs four_bar_path_synthesis.py in the same folder.
Run:  python six_bar_wing.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

import four_bar_path_synthesis as fb


# =============================================================================
# Settings
# =============================================================================
S = 120                                          # precision points per cycle
TH2 = np.linspace(0, 2 * np.pi, S, endpoint=False)
t = TH2
DESIRED = np.c_[50 * np.sin(t) + 0,              # desired path of P (mm)
                12.64 * np.sin(2 * t) + 15]

BODY_AREA = 1000.0                               # mm^2: area for the fixed pivots O2, O4
BODY_MIN_W = 10.0                                # mm
BODY_MIN_H = 10.0                                # mm
BODY_CENTER_X = None                             # None = body anywhere, 0.0 = centred on x = 0
W_AREA = 10.0                                    # step 1: signed-area term (figure-eight vs oval)
W_POS = 1.0                                      # step 1: point-by-point position term (shape)


AOA = 45                       # angle of attack on both strokes (deg)


def target_angle(x):
    """Desired wing angle in degrees from the +y-axis (positive towards +x, clockwise),
    x = % of the flap cycle: 0-50 % downstroke, 50-100 % upstroke (see stroke_phase).
    0 at both stroke reversals, -(90 - AOA) mid-downstroke, +(90 - AOA) mid-upstroke."""
    return -np.sin(2 * np.pi * x / 100) * (90 - AOA)


DOWNSTROKE_DIR = 1             # -1: on the downstroke P moves towards -x, +1: towards +x
                               # (+1 with the crank turned backwards in six_bar_analysis.py
                               #  gives a downstroke towards -x)

LINK6_ON = ["ground", "crank", "rocker"]   # where link 6 may be attached; best is kept
O6_IN_BODY = True              # "ground": the fixed pivot O6 must lie inside the body
MIN_TRANSMISSION = 20.0        # deg: minimum angle between wing and link 6 (avoid locking)
N_STARTS = 300                 # random starting guesses per option
N_DENSE = 720                  # crank positions used to check the whole motion


def stroke_phase(desired_xy, down_dir=None):
    """
    % of the flap cycle at every point of the desired path, following the motion:
      downstroke = P moves in the down_dir x-direction: 0 % (stroke reversal where it
                   starts) ... 50 % (reversal at the other end)
      upstroke   = P moves back:                       50 % ... 100 %
    Each stroke is spread evenly over its own points, so the target function's
    first half always lands on the downstroke and its second half on the upstroke.
    """
    down_dir = DOWNSTROKE_DIR if down_dir is None else down_dir
    x = np.asarray(desired_xy, float)[:, 0]
    s = len(x)
    down = (np.roll(x, -1) - x) * down_dir > 0            # step j -> j+1 is a downstroke step
    starts_down = np.where(down & ~np.roll(down, 1))[0]
    starts_up = np.where(~down & np.roll(down, 1))[0]
    if len(starts_down) != 1 or len(starts_up) != 1:
        raise ValueError("The path must have exactly one downstroke and one upstroke "
                         "(one back-and-forth motion along x per cycle).")
    j0, j1 = starts_down[0], starts_up[0]
    n_down = (j1 - j0) % s
    n_up = s - n_down
    phase = np.empty(s)
    for k in range(s):
        j = (j0 + k) % s
        phase[j] = 50.0 * k / n_down if k < n_down else 50.0 + 50.0 * (k - n_down) / n_up
    return phase


# =============================================================================
# Step 1: four-bar for the path of P
# =============================================================================
def step1_fourbar():
    size = np.max(np.ptp(DESIRED, axis=0))
    prob, x0 = fb.atlas_search(DESIRED, TH2, closed=True, orientation_free=True,
                               z_min=0.025 * size, z_max=3.0 * size,
                               n_samples=40000, n_refine=200,
                               box_joints="ground",
                               body_area=BODY_AREA, body_min_w=BODY_MIN_W,
                               body_min_h=BODY_MIN_H, body_center_x=BODY_CENTER_X,
                               w_area=W_AREA, w_pos=W_POS,
                               angle_link=None, verbose=False)
    return prob, x0


def fourbar_motion(prob, th2):
    """Positions of O2, O4, A, B, P (path coordinates) and the placed crank angle
    at the (local) input angles th2."""
    O2, psi, _ = prob.placement()
    R = np.exp(1j * psi)
    pos = fb.position_analysis(prob.x, th2, prob.branch, prob.cp_side)
    return dict(O2=O2, O4=O2 + R * pos["O4"], A=O2 + R * pos["A"],
                B=O2 + R * pos["B"], P=O2 + R * pos["P"],
                crank=np.asarray(th2, float) + psi)


# =============================================================================
# Step 2: wing dyad P-C-D for the wing angle
# =============================================================================
def mount_point(v, m, where):
    """
    Point D that link 6 is attached to, for design v = [v0, v1, |PC|, |DC|]:
      "ground": D = O6 = (v0, v1), fixed
      "crank" : D on the crank,  distance v0 from O2, at angle v1 from the crank O2->A
      "rocker": D on the rocker, distance v0 from O4, at angle v1 from the rocker O4->B
    """
    if where == "ground":
        return (v[0] + 1j * v[1]) * np.ones_like(m["P"])
    if where == "crank":
        return m["O2"] + v[0] * np.exp(1j * (m["crank"] + v[1]))
    return m["O4"] + v[0] * np.exp(1j * (np.angle(m["B"] - m["O4"]) + v[1]))


def dyad(v, m, where, branch):
    """
    Joint C: intersection of the circle around P (radius |PC|) and around D
    (radius |DC|). Returns C and D, or None if the dyad cannot be assembled
    at every position.
    """
    D = mount_point(v, m, where)
    P = m["P"]
    L_pc, L_dc = v[2], v[3]
    dv = D - P
    d = np.abs(dv)
    if np.any(d < 1e-12):
        return None
    a = (L_pc**2 - L_dc**2 + d**2) / (2 * d)
    h2 = L_pc**2 - a**2
    if np.any(h2 < 0):
        return None
    u = dv / d
    C = P + a * u + branch * 1j * np.sqrt(h2) * u
    return C, D


def wing_angle(C, P, flip):
    """Wing chord direction from the +x-axis: P->C (flip = 0) or C->P (flip = pi)."""
    return np.angle(C - P) + flip


def transmission(C, P, D):
    """Angle between wing and link 6 at C, folded to 0...90 deg."""
    mu = np.rad2deg(np.abs(fb.wrap(np.angle(C - D) - np.angle(C - P))))
    return np.minimum(mu, 180 - mu)


def step2_wing(prob):
    mot_j = fourbar_motion(prob, prob.th2)                       # at the precision points
    th_dense = np.linspace(0, 2 * np.pi, N_DENSE, endpoint=False)
    mot_d = fourbar_motion(prob, th_dense)                       # whole motion

    phase = stroke_phase(DESIRED)                                # % of the flap cycle
    target_x = np.deg2rad(90.0 - target_angle(phase))           # from +x-axis

    xmin, xmax, ymin, ymax = prob.body_rectangle(prob.x)
    size = np.max(np.ptp(DESIRED, axis=0))
    rng = np.random.default_rng(0)
    best = None
    for where in LINK6_ON:
        if where == "ground":
            if O6_IN_BODY:
                lo, hi = [xmin, ymin], [xmax, ymax]
            else:
                lo, hi = [-5 * size, -5 * size], [5 * size, 5 * size]
        else:
            lo, hi = [0.05 * size, -np.pi], [2.0 * size, np.pi]
        lo = lo + [0.05 * size, 0.05 * size]
        hi = hi + [2.0 * size, 3.0 * size]

        for flip in (0.0, np.pi):                                # chord P->C or C->P
            for branch in (+1, -1):                              # C on either side of P-D
                def residual(v):
                    res = dyad(v, mot_j, where, branch)
                    if res is None:
                        return np.full(len(mot_j["P"]), np.pi)   # cannot assemble
                    return fb.wrap(target_x - wing_angle(res[0], mot_j["P"], flip))

                for _ in range(N_STARTS):
                    v0 = rng.uniform(lo, hi)
                    if dyad(v0, mot_d, where, branch) is None:
                        continue
                    sol = least_squares(residual, v0, bounds=(lo, hi))
                    res = dyad(sol.x, mot_d, where, branch)      # check the whole motion
                    if res is None:
                        continue
                    if transmission(res[0], mot_d["P"], res[1]).min() < MIN_TRANSMISSION:
                        continue
                    rms = np.rad2deg(np.sqrt(np.mean(residual(sol.x) ** 2)))
                    if best is None or rms < best["rms"]:
                        best = dict(rms=rms, v=sol.x, where=where, branch=branch, flip=flip)
        if best is not None:
            print(f"  best so far: link 6 on {best['where']}, wing angle RMS {best['rms']:.1f} deg")
    if best is None:
        raise ValueError("No wing dyad found: add options to LINK6_ON, allow O6 outside "
                         "the body (O6_IN_BODY = False) or lower MIN_TRANSMISSION.")
    best.update(mot_j=mot_j, mot_d=mot_d, th_dense=th_dense, target_x=target_x, phase=phase)
    return best


# =============================================================================
# Report and plots
# =============================================================================
def report(prob, w):
    mot_d = w["mot_d"]
    C, D = dyad(w["v"], mot_d, w["where"], w["branch"])
    P = mot_d["P"]
    ang = wing_angle(C, P, w["flip"])
    O2, psi, Pg = prob.placement()
    print("\nStep 1: four-bar (path of P)")
    for k in range(6):
        print(f"  {fb.VAR_NAMES[k]:>3s} = {prob.x[k]:8.3f} mm")
    print(f"  theta1 = {float(prob.x[6] + psi):.4f} rad (placed),  O2 = "
          f"({O2.real:.3f}, {O2.imag:.3f}),  psi = {float(psi):.4f} rad")
    print(f"  path RMS error {np.sqrt(np.mean(np.abs(Pg - prob.Pd) ** 2)):.2f} mm")

    print("\nStep 2: wing dyad")
    v = w["v"]
    if w["where"] == "ground":
        print(f"  link 6 on the body: O6 = ({v[0]:.3f}, {v[1]:.3f})")
    else:
        print(f"  link 6 on the {w['where']}: D at {v[0]:.3f} mm from "
              f"{'O2' if w['where'] == 'crank' else 'O4'}, {np.rad2deg(v[1]):.2f} deg from the "
              f"{'crank O2->A' if w['where'] == 'crank' else 'rocker O4->B'}")
    print(f"  |PC| = {v[2]:.3f} mm   wing link")
    print(f"  |DC| = {v[3]:.3f} mm   link 6")
    print(f"  wing chord along {'P->C' if w['flip'] == 0 else 'C->P'}, assembly branch {w['branch']}")
    print(f"  wing angle RMS error {w['rms']:.1f} deg")
    print(f"  min transmission angle at C {transmission(C, P, D).min():.0f} deg")
    j0 = int(np.argmin(w["phase"]))
    print(f"  downstroke starts at precision point {j0}: crank angle "
          f"{float(prob.th2[j0] + psi):.4f} rad (placed frame)")

    # joint rotations over one cycle (change of the angle between the connected links)
    if w["where"] == "ground":
        mount_dir = np.zeros_like(P, dtype=float)
    elif w["where"] == "crank":
        mount_dir = mot_d["crank"]
    else:
        mount_dir = np.angle(mot_d["B"] - mot_d["O4"])
    rot = {
        "P (coupler - wing)": np.unwrap(np.angle(C - P) - np.angle(mot_d["B"] - P)),
        "C (wing - link 6)": np.unwrap(np.angle(C - D) - np.angle(C - P)),
        f"D (link 6 - {w['where']})": np.unwrap(np.angle(C - D) - mount_dir),
    }
    print("  joint rotation over one cycle:")
    for k, r in rot.items():
        print(f"    {k}: {np.rad2deg(np.ptp(r)):.0f} deg")
    return C, D, ang


def plot(prob, w, C, D, ang):
    mot_d, th = w["mot_d"], w["th_dense"]
    j = 0
    m = {k: (v if np.ndim(v) == 0 else v[j]) for k, v in mot_d.items()}

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    a = ax[0]
    a.plot(prob.Pd.real, prob.Pd.imag, "k-", lw=2, label="Desired path")
    a.plot(mot_d["P"].real, mot_d["P"].imag, ":", color="tab:blue", lw=2, label="Path of P")
    xmin, xmax, ymin, ymax = prob.body_rectangle(prob.x)
    a.fill([xmin, xmax, xmax, xmin], [ymin, ymin, ymax, ymax], color="tab:green", alpha=0.25,
           label="Body")
    for p, q in [("O2", "A"), ("A", "B"), ("B", "O4"), ("A", "P"), ("P", "B")]:
        a.plot([m[p].real, m[q].real], [m[p].imag, m[q].imag], "-", color="tab:blue")
    if w["where"] == "crank":
        a.plot([m["O2"].real, D[j].real], [m["O2"].imag, D[j].imag], "-", color="tab:blue")
    if w["where"] == "rocker":
        a.plot([m["O4"].real, D[j].real], [m["O4"].imag, D[j].imag], "-", color="tab:blue")
    a.plot([m["P"].real, C[j].real], [m["P"].imag, C[j].imag], "-", color="tab:red", lw=3,
           label="Wing (P-C)")
    a.plot([C[j].real, D[j].real], [C[j].imag, D[j].imag], "-", color="tab:orange", lw=1.5,
           label="Link 6 (C-D)")
    fixed = [("O2", m["O2"]), ("O4", m["O4"])] + ([("O6", D[j])] if w["where"] == "ground" else [])
    for k, z in fixed:
        a.plot(z.real, z.imag, "o", ms=9, mfc="none", mec="k", mew=2)
        a.annotate(k, (z.real, z.imag), textcoords="offset points", xytext=(6, 6), fontsize=8)
    moving = [("A", m["A"]), ("B", m["B"]), ("P", m["P"]), ("C", C[j])]
    if w["where"] != "ground":
        moving.append(("D", D[j]))
    for k, z in moving:
        a.plot(z.real, z.imag, "o", ms=5, mfc="w", mec="k")
        a.annotate(k, (z.real, z.imag), textcoords="offset points", xytext=(6, -10), fontsize=8)
    a.set_aspect("equal"); a.grid(alpha=0.3); a.legend(loc="best", fontsize=8)
    a.set_title(r"Six-bar at $\theta_2 = 0$")

    b = ax[1]
    # % of the flap cycle along the dense motion (interpolated between the points)
    ph_j = w["phase"]
    th_j = np.r_[prob.th2, prob.th2[0] + 2 * np.pi]
    ph_unw = np.unwrap(np.r_[ph_j, ph_j[0]] * 2 * np.pi / 100) * 100 / (2 * np.pi)
    ph_d = np.mod(np.interp(th, th_j, ph_unw), 100)
    order = np.argsort(ph_d)
    b.axvspan(0, 50, color="tab:blue", alpha=0.08, label="Downstroke")
    b.plot(ph_d[order], fb.to_reference(ang, "y")[order], label="Wing angle")
    b.plot(ph_j, fb.to_reference(w["target_x"], "y"), "o", ms=3,
           color="tab:red", label="Target")
    b.set_xlim(0, 100); b.set_xticks(np.arange(0, 101, 10))
    b.set_xlabel("% of flap cycle (0-50 downstroke, 50-100 upstroke)")
    b.set_ylabel("Wing angle from y-axis (deg)")
    b.set_title("Wing angle"); b.grid(); b.legend()
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    print("Step 1: optimising the four-bar for the path of P ...")
    prob, x0 = step1_fourbar()
    print("Step 2: optimising the wing dyad for the wing angle ...")
    w = step2_wing(prob)
    C, D, ang = report(prob, w)
    plot(prob, w, C, D, ang)
    plt.show()