r"""
Path-generation synthesis of a planar four-bar linkage with the exact-gradient
(sensitivity) method of

    R. Sancibrian et. al,
    "Gradient-based optimization of path synthesis problems in planar mechanisms",
    Mechanism and Machine Theory 39 (2004) 839-856.

Every block below is tagged with the section / equation of the paper it implements.

---------------------------------------------------------------------------------
Mechanism of Fig. 1 and conventions
---------------------------------------------------------------------------------
        P  (coupler point)
       / \
   Z5 /   \ Z6           Z3 : coupler side A->B          angle theta3
     /     \             Z5 : A -> P                      angle theta5 = alpha1 + theta3
    A-------B            Z6 : P -> B                      angle theta6
    |  Z3    \           Z2 : input crank O2 -> A         angle theta2 (prescribed)
 Z2 |         \ Z4       Z4 : B -> O4                     angle theta4
    |          \         Z1 : ground; O4 = O2 + Z1*exp(i*theta1)
    O2 ======== O4

Loop closure [Eq. 6]  sum_{p=1..4} Z_p e^{i th_p} = 0  is written head-to-tail
O2 -> A -> B -> O4 -> O2, so the ground term is Z1 e^{i(theta1+pi)}.
Coupler-triangle loop [Eq. 11]: Z5 e^{i th5} + Z6 e^{i th6} - Z3 e^{i th3} = 0
(same triangle as the paper; the sign of the Z3 term follows from the
directions chosen above).

Precision points: the desired path is a list of s points, and precision point j
is reached at input angle theta2^j (the "controlled input" of Sec. 2/Eq. 10).
By default theta2^j is uniformly spaced; you can pass your own timing.

Two solvers are provided, both driven by the paper's exact sensitivities A, B:
  method="paper" : the per-precision-point, one-link-at-a-time update of Sec. 4
                   (Eqs. 21-24), exactly as described.
  method="lm"    : [ADDED] Levenberg-Marquardt on the error of Eq. 18, with the
                   Jacobian built from A_k^j (Eq. 13) and d(lambda)/dZ (Eqs. 13+19).
                   Same exact gradients, but a true descent on E, so it converges
                   much more reliably. Constraints of Sec. 5 are enforced the same way.

Run:  python four_bar_path_synthesis.py
"""

import numpy as np
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt

VAR_NAMES = ["Z1", "Z2", "Z3", "Z4", "Z5", "Z6", "theta1"]
IZ1, IZ2, IZ3, IZ4, IZ5, IZ6, ITH1 = range(7)
NVAR = 7


def wrap(a):
    """Wrap angle(s) to (-pi, pi]."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


# =============================================================================
# 1. Position analysis of the four-bar  (solves the loop closure [Eq. 6] and the
#    coupler triangle [Eq. 11] for every precision point j)
# =============================================================================
def position_analysis(x, th2, branch=+1, cp_side=+1):
    """
    Returns a dict with all link angles theta_p^j and coupler point P^j for the
    input angles th2 (array), or None if the linkage cannot be assembled.
    O2 is at the origin; the final placement is done afterwards (Sec. 3).
    """
    Z1, Z2, Z3, Z4, Z5, Z6, th1 = x
    if min(Z1, Z2, Z3, Z4, Z5, Z6) <= 0:
        return None

    # coupler triangle angle alpha1 (angle between Z3 and Z5), law of cosines
    c = (Z3**2 + Z5**2 - Z6**2) / (2 * Z3 * Z5)
    if abs(c) > 1:
        return None
    alpha1 = cp_side * np.arccos(c)

    O4 = Z1 * np.exp(1j * th1)
    A = Z2 * np.exp(1j * th2)
    dvec = O4 - A
    d = np.abs(dvec)
    if np.any(d < 1e-12):
        return None
    # B = intersection of circle(A, Z3) and circle(O4, Z4)
    a = (Z3**2 - Z4**2 + d**2) / (2 * d)
    h2 = Z3**2 - a**2
    if np.any(h2 < 0):
        return None                                  # cannot assemble
    u = dvec / d
    B = A + a * u + branch * 1j * np.sqrt(h2) * u    # fixed branch -> continuity

    th3 = np.angle(B - A)                            # Z3: A -> B
    th4 = np.angle(O4 - B)                           # Z4: B -> O4
    th5 = th3 + alpha1                               # theta5 = alpha1 + theta3 (Sec. 2)
    P = A + Z5 * np.exp(1j * th5)                    # coupler point
    th6 = np.angle(B - P)                            # Z6: P -> B  [Eq. 11 triangle]
    th1_loop = th1 + np.pi                           # ground term in [Eq. 6]
    return dict(th1_loop=th1_loop, th2=np.asarray(th2, float), th3=th3, th4=th4,
                th5=th5, th6=th6, P=P, A=A, B=B, O4=O4, alpha1=alpha1)


# =============================================================================
# 2. Link-dimension sensitivity analysis  [Sec. 2, Eqs. 2-13]
# =============================================================================
def _solve_angle_pair(Za, tha, Zb, thb, c):
    """
    Solves  i (Za*Ea*e^{i tha} + Zb*Eb*e^{i thb}) = -c   for the real unknowns Ea, Eb.
    Splitting into real / imaginary parts gives the 2x2 linear system of [Eq. 9]:
        Ea Za sin(tha) + Eb Zb sin(thb) =  Re(c)        (for c = e^{i th_k}: cos th_k)
        Ea Za cos(tha) + Eb Zb cos(thb) = -Im(c)        (for c = e^{i th_k}: -sin th_k)
    (the minus sign on the right of [Eq. 8] was lost in the PDF text extraction).
    Also returns |sin(thb - tha)|, a scale-free measure of the conditioning
    (end of Sec. 2: ill-conditioned systems must be detected and discarded).
    """
    M11 = Za * np.sin(tha); M12 = Zb * np.sin(thb)
    M21 = Za * np.cos(tha); M22 = Zb * np.cos(thb)
    r1 = np.real(c); r2 = -np.imag(c)
    det = M11 * M22 - M12 * M21
    Ea = (r1 * M22 - M12 * r2) / det
    Eb = (M11 * r2 - r1 * M21) / det
    cond = np.abs(det) / (Za * Zb)                   # = |sin(thb - tha)|
    return Ea, Eb, cond


def sensitivities(x, pos):
    """
    Angular sensitivities E_pk = d theta_p / d x_k  at every precision point and the
    derivative of the coupler point dP/dx_k.

    * Four-bar loop [Eqs. 7-10]: ground (link 1) and input (link 2) keep their
      orientation (E_1k = E_hk = 0, "two of these unknowns will be zero"), so the
      unknowns are q = 3, r = 4.  Eq. (9) (position j) and Eq. (10) (position j+1)
      are the same system evaluated at different input angles, so here it is solved
      once, vectorised, for all positions.
    * Coupler loop [Eqs. 11-12]: gives E_5k and E_6k (k = 3, 5, 6 change the
      triangle shape; for the other k the triangle just rotates with Z3).
    """
    Z1, Z2, Z3, Z4, Z5, Z6, th1 = x
    th = pos
    s = len(th["th2"])
    e = {p: np.exp(1j * th[p]) for p in ("th2", "th3", "th4", "th5", "th6")}
    e1 = np.exp(1j * th["th1_loop"]) * np.ones(s)

    E3 = np.zeros((NVAR, s)); E4 = np.zeros((NVAR, s))
    E5 = np.zeros((NVAR, s)); E6 = np.zeros((NVAR, s))
    cond_min = np.full(s, np.inf)
    for k in range(NVAR):
        # explicit derivative of the loop  sum Z_p e^{i th_p}  w.r.t. x_k   [Eq. 7]
        # (D_pk = Kronecker delta)
        if k == IZ1:
            c = e1
        elif k == IZ2:
            c = e["th2"]
        elif k == IZ3:
            c = e["th3"]
        elif k == IZ4:
            c = e["th4"]
        elif k == ITH1:
            c = 1j * Z1 * e1                         # derivative w.r.t. ground angle
        else:                                        # Z5, Z6 are not in loop [Eq. 6]
            c = np.zeros(s, complex)
        E3[k], E4[k], cnd = _solve_angle_pair(Z3, th["th3"], Z4, th["th4"], c)   # [Eq. 9/10]
        cond_min = np.minimum(cond_min, cnd)

        # coupler triangle  Z5 e^{i th5} + Z6 e^{i th6} - Z3 e^{i th3} = 0      [Eq. 12]
        ct = -1j * Z3 * E3[k] * e["th3"]
        if k == IZ3:
            ct = ct - e["th3"]
        if k == IZ5:
            ct = ct + e["th5"]
        if k == IZ6:
            ct = ct + e["th6"]
        E5[k], E6[k], cnd = _solve_angle_pair(Z5, th["th5"], Z6, th["th6"], ct)
        cond_min = np.minimum(cond_min, cnd)

    # derivative of the coupler point P = Z2 e^{i th2} + Z5 e^{i th5}   (the dyad of [Eq. 2])
    # dP/dx_k = D_2k e^{i th2} + D_5k e^{i th5} + i Z5 E_5k e^{i th5}   (E_2k = 0, input controlled)
    dP = 1j * Z5 * E5 * e["th5"][None, :]
    dP[IZ2] += e["th2"]
    dP[IZ5] += e["th5"]
    return dict(E3=E3, E4=E4, E5=E5, E6=E6, dP=dP, cond=cond_min)


def signed_area(P):
    """Signed area of a closed curve given as complex points (+ = counter-clockwise)."""
    return 0.5 * np.sum(np.imag(np.conj(P) * np.roll(P, -1)))


def pair_indices(s, closed):
    """Consecutive precision-point couples (j, j+1)."""
    j = np.arange(s if closed else s - 1)
    return j, (j + 1) % s


def displacement_vectors(P, closed):
    """Module n^j and argument phi^j of the vector from point j to j+1  [Eqs. 16-17]."""
    j, jn = pair_indices(len(P), closed)
    d = P[jn] - P[j]                                 # left-hand side of [Eq. 2]
    return np.abs(d), np.angle(d)


def module_argument_sensitivities(dP, n_g, phi_g, closed):
    """
    A_k^j = d n_g^j / d Z_k   and   B_k^j = d phi_g^j / d Z_k        [Eqs. 3-5, 13]
    The right-hand side of [Eq. 4] is exactly dP^{j+1} - dP^j; H and G are its real
    and imaginary parts [Eq. 5]. Solving [Eq. 5]:
        A = H cos(phi) + G sin(phi)
        B = (G cos(phi) - H sin(phi)) / n
    (the printed B in Eq. 13 lost its signs in the PDF; this is the solution of Eq. 5).
    """
    s = dP.shape[1]
    j, jn = pair_indices(s, closed)
    rhs = dP[:, jn] - dP[:, j]                       # right-hand side of [Eq. 4]
    H, G = rhs.real, rhs.imag
    cphi, sphi = np.cos(phi_g)[None, :], np.sin(phi_g)[None, :]
    A = H * cphi + G * sphi                          # [Eq. 13]
    B = (G * cphi - H * sphi) / n_g[None, :]         # [Eq. 13]
    return A, B


# =============================================================================
# 3. Synthesis error  [Sec. 3, Eqs. 16-19]
# =============================================================================
def relative_angles(phi, closed):
    """lambda^j = phi^{j+1} - phi^j   [Eq. 19] (removes the Orientation Effect)."""
    if closed:
        return wrap(np.roll(phi, -1) - phi)
    return wrap(phi[1:] - phi[:-1])


def synthesis_error(n_d, n_g, ang_d, ang_g, w_n, w_phi):
    """
    E = w_n (n_d-n_g)^T(n_d-n_g) + w_phi (ang_d-ang_g)^T(ang_d-ang_g)    [Eq. 18]
    The paper calls the weights lambda_n, lambda_phi; renamed here so they are not
    confused with the relative angles lambda of [Eq. 19].
    ang = phi (absolute orientation kept) or lambda (orientation-free, [Eq. 19]).
    Only relative coordinates are used, so the Translation Effect is removed.
    """
    dn = n_d - n_g
    da = wrap(ang_d - ang_g)
    return w_n * dn @ dn + w_phi * da @ da


# =============================================================================
# 4. The synthesis problem and the optimisation algorithm  [Secs. 4-5]
# =============================================================================
class FourBarPathSynthesis:
    def __init__(self, desired_xy, x0, th2=None, closed=True, branch=+1, cp_side=+1,
                 orientation_free=True, design_vars=None, z_min=None, z_max=None,
                 grashof_crank=True, w_n=1.0, w_phi=None, relax=0.3, min_cond=0.05,
                 design_box=None, box_joints="all", n_check=180, grashof_margin=0.05,
                 angle_link=None, angle_targets=None, w_link=None, angle_ref="x",
                 body_area=None, body_min_w=0.0, body_min_h=0.0, body_center_x=None,
                 w_area=None, w_pos=None):
        """
        desired_xy        (s,2) desired precision points (any units, e.g. mm)
        x0                initial guess [Z1..Z6, theta1] (Sec. 4 "definition of the
                          initial linkage"; the paper picks it from a coupler-curve atlas)
        th2               input angle at each precision point (controlled input, Eq. 10);
                          default: uniform over 2*pi (closed) - pass your own for open paths
        orientation_free  True -> use lambda [Eq. 19]; False -> absolute phi [Eq. 18]
        design_vars       indices of x that may change (constraints Sec. 5)
        z_min, z_max      link length bounds   Zmax >= Z_k + dZ_k >= Zmin  [Sec. 5]
        grashof_crank     keep the input link a full-rotation crank [Sec. 5 / Example 1]
        grashof_margin    [ADDED] require (p+q) - (s+l) >= grashof_margin * (p+q).
                          With a margin near 0 the linkage is close to a change point
                          (e.g. a kite with Z2~Z3, Z1~Z4): its two assembly modes meet
                          and the real mechanism can switch to the other mode (for a
                          kite: B sits on O2 and P just draws a circle).
        w_n, w_phi        weights of [Eq. 18] (w_phi default balances mm vs rad)
        relax             [ADDED] under-relaxation factor on each link increment
        design_box        design-space constraint [Sec. 5, "overall dimensions of the
                          linkage" / kinematic pairs in a restricted area, Sec. 3]:
                          (xmin, xmax, ymin, ymax) in the coordinates of desired_xy.
                          None = no constraint.
        box_joints        "all"    -> O2, O4, A, B and P must stay inside the box
                          "ground" -> only the fixed pivots O2, O4 must be inside
        n_check           crank positions checked over the motion (between the
                          precision points too)
        angle_link        link (and direction) whose angle is prescribed. None = off.
                          "A->P" (= Z5, theta5), "P->A", "P->B" (= Z6, theta6), "B->P"
        angle_targets     (indices, angles): precision points j and the desired angle of
                          that link there, in rad from the +x-axis, counter-clockwise
                          (made by stroke_angle_targets or cycle_angle_targets)
        angle_ref         "x" or "y": axis the angle is shown from in plots/report
                          ("y" = from the +y-axis, positive towards +x / clockwise)
        w_link            weight of the link-angle error, default (path size / 10)^2
        body_area         body-area constraint (instead of a fixed design box): the
                          body is the smallest rectangle around O2 and O4, at least
                          body_min_w wide and body_min_h tall; its area must be
                          <= body_area. Position and shape of the body are free.
                          None = off.
        body_min_w/h      minimum body width / height (so the body cannot become a line)
        w_pos             weight of the position error (None = off): the distance between
                          generated and desired path point by point, after the best
                          shift (and rotation, if orientation is free). This compares the
                          shape itself, so ovals and thin loops are penalised directly.
        w_area            weight of the signed-area error (None = off). The signed area
                          of a closed curve is + for a counter-clockwise loop and - for a
                          clockwise one, so a figure-eight (two opposite loops) has about
                          zero area and an oval a large one. The error term
                          (w_area / size^2) * (A_desired - A_generated)^2 therefore
                          pushes the result towards a figure-eight instead of an oval.
        body_center_x     None: body may sit anywhere. A number: the body is centred
                          on this x (e.g. 0 = centred under the stroke), so its width
                          is 2 * the largest distance of a pivot from that x.
        min_cond          [ADDED] reject designs whose sensitivity systems become
                          ill-conditioned (|sin(transmission angle)| < min_cond), cf. end of Sec. 2
        """
        P = np.asarray(desired_xy, float)
        self.Pd = P[:, 0] + 1j * P[:, 1]
        self.s = len(self.Pd)
        self.closed = closed
        if th2 is None:
            th2 = np.linspace(0, 2 * np.pi, self.s, endpoint=not closed)
        self.th2 = np.asarray(th2, float)
        self.branch, self.cp_side = branch, cp_side
        self.orientation_free = orientation_free
        self.design_vars = list(range(NVAR)) if design_vars is None else list(design_vars)
        self.z_min = 1e-3 if z_min is None else z_min
        self.z_max = np.inf if z_max is None else z_max
        self.grashof_crank = grashof_crank
        self.grashof_margin = grashof_margin
        self.relax = relax
        self.min_cond = min_cond
        self.design_box = design_box
        self.box_joints = box_joints
        self.body_area = body_area
        self.body_min_w, self.body_min_h = body_min_w, body_min_h
        self.body_center_x = body_center_x
        size_p = np.max(np.ptp(P, axis=0))
        self.w_area = None if (w_area is None or not closed) else w_area / size_p ** 2
        self.area_d = signed_area(self.Pd) if closed else 0.0
        self.w_pos = w_pos
        if closed:
            self.th2_check = np.linspace(0, 2 * np.pi, n_check, endpoint=False)
        else:
            self.th2_check = np.linspace(self.th2.min(), self.th2.max(), n_check)
        self.x = np.asarray(x0, float).copy()

        # prescribed link angle (flapping motion)
        self.angle_link = {"Z5": "A->P", "Z6": "P->B"}.get(angle_link, angle_link)
        self.angle_ref = angle_ref
        if angle_link is not None:
            if self.angle_link not in ("A->P", "P->A", "P->B", "B->P"):
                raise ValueError('angle_link must be "A->P", "P->A", "P->B" or "B->P"')
            idx, ang = angle_targets
            self.angle_idx = np.asarray(idx, int)
            self.angle_target = np.asarray(ang, float)
            # an angle relative to the x-axis fixes the orientation of the mechanism,
            # so the rotation-free error (Eq. 19) is switched off
            self.orientation_free = orientation_free = False
            size = np.max(np.ptp(P, axis=0))
            self.w_link = (size / 10.0) ** 2 if w_link is None else w_link

        # desired vectors  [Eq. 16]
        self.n_d, self.phi_d = displacement_vectors(self.Pd, closed)
        self.lam_d = relative_angles(self.phi_d, closed)
        self.w_n = w_n
        self.w_phi = (np.mean(self.n_d) ** 2) if w_phi is None else w_phi
        self.history = []

    # ---------- analysis helpers ----------
    def analyse(self, x):
        return position_analysis(x, self.th2, self.branch, self.cp_side)

    def position_residual(self, Pg):
        """Desired minus generated point positions after the best shift (and rotation,
        if orientation is free) of the generated path. Complex, one per point."""
        Pd0 = self.Pd - self.Pd.mean()
        Pg0 = Pg - Pg.mean()
        R = np.exp(1j * np.angle(np.sum(np.conj(Pg0) * Pd0))) if self.orientation_free else 1.0
        return Pd0 - R * Pg0, R

    def link_angle(self, pos):
        """Angle of the prescribed link from the +x-axis at every point:
        A->P = theta5, P->B = theta6; the reversed directions add 180 deg."""
        th = pos["th5"] if self.angle_link in ("A->P", "P->A") else pos["th6"]
        return th + np.pi if self.angle_link in ("P->A", "B->P") else th

    def link_angle_error(self, pos):
        """w_link * sum_j (theta_target^j - theta_link^j)^2 over the target points."""
        if self.angle_link is None:
            return 0.0
        d = wrap(self.angle_target - self.link_angle(pos)[self.angle_idx])
        return self.w_link * d @ d

    def error(self, x):
        pos = self.analyse(x)
        if pos is None:
            return np.inf
        n_g, phi_g = displacement_vectors(pos["P"], self.closed)          # [Eq. 17]
        if self.orientation_free:
            E = synthesis_error(self.n_d, n_g, self.lam_d,
                                relative_angles(phi_g, self.closed), self.w_n, self.w_phi)
        else:
            E = synthesis_error(self.n_d, n_g, self.phi_d, phi_g, self.w_n, self.w_phi)
        if self.w_area is not None:                              # signed-area term
            E += self.w_area * (self.area_d - signed_area(pos["P"])) ** 2
        if self.w_pos is not None:                               # position term
            rp, _ = self.position_residual(pos["P"])
            E += self.w_pos * np.sum(np.abs(rp) ** 2)
        return E + self.link_angle_error(pos)                   # Eq. 18 + link angles

    # ---------- constraints  [Sec. 5] ----------
    def feasible(self, x):
        Z = x[:6]
        if np.any(Z < self.z_min) or np.any(Z > self.z_max):        # link-length bounds
            return False
        if self.grashof_crank:                                       # Grashof condition
            L = np.sort(Z[:4])
            if L[0] + L[3] > L[1] + L[2] or min(Z[0], Z[1]) > L[0] + 1e-12:
                return False
            if (L[1] + L[2]) - (L[0] + L[3]) < self.grashof_margin * (L[1] + L[2]):
                return False                                         # too close to change point
        pos = self.analyse(x)                                        # mobility / assembly
        if pos is None:
            return False
        sens = sensitivities(x, pos)                                 # [ADDED] conditioning
        if np.min(sens["cond"]) < self.min_cond:
            return False
        if self.design_box is not None:                              # design space
            if not self.inside_design_box(x, pos):
                return False
        if self.body_area is not None:                               # body area
            xmin, xmax, ymin, ymax = self.body_rectangle(x, pos)
            if (xmax - xmin) * (ymax - ymin) > self.body_area * (1 + 1e-9):
                return False
        return True

    # ---------- body-area constraint ----------
    def body_rectangle(self, x, pos=None):
        """
        Smallest body rectangle (xmin, xmax, ymin, ymax) around the fixed pivots
        O2 and O4, in path coordinates, at least body_min_w x body_min_h.
        With body_center_x set, the rectangle is centred on that x.
        """
        pos = self.analyse(x) if pos is None else pos
        Pg = pos["P"]
        mg, md = Pg.mean(), self.Pd.mean()
        psi = (np.angle(np.sum(np.conj(Pg - mg) * (self.Pd - md)))
               if self.orientation_free else 0.0)
        R = np.exp(1j * psi)
        O2 = md - R * mg
        piv = O2 + R * np.array([0, pos["O4"]])                 # placed O2 and O4
        if self.body_center_x is None:
            cx = (piv.real.min() + piv.real.max()) / 2
            w = piv.real.max() - piv.real.min()
        else:
            cx = self.body_center_x
            w = 2 * np.max(np.abs(piv.real - cx))
        cy = (piv.imag.min() + piv.imag.max()) / 2
        h = piv.imag.max() - piv.imag.min()
        w, h = max(w, self.body_min_w), max(h, self.body_min_h)
        return cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2

    # ---------- design-space constraint  [Sec. 5] ----------
    def placed_joints(self, x, pos=None):
        """
        All joint positions over the motion, in the coordinates of the desired path.
        The mechanism is placed with the same rigid-body fit as placement() (the
        error of Eqs. 16-19 does not fix position, so the box is checked after
        placing the linkage where it would actually sit).
        """
        pos = self.analyse(x) if pos is None else pos
        Pg = pos["P"]
        mg, md = Pg.mean(), self.Pd.mean()
        psi = (np.angle(np.sum(np.conj(Pg - mg) * (self.Pd - md)))
               if self.orientation_free else 0.0)
        R = np.exp(1j * psi)
        O2 = md - R * mg
        dense = position_analysis(x, self.th2_check, self.branch, self.cp_side)
        if dense is None:
            return None
        if self.box_joints == "ground":
            pts = np.array([0, dense["O4"]])
        else:
            pts = np.r_[0, dense["O4"], dense["A"], dense["B"], dense["P"]]
        return O2 + R * pts

    def inside_design_box(self, x, pos=None):
        pts = self.placed_joints(x, pos)
        if pts is None:
            return False
        xmin, xmax, ymin, ymax = self.design_box
        return bool(np.all((pts.real >= xmin) & (pts.real <= xmax) &
                           (pts.imag >= ymin) & (pts.imag <= ymax)))

    # ---------- one iteration n -> n+1  [Sec. 4] ----------
    def iterate(self, it):
        nvec = len(self.n_d)
        for j in range(nvec):                       # every couple of consecutive points
            pos = self.analyse(self.x)
            sens = sensitivities(self.x, pos)                               # [Sec. 2]
            n_g, phi_g = displacement_vectors(pos["P"], self.closed)        # [Eq. 17]
            A, B = module_argument_sensitivities(sens["dP"], n_g, phi_g, self.closed)

            # module and argument adjusted alternately along the path  [Sec. 4]
            use_module = (j + it) % 2 == 0
            if use_module:
                resid = self.n_d[j] - n_g[j]
                dsens = A[:, j]                                             # A_k^j
            else:
                if self.orientation_free:                                  # lambda [Eq. 19]
                    if not self.closed and j >= nvec - 1:
                        continue
                    jn = (j + 1) % nvec
                    lam_g = wrap(phi_g[jn] - phi_g[j])
                    lam_d = self.lam_d[j]
                    resid = wrap(lam_d - lam_g)
                    dsens = B[:, jn] - B[:, j]                              # d lambda / dZ
                else:
                    resid = wrap(self.phi_d[j] - phi_g[j])
                    dsens = B[:, j]                                         # B_k^j

            # link increments from the first-order Taylor expansion  [Eqs. 21-23]
            #   dZ_k = (n_d - n_g) / A_k     or     dZ_k = (phi_d - phi_g) / B_k
            dZ = np.full(NVAR, np.nan)
            for k in self.design_vars:
                if abs(dsens[k]) > 1e-9:                # negligible sensitivities rejected
                    dZ[k] = resid / dsens[k]
            if np.all(np.isnan(dZ)):
                continue

            # minimum link increment ratio  rho = |dZ_k / Z_k|  [Eq. 24]
            scale = np.where(np.arange(NVAR) == ITH1, np.pi, np.abs(self.x))
            rho = np.abs(dZ / scale)
            rho[np.isnan(rho)] = np.inf
            # try links in order of increasing rho; if a constraint is violated,
            # "another link is selected based on the priority given by the sensitivity" [Sec. 5]
            for k in np.argsort(rho):
                if not np.isfinite(rho[k]):
                    break
                x_try = self.x.copy()
                x_try[k] += self.relax * dZ[k]          # [ADDED] relax <= 1
                if self.feasible(x_try):
                    self.x = x_try
                    break

    def run(self, max_iter=200, tol=1e-12, verbose=True, method="lm"):
        """Repeat iterations until the synthesis error stops improving  [Sec. 4]."""
        if not self.feasible(self.x):
            raise ValueError("Initial guess violates the constraints / cannot assemble.")
        if method == "lm":
            return self.run_lm(max_iter, tol, verbose)
        E = self.error(self.x)
        self.history = [E]
        if verbose:
            print(f"[paper Sec.4] iter   0   E = {E:.6g}")
        for it in range(1, max_iter + 1):
            x_prev = self.x.copy()
            self.iterate(it)
            E_new = self.error(self.x)
            if E_new > E:
                # [ADDED] safeguard: backtrack along the net change of the sweep,
                # and if that fails reject the sweep and shrink the relaxation
                d = self.x - x_prev
                self.x = x_prev
                for t in (0.5, 0.25, 0.125, 0.0625):
                    xt = x_prev + t * d
                    if self.feasible(xt) and self.error(xt) < E:
                        self.x = xt
                        break
                else:
                    self.relax *= 0.7
                E_new = self.error(self.x)
            self.history.append(E_new)
            if verbose and (it % 10 == 0 or it == 1):
                print(f"[paper Sec.4] iter {it:3d}   E = {E_new:.6g}   relax = {self.relax:.3g}")
            if self.relax < 1e-6:
                break
            E = E_new
        return self.x

    # ---------- Levenberg-Marquardt (optimization method) ----------
    def residuals_jacobian(self, x):
        """
        Residuals of [Eq. 18] (square-rooted weights) and their exact Jacobian:
          d n_g^j / dZ_k      = A_k^j                      [Eq. 13]
          d lambda_g^j / dZ_k = B_k^{j+1} - B_k^j          [Eq. 13 + Eq. 19]
          d phi_g^j / dZ_k    = B_k^j                      (if orientation is kept)
        """
        pos = self.analyse(x)
        sens = sensitivities(x, pos)
        n_g, phi_g = displacement_vectors(pos["P"], self.closed)
        A, B = module_argument_sensitivities(sens["dP"], n_g, phi_g, self.closed)
        if self.orientation_free:
            ang_g, ang_d = relative_angles(phi_g, self.closed), self.lam_d
            dang = (np.roll(B, -1, axis=1) - B) if self.closed else (B[:, 1:] - B[:, :-1])
        else:
            ang_g, ang_d, dang = phi_g, self.phi_d, B
        r = np.r_[np.sqrt(self.w_n) * (self.n_d - n_g),
                  np.sqrt(self.w_phi) * wrap(ang_d - ang_g)]
        J = -np.r_[np.sqrt(self.w_n) * A.T, np.sqrt(self.w_phi) * dang.T]
        if self.angle_link is not None:
            # link-angle residuals; exact gradient d theta5/dZ_k = E_5k, d theta6/dZ_k = E_6k [Eq. 12]
            Ek = sens["E5"] if self.angle_link in ("A->P", "P->A") else sens["E6"]
            j = self.angle_idx
            r = np.r_[r, np.sqrt(self.w_link) * wrap(self.angle_target - self.link_angle(pos)[j])]
            J = np.r_[J, -np.sqrt(self.w_link) * Ek[:, j].T]
        if self.w_area is not None:
            # signed area A = 1/2 sum Im(conj(P_j) P_j+1); exact gradient from dP/dx_k
            Pg, dP = pos["P"], sens["dP"]
            Pn, dPn = np.roll(Pg, -1), np.roll(dP, -1, axis=1)
            dA = 0.5 * np.sum(np.imag(np.conj(dP) * Pn[None, :] + np.conj(Pg)[None, :] * dPn), axis=1)
            r = np.r_[r, np.sqrt(self.w_area) * (self.area_d - signed_area(Pg))]
            J = np.r_[J, -np.sqrt(self.w_area) * dA[None, :]]
        if self.w_pos is not None:
            # position residuals; derivative of the centred generated point is dP - mean(dP)
            # (the best-fit rotation is held fixed in the Jacobian; at the best fit its
            # effect on the gradient is zero)
            rp, R = self.position_residual(pos["P"])
            dP0 = R * (sens["dP"] - sens["dP"].mean(axis=1, keepdims=True))
            r = np.r_[r, np.sqrt(self.w_pos) * rp.real, np.sqrt(self.w_pos) * rp.imag]
            J = np.r_[J, -np.sqrt(self.w_pos) * dP0.real.T, -np.sqrt(self.w_pos) * dP0.imag.T]
        return r, J

    def run_lm(self, max_iter=200, tol=1e-12, verbose=True):
        dv = self.design_vars
        x = self.x.copy()
        r, J = self.residuals_jacobian(x)
        E = r @ r                                   # = E of [Eq. 18]
        self.history = [E]
        mu = 1e-3
        if verbose:
            print(f"[LM] iter   0   E = {E:.6g}")
        for it in range(1, max_iter + 1):
            Jd = J[:, dv]
            g, M = Jd.T @ r, Jd.T @ Jd
            accepted = False
            while mu < 1e10:
                dx = np.linalg.solve(M + mu * np.diag(np.diag(M) + 1e-12), -g)
                xn = x.copy(); xn[dv] += dx
                if self.feasible(xn):               # constraints of [Sec. 5]
                    rn, Jn = self.residuals_jacobian(xn)
                    En = rn @ rn
                    if En < E:
                        accepted = True
                        break
                mu *= 4
            if not accepted:
                break
            dE = E - En
            x, r, J, E = xn, rn, Jn, En
            mu = max(mu / 3, 1e-12)
            self.history.append(E)
            if verbose and (it % 10 == 0 or it == 1):
                print(f"[LM] iter {it:3d}   E = {E:.6g}")
            if dE <= tol * max(self.history[0], 1e-30):
                break
        self.x = x
        return x

    # ---------- final placement (Translation / Rotation Effects)  [Sec. 1, 3] ----------
    def placement(self):
        """
        Because E uses relative coordinates, the optimum is found irrespective of where
        the mechanism sits. Here the optimal mechanism is translated (and, if
        orientation-free, rotated) as a rigid body onto the desired path.
        Returns ground pivot O2, rotation psi, and the placed coupler path.
        """
        Pg = self.analyse(self.x)["P"]
        mg, md = Pg.mean(), self.Pd.mean()
        psi = np.angle(np.sum(np.conj(Pg - mg) * (self.Pd - md))) if self.orientation_free else 0.0
        R = np.exp(1j * psi)
        O2 = md - R * mg
        return O2, psi, O2 + R * Pg


# =============================================================================
# [ADDED] Initial guess. The paper takes the starting linkage from a catalogue of
# coupler curves (Example 1, Sec. 6.1). As a stand-in, sample random crank-rocker
# linkages scaled to the path size ("random atlas"), keep the best few by E [Eq. 18],
# optimise each and return the best.
# =============================================================================
def atlas_search(desired_xy, th2, n_samples=3000, n_refine=8, seed=0, verbose=True,
                 **problem_kwargs):
    rng = np.random.default_rng(seed)
    P = np.asarray(desired_xy, float)
    size = np.max(np.ptp(P, axis=0))
    probe = FourBarPathSynthesis(desired_xy, np.ones(NVAR), th2=th2, **problem_kwargs)
    cands = []
    for _ in range(n_samples):
        Z2 = size * rng.uniform(0.1, 0.6)
        Z1, Z3, Z4 = Z2 * rng.uniform(1.0, 6.0, 3)
        Z5 = Z3 * rng.uniform(0.2, 3.0)
        Z6 = Z3 * rng.uniform(0.2, 3.0)
        x = np.array([Z1, Z2, Z3, Z4, Z5, Z6, rng.uniform(-np.pi, np.pi)])
        for br in (+1, -1):                  # both assembly branches (open / crossed)
            for cp in (+1, -1):              # coupler point on either side of Z3
                probe.branch, probe.cp_side = br, cp
                if probe.feasible(x):
                    cands.append((probe.error(x), x, br, cp))
    if not cands:
        raise ValueError("No starting linkage satisfies the constraints. Enlarge the "
                         "design box / link bounds, or increase n_samples.")
    cands.sort(key=lambda c: c[0])
    best = None
    for E0, x, br, cp in cands[:n_refine]:
        prob = FourBarPathSynthesis(desired_xy, x, th2=th2, branch=br, cp_side=cp,
                                    **problem_kwargs)
        prob.run(max_iter=300, method="lm", verbose=False)
        if verbose:
            print(f"  atlas start E = {E0:10.4g}  ->  optimum E = {prob.history[-1]:.4g}")
        if best is None or prob.history[-1] < best[0].history[-1]:
            best = (prob, x)
    return best          # (problem with optimal x, its initial guess)

def stroke_angle_targets(desired_xy, down_deg=45.0, up_deg=135.0, keep=0.5):
    """
    Link-angle targets for flapping: down_deg where the desired path moves down
    (downstroke), up_deg where it moves up (upstroke). Points near the top and bottom
    of a stroke (wing turning over) are left free: only points whose vertical speed is
    at least `keep` times the maximum get a target. Returns (indices, angles in rad).
    """
    y = np.asarray(desired_xy, float)[:, 1]
    dy = (np.roll(y, -1) - np.roll(y, 1)) / 2
    idx = np.where(np.abs(dy) >= keep * np.max(np.abs(dy)))[0]
    ang = np.where(dy[idx] < 0, np.deg2rad(down_deg), np.deg2rad(up_deg))
    return idx, ang


def cycle_angle_targets(th2, target_deg, reference="y"):
    """
    Link-angle targets from a function of the flap cycle.
    target_deg(x) gives the desired angle in degrees at x = % of the flap cycle
    (0...100), where 0 % is input angle theta2 = 0 and 100 % is one full crank turn.
    reference="y": the angle is measured from the +y-axis, positive towards +x
                   (clockwise), i.e. angle_from_x = 90 deg - angle_from_y
    reference="x": measured from the +x-axis, counter-clockwise
    Every precision point gets a target. Returns (indices, angles from +x in rad).
    """
    th2 = np.asarray(th2, float)
    x_pct = np.mod(th2, 2 * np.pi) / (2 * np.pi) * 100
    ang = np.asarray(target_deg(x_pct), float)
    if reference == "y":
        ang = 90.0 - ang
    return np.arange(len(th2)), np.deg2rad(ang)


def to_reference(angle_x_rad, reference):
    """Angle from +x (rad, ccw) -> degrees in the chosen reference, wrapped to -180...180."""
    a = np.rad2deg(angle_x_rad)
    if reference == "y":
        a = 90.0 - a
    return (a + 180) % 360 - 180


# =============================================================================
# 6. Plotting
# =============================================================================
def plot_result(prob, x_init, fname="four_bar_synthesis_result.png", title=""):
    O2, psi, P_opt = prob.placement()
    R = np.exp(1j * psi)

    # initial guess placed with the same rigid-body fit (for a fair visual comparison)
    x_opt = prob.x
    prob.x = x_init
    _, _, P_init = prob.placement()
    prob.x = x_opt

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    a = ax[0]
    a.plot(prob.Pd.real, prob.Pd.imag, "k-", lw=2.5, label="Desired")
    a.plot(prob.Pd.real, prob.Pd.imag, "k.", ms=3)
    a.plot(P_init.real, P_init.imag, "--", color="tab:red", label="Initial guess")
    a.plot(P_opt.real, P_opt.imag, ":", color="tab:blue", lw=2.5, label="Optimal")

    # draw optimal linkage at precision point 0
    pos = prob.analyse(prob.x)
    j = 0
    pts = {k: O2 + R * v for k, v in dict(O2=0, A=pos["A"][j], B=pos["B"][j],
                                           O4=pos["O4"], P=pos["P"][j]).items()}
    for p, q in [("O2", "A"), ("A", "B"), ("B", "O4"), ("A", "P"), ("P", "B")]:
        a.plot([pts[p].real, pts[q].real], [pts[p].imag, pts[q].imag], "-",
               color="tab:blue", lw=1.5)
    a.plot([pts["O2"].real, pts["O4"].real], [pts["O2"].imag, pts["O4"].imag], "k:", lw=1)
    for k in ("O2", "O4"):
        a.plot(pts[k].real, pts[k].imag, "o", ms=10, mfc="none", mec="k", mew=2)
    for k in ("A", "B", "P"):
        a.plot(pts[k].real, pts[k].imag, "o", ms=6, mfc="w", mec="tab:blue")
    offsets = dict(O2=(9, -13, "left"), O4=(9, 7, "left"), A=(9, -11, "left"),
                   B=(-9, 7, "right"), P=(9, 7, "left"))      # label every joint
    for k, (dx, dy, ha) in offsets.items():
        a.annotate(f"{k} ({pts[k].real:.0f}, {pts[k].imag:.0f})", (pts[k].real, pts[k].imag),
                   textcoords="offset points", xytext=(dx, dy), ha=ha, fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    if prob.design_box is not None:
        xmin, xmax, ymin, ymax = prob.design_box
        a.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin],
               "-", color="tab:green", lw=1.5, label="Design space")
    if prob.body_area is not None:
        xmin, xmax, ymin, ymax = prob.body_rectangle(prob.x)
        a.fill([xmin, xmax, xmax, xmin], [ymin, ymin, ymax, ymax], color="tab:green",
               alpha=0.25, label=f"Body ({xmax - xmin:.0f} x {ymax - ymin:.0f})")
    a.set_aspect("equal"); a.grid(alpha=.3); a.legend(loc="best")
    a.set_title(title or "Path synthesis (cf. Fig. 3 of the paper)")

    b = ax[1]
    b.semilogy(prob.history, "o-", ms=3)
    b.set_xlabel("iteration n"); b.set_ylabel("synthesis error E  [Eq. 18]")
    b.set_title("Convergence"); b.grid(alpha=.3, which="both")
    fig.tight_layout()
    return fig


def plot_link_angle(prob):
    """Angle of the prescribed link over one crank turn, with the targets."""
    pos = prob.analyse(prob.x)
    th = np.rad2deg(prob.th2)
    ref = prob.angle_ref
    fig, ax = plt.subplots()
    ax.plot(th, to_reference(prob.link_angle(pos), ref), label=f"angle of {prob.angle_link}")
    ax.plot(th[prob.angle_idx], to_reference(prob.angle_target, ref), "o", ms=3,
            color="tab:red", label="target")
    ax.set_xlabel(r"Input angle $\theta_2$ (deg)")
    ax.set_ylabel(f"Angle of {prob.angle_link} from {ref}-axis (deg)")
    ax.set_xlim(0, 360); ax.grid(); ax.legend()
    fig.tight_layout()
    return fig


def joint_positions(prob, j=0):
    """
    Positions of all joints of the optimal, placed linkage at precision point j
    (in the coordinates of the desired path):
      O2 = input crank pivot (fixed), O4 = rocker pivot (fixed),
      A  = crank-coupler joint,       B  = coupler-rocker joint,
      P  = coupler point (traces the path).
    """
    O2, psi, _ = prob.placement()
    R = np.exp(1j * psi)
    pos = prob.analyse(prob.x)
    local = dict(O2=0, O4=pos["O4"], A=pos["A"][j], B=pos["B"][j], P=pos["P"][j])
    return {k: O2 + R * v for k, v in local.items()}, psi


def report(prob, x_init):
    O2, psi, _ = prob.placement()
    print("\nDimensions (cf. Table 1 of the paper)")
    print(f"  {'':>8s} {'initial':>12s} {'optimum':>12s}")
    for k in range(6):
        print(f"  {VAR_NAMES[k]:>8s} {x_init[k]:12.3f} {prob.x[k]:12.3f}")
    print(f"  {'theta1':>8s} {x_init[6]:12.4f} {wrap(prob.x[6] + psi):12.4f}  rad (after placement)")
    print(f"  input angles shifted by {psi:.4f} rad")
    print(psi)
    pts, _ = joint_positions(prob, j=0)
    names = dict(O2="fixed pivot, input crank Z2", O4="fixed pivot, rocker Z4",
                 A="moving joint crank-coupler", B="moving joint coupler-rocker",
                 P="coupler point (traces path)")
    print("\nJoint positions at precision point 0 (path coordinates)")
    for k in ("O2", "O4", "A", "B", "P"):
        print(f"  {k:>2s} = ({pts[k].real:9.3f}, {pts[k].imag:9.3f})   {names[k]}")
    print(f"  E: {prob.history[0]:.5g} -> {prob.history[-1]:.5g} in {len(prob.history) - 1} iterations")
    if prob.body_area is not None:
        xmin, xmax, ymin, ymax = prob.body_rectangle(prob.x)
        print(f"  body: x {xmin:.1f} ... {xmax:.1f}, y {ymin:.1f} ... {ymax:.1f}  "
              f"({xmax - xmin:.1f} x {ymax - ymin:.1f}, area {(xmax - xmin) * (ymax - ymin):.0f}"
              f" of max {prob.body_area:.0f})")
    if prob.angle_link is not None:
        pos = prob.analyse(prob.x)
        d = np.rad2deg(wrap(prob.angle_target - prob.link_angle(pos)[prob.angle_idx]))
        print(f"  angle of {prob.angle_link}: RMS error {np.sqrt(np.mean(d**2)):.1f} deg, "
              f"max {np.max(np.abs(d)):.1f} deg over {len(d)} target points")


# =============================================================================
# 7. Path chpice/design space
# =============================================================================
if __name__ == "__main__":
    RUN_EXAMPLE_A = False      # set True to also run Example A (slower)

    s = 120
    th2 = np.linspace(0, 2 * np.pi, s, endpoint=False)
    t = th2
    desired_B = np.c_[50 * np.sin(t) + 0,
                      12.64 * np.sin(2*t) + 15]
    
    #desired_B = np.c_[150 + 75 * np.cos(t),
    #                 125 + 32.5 * np.sin(2 * t)] #Parametric equation for infinity sign (140 mm wide, 70 mm tall)

    # Design space: (xmin, xmax, ymin, ymax) in the same units/coordinates as the path.
    # Set DESIGN_BOX = None to switch the constraint off.
    DESIGN_BOX = None          # fixed box switched off; the body-area constraint is used instead

    # Body: the fixed pivots O2, O4 must fit in a rectangle of at most BODY_AREA,
    # at least BODY_MIN_W wide and BODY_MIN_H tall. Position and shape are free.
    # BODY_CENTER_X = 0 keeps the body centred under the stroke; None = anywhere.
    BODY_AREA = 1000.0         # mm^2  (= 10 cm^2)
    BODY_MIN_W = 10.0          # mm
    BODY_MIN_H = 10.0          # mm
    BODY_CENTER_X = None

    # Wing angle (angle of attack): the wing link follows a target over the flap cycle,
    # measured from the +y-axis, positive towards +x (clockwise).
    # x = % of the flap cycle (0 % at theta2 = 0, 100 % after one crank turn).
    # The wing always contains P, so it lies along A->P (Z5) or B->P (Z6).
    # WING_LINKS lists the options to try; the optimisation is run for each and the
    # design with the lowest total error E is kept.
    #   ["A->P", "B->P"]  try both      ["B->P"]  wing on B->P only
    #   None              path only (no wing-angle target)
    WING_LINKS = ["A->P", "B->P"]
    AOA = 45                                      # angle of attack on both strokes (deg)
    def target_angle(x):
        return -np.sin(2 * np.pi * x / 100) * (90 - AOA)                 # deg from y-axis
    ANGLE_TARGETS = cycle_angle_targets(th2, target_angle, reference="y")
    W_LINK = None              # weight of the angle error; None = default, larger = angle counts more
    W_AREA = 10.0              # weight of the signed-area term (figure-eight vs oval); None = off
    W_POS = 1.0                # weight of the point-by-point position error (shape); None = off

    size = np.max(np.ptp(desired_B, axis=0))     # largest extent of the path
    Z_MIN = 0.025 * size                         # shortest allowed link
    Z_MAX = 3.0 * size                           # longest allowed link

    best = None
    for wing in (WING_LINKS if WING_LINKS is not None else [None]):
        prob, x0 = atlas_search(desired_B, th2, closed=True, orientation_free=True,
                                z_min=Z_MIN, z_max=Z_MAX, n_samples=40000, n_refine=200,
                                design_box=DESIGN_BOX, box_joints="ground",    # only O2 and O4 inside the box
                                body_area=BODY_AREA, body_min_w=BODY_MIN_W, body_min_h=BODY_MIN_H,
                                body_center_x=BODY_CENTER_X, w_area=W_AREA, w_pos=W_POS,
                                angle_link=wing, angle_targets=ANGLE_TARGETS,
                                w_link=W_LINK, angle_ref="y", verbose=False)
        print(f"wing on {wing}: total error E = {prob.history[-1]:.0f}")
        if best is None or prob.history[-1] < best[0].history[-1]:
            best = (prob, x0)
    probB, x0B = best
    print(f"-> best: wing on {probB.angle_link}")
    _, _, PgB = probB.placement()
    print(f"  RMS distance generated-desired path: {np.sqrt(np.mean(np.abs(PgB - probB.Pd)**2)):.3f}")
    report(probB, x0B)
    plot_result(probB, x0B, "infinity_parh.png",
                "infinity path (approximate synthesis)")

    if probB.angle_link is not None:
        plot_link_angle(probB)

    plt.show()                 # opens all figure windows at the end               # opens all figure windows at the end             # opens all figure windows at the end