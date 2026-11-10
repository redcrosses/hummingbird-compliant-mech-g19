"""Fit the 8-bar slider linkage to a target path: one SLSQP run from one start.

  minimize    0.5 * || (G(theta2_i) - T_i) / scale ||^2          (path-array residuals)
  subject to  at each pass through the TARGET's self-intersection:
                  dist(path polyline, crossing) <= pin_tol * scale   (pass through it)
              margin <= F_ratio <= 1 - margin at every crank angle      (F stays on slider D-G)
                  v_min <= |v_G| <= v_max                            (velocity gate)
                  a_min <= slider angle <= a_max                     (angle window)
                  |angle_a - angle_b| >= min_angle_sep               (between passes)
              lower <= x <= upper                                    (link-length bounds)

T_i is the target point crank sample i should land on. Because the target has no
timing, T_i is found inside the objective on every evaluation: a monotone, cyclic
nearest-point tracking along the target, refined by exact projection onto the target
polyline. The tracking can only advance a few target points per sample, so the two
branches of a figure-eight are never confused at the crossing.

If your target is already timed to the crank samples, use correspondence='index'
(T_i is then simply target[i]).
"""
import csv

import numpy as np
from scipy.optimize import minimize

from main_upd import (analyze, find_crossings, crossing_passes,
                              plot_results, DEFAULT_DIMENSIONS)

# Free design variables (link lengths). L42 is tied to L41 as in your original script;
# phi1 and phi2 are held fixed (see `phi` in optimize()).
NAMES = ['L21', 'L22', 'L3', 'L41', 'L51', 'L52', 'L6', 'L71', 'Lslider']
DIM_NAMES = ['L21', 'L22', 'L3', 'L41', 'L42', 'L51', 'L52', 'L6', 'L71',
             'Lslider', 'phi1', 'phi2']  # order expected by analyze()


def x_to_dimensions(x, phi=DEFAULT_DIMENSIONS[10:12]):
    L21, L22, L3, L41, L51, L52, L6, L71, Ls = x
    return np.array([L21, L22, L3, L41, L41, L51, L52, L6, L71, Ls, phi[0], phi[1]])


def dimensions_to_x(d):
    return np.array([d[0], d[1], d[2], d[3], d[5], d[6], d[7], d[8], d[9]])


def default_bounds(nominal=DEFAULT_DIMENSIONS, len_frac=0.3):
    return [(v * (1 - len_frac), v * (1 + len_frac)) for v in dimensions_to_x(nominal)]


# ----------------------------------------------------------------------------
# Target path handling
# ----------------------------------------------------------------------------
def resample_closed(path, n=400):
    """Resample a closed polyline at n points, uniform in arc length (no repeated end)."""
    p = np.vstack([path, path[:1]]) if not np.allclose(path[0], path[-1]) else path
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])
    u = np.linspace(0, s[-1], n, endpoint=False)
    return np.stack([np.interp(u, s, p[:, 0]), np.interp(u, s, p[:, 1])], axis=1)


def track_target(Pc, D, window, flips=(False, True)):
    """Assign every path sample to a target sample: monotone and cyclic.

    Each sample may only advance 0..window target points past the previous one. All M
    start points (and the given traversal directions) are tried; the lowest total
    squared distance wins. Returns (indices into D, total squared distance, flip).
    """
    M, N = len(D), len(Pc)
    ar, offs = np.arange(M), np.arange(window + 1)
    best_cost, best_s, best_flip = np.inf, None, False
    for flip in flips:
        Dd = D[::-1] if flip else D
        S = np.empty((N, M), dtype=int)
        s = ar.copy()
        S[0] = s
        cost = ((Dd[s] - Pc[0]) ** 2).sum(1)
        for i in range(1, N):
            cand = (s[:, None] + offs) % M
            d2 = ((Dd[cand] - Pc[i]) ** 2).sum(-1)
            j = d2.argmin(1)
            s = cand[ar, j]
            cost += d2[ar, j]
            S[i] = s
        k = cost.argmin()
        if cost[k] < best_cost:
            seq = S[:, k]
            best_cost, best_s, best_flip = cost[k], ((M - 1 - seq) if flip else seq), flip
    return best_s, best_cost, best_flip


def _wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def interp_at_point(Pc, speed, theta, pt, k):
    """Distance^2 from `pt` to the path polyline near sample k, and the speed / slider
    angle interpolated at the closest point (between the two bracketing samples)."""
    n = len(Pc)
    best = None
    for j in (k - 1, k):
        a, b = j % n, (j + 1) % n
        AB = Pc[b] - Pc[a]
        u = np.clip(np.dot(pt - Pc[a], AB) / max(np.dot(AB, AB), 1e-18), 0.0, 1.0)
        d2 = np.sum((pt - (Pc[a] + u * AB)) ** 2)
        if best is None or d2 < best[0]:
            best = (d2, (1 - u) * speed[a] + u * speed[b], (1 - u) * theta[a] + u * theta[b])
    return best


# ----------------------------------------------------------------------------
# Problem definition
# ----------------------------------------------------------------------------
class PathProblem:
    """Residuals and constraints as functions of the design vector x (9 link lengths)."""

    ANG_SCALE = 30.0  # deg, normalizes angle constraints

    def __init__(self, target, T=10.0, N=100, M=400, gate=None, angle_gate=None,
                 min_angle_sep=None, gate_point=None, pin_tol=0.005,
                 correspondence='track', window=None, phi=DEFAULT_DIMENSIONS[10:12],
                 slider_margin=0.02):
        self.slider_margin = slider_margin
        self.phi = np.asarray(phi, float)      # fixed angles phi1, phi2 (rad)
        target = np.asarray(target, float)
        if np.allclose(target[0], target[-1]):
            target = target[:-1]
        self.correspondence = correspondence
        if correspondence == 'track':
            self.D = resample_closed(target, M)
        elif correspondence == 'index':
            if len(target) != N - 1:
                raise ValueError(f"correspondence='index' needs {N - 1} target points "
                                 f"(one per crank sample, closing point dropped); got {len(target)}")
            self.D = target
        else:
            raise ValueError("correspondence must be 'track' or 'index'")
        self.M = len(self.D)
        self.scale = float(np.hypot(*np.ptp(self.D, axis=0)))
        self.T, self.N, self.window = T, N, window
        self.gate, self.angle_gate, self.min_angle_sep = gate, angle_gate, min_angle_sep
        self.pin_tol = pin_tol
        self.ref_speed, self._flips = 1.0, (False, True)
        self._npts, self._ncon = 2 * (N - 1), 0
        self._key = self._res = self._st_key = self._st = None
        self.last_error = None

        # Self-intersection(s) of the TARGET and the target samples at each pass
        self.cross = []
        if gate is not None or angle_gate is not None or min_angle_sep is not None:
            closed = np.vstack([self.D, self.D[:1]])
            pts, _ = find_crossings(closed, np.arange(len(closed)))
            if gate_point is not None and len(pts):
                pts = pts[[np.argmin(np.hypot(*(pts - np.asarray(gate_point)).T))]]
            self.cross = [(p, crossing_passes(self.D, p)) for p in pts]
            self.cross = [(p, q) for p, q in self.cross if len(q) >= 2]
            if not self.cross:
                raise ValueError('target path has no self-intersection to apply the gate at')

    # -- mechanism evaluation (cached: residuals and constraints share one solve) --
    def evaluate(self, x):
        key = np.asarray(x, float).tobytes()
        if key != self._key:
            try:
                res = analyze(x_to_dimensions(x, self.phi), T=self.T, N=self.N)
                P = res['path']
                ok = (np.all(np.isfinite(P)) and np.all(np.isfinite(res['speed']))
                      and np.all(np.isfinite(res['F_ratio']))
                      and np.hypot(*np.diff(P, axis=0).T).max() < 0.25 * self.scale)
            except Exception as e:
                res, ok = None, False
                self.last_error = f'{type(e).__name__}: {e}'
            self._key, self._res = key, (res if ok else None)
        return self._res

    @staticmethod
    def _pc(P):
        # analyze() spans exactly one crank revolution, so the last sample repeats the
        # first. Always drop it: the array sizes must never depend on solver noise,
        # or SLSQP sees a different number of constraints from one call to the next.
        return P[:-1]

    def assign(self, Pc):
        if self.correspondence == 'index':
            return np.arange(len(Pc))
        w = self.window or int(max(8, 6 * self.M / len(Pc)))
        s, _, flip = track_target(Pc, self.D, w, self._flips)
        self._flips = (flip,)       # direction is fixed by the first (starting) design
        return s

    def targets(self, Pc, s):
        """Target point for every crank sample: the exact projection of the path point
        onto the target polyline segments next to the assigned sample."""
        if self.correspondence == 'index':
            return self.D[s]
        D, M = self.D, self.M
        best = D[s].copy()
        bd = ((Pc - best) ** 2).sum(1)
        for nb in (s - 1, s + 1):
            A, B = D[s], D[nb % M]
            AB = B - A
            u = np.clip(((Pc - A) * AB).sum(1) / np.maximum((AB ** 2).sum(1), 1e-18), 0, 1)
            Q = A + u[:, None] * AB
            d = ((Pc - Q) ** 2).sum(1)
            m = d < bd
            best[m], bd[m] = Q[m], d[m]
        return best

    def crank_indices(self, s):
        """For every target pass, the crank sample assigned closest to it."""
        out = []
        for pt, passes in self.cross:
            ks = [int(np.abs((s - p + self.M / 2) % self.M - self.M / 2).argmin())
                  for p in passes]
            out.append((pt, ks))
        return out

    def state(self, x):
        """(path, target points, crossing groups, mechanism result) for design x,
        with the target assignment recomputed for this x. Cached per x."""
        res = self.evaluate(x)
        if res is None:
            return None
        if self._st_key != self._key:
            Pc = self._pc(res['path'])
            s = self.assign(Pc)
            self._st = (Pc, self.targets(Pc, s), self.crank_indices(s), res)
            self._st_key = self._key
        return self._st

    # -- smooth path-array residual --
    def residual(self, x):
        st = self.state(x)
        if st is None or st[0].size != self._npts:
            return np.full(self._npts, 10.0)
        return ((st[0] - st[1]) / self.scale).ravel()

    # -- constraints, g >= 0 --
    def _pass_values(self, res, groups):
        """For every pass through a target crossing: squared distance from the crossing
        to the path polyline, and speed / slider angle interpolated AT the crossing
        (between the two path samples that bracket it, near the assigned crank sample)."""
        Pc, speed, theta = self._pc(res['path']), res['speed'], res['slider_theta']
        n = len(Pc)
        vals = []
        for pt, ks in groups:
            per_group = []
            for k in ks:
                per_group.append(interp_at_point(Pc, speed, theta, pt, k))
            vals.append(per_group)
        return vals

    def _g(self, res, groups):
        g = []
        for per_group in self._pass_values(res, groups):
            for d2, v, a in per_group:
                g.append(1.0 - d2 / (self.pin_tol * self.scale) ** 2)
                if self.gate is not None:
                    vmin, vmax = self.gate
                    if np.isfinite(vmin):
                        g.append((v - vmin) / self.ref_speed)
                    if np.isfinite(vmax):
                        g.append((vmax - v) / self.ref_speed)
                if self.angle_gate is not None:
                    g.append((a - self.angle_gate[0]) / self.ANG_SCALE)
                    g.append((self.angle_gate[1] - a) / self.ANG_SCALE)
            if self.min_angle_sep is not None:
                for i in range(len(per_group)):
                    for j in range(i + 1, len(per_group)):
                        d = abs(_wrap180(per_group[i][2] - per_group[j][2]))
                        g.append((d - self.min_angle_sep) / self.ANG_SCALE)
        if self.slider_margin is not None:      # F must stay between D and G
            r = res['F_ratio'][:len(self._pc(res['path']))]
            g.extend(r - self.slider_margin)
            g.extend((1.0 - self.slider_margin) - r)
        return np.array(g)

    def constraints(self, x):
        st = self.state(x)
        g = None if st is None else self._g(st[3], st[2])
        if g is None or g.size != self._ncon:      # never change the constraint count
            return np.full(self._ncon, -1.0)
        return g

    # -- the single solve --
    def fit(self, x0, bounds, maxiter=200, verbose=False):
        x0 = np.asarray(x0, float)
        lo, hi = np.array(bounds).T
        xref = np.maximum(0.5 * (np.abs(lo) + np.abs(hi)), 1e-9)   # normalized u = x/xref
        self._flips = (False, True)
        st = self.state(x0)
        if st is None:
            return None
        self._npts = st[0].size
        self.ref_speed = max(float(np.mean(st[3]['speed'])), 1e-9)
        self._ncon = len(self._g(st[3], st[2]))
        cons = ([dict(type='ineq', fun=lambda u: self.constraints(u * xref))]
                if self._ncon else [])
        sol = minimize(lambda u: 0.5 * np.sum(self.residual(u * xref) ** 2), x0 / xref,
                       method='SLSQP', bounds=list(zip(lo / xref, hi / xref)),
                       constraints=cons,
                       options=dict(maxiter=maxiter, ftol=1e-12, eps=1e-5))
        if verbose:
            print(f'SLSQP: {sol.nit} iterations, status {sol.status} ({sol.message})')
        x = sol.x * xref
        return x if self.state(x) is not None and np.all(np.isfinite(x)) else x0

    def diagnostics(self, x):
        """Final tracking error and constraint values for design x."""
        st = self.state(x)
        if st is None:
            return None
        Pc, T, groups, res = st
        err = np.hypot(*(Pc - T).T)
        g = self._g(res, groups) if groups else np.array([])
        pv = [v for grp in self._pass_values(res, groups) for v in grp]
        fr = res['F_ratio'][:len(Pc)]
        return dict(rms_error=float(np.sqrt(np.mean(err ** 2))), max_error=float(err.max()),
                    F_ratio_range=(float(fr.min()), float(fr.max())),
                    min_constraint=float(g.min()) if len(g) else 0.0,
                    speeds=np.array([v[1] for v in pv]), angles=np.array([v[2] for v in pv]),
                    crossings=[p for p, _ in groups])


# ----------------------------------------------------------------------------
# Output: parameter CSV
# ----------------------------------------------------------------------------
def save_parameters_csv(out, path='optimized_parameters.csv', nominal=DEFAULT_DIMENSIONS):
    """Write the optimized dimensions (plus nominal and search bounds) to CSV.
    Angles are stored in radians, i.e. exactly what analyze() expects."""
    bmap = dict(zip(NAMES, out['bounds'])) if out.get('bounds') is not None else {}
    if 'L41' in bmap:
        bmap['L42'] = bmap['L41']
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['parameter', 'value', 'unit', 'nominal', 'lower_bound', 'upper_bound'])
        for name, v, nom in zip(DIM_NAMES, out['dimensions'], nominal):
            lo, hi = bmap.get(name, ('', ''))
            w.writerow([name, repr(float(v)), 'rad' if name.startswith('phi') else 'length',
                        repr(float(nom)), lo, hi])
    return path


def load_parameters_csv(path='optimized_parameters.csv'):
    """Read a CSV written by save_parameters_csv -> 12-vector for analyze()."""
    with open(path, newline='') as f:
        vals = {row['parameter']: float(row['value']) for row in csv.DictReader(f)}
    return np.array([vals[n] for n in DIM_NAMES])


# ----------------------------------------------------------------------------
# Quick look at a saved design
# ----------------------------------------------------------------------------
def plot_from_csv(csv_path='optimized_parameters.csv', target=None, T=10.0, N=100,
                  show=True, animate=True, save_path=None, verbose=True):
    """Load a parameter CSV, simulate it, plot it and print the key numbers.

    Use the same T and N as the optimization (speeds scale with 1/T). `target`
    (optional) is overlaid on the plots. Speeds and angles at the path's
    self-intersection are interpolated at the crossing itself.

    Returns dict: dimensions, result (analyze output), speed_stats, crossing
    (list of dicts: point, times, speeds, angles), F_ratio_range.
    """
    dims = load_parameters_csv(csv_path)
    res = analyze(dims, T=T, N=N)
    Pc, speed, theta, t = res['path'][:-1], res['speed'], res['slider_theta'], res['t']

    crossing = []
    for pt, idx in zip(res['crossings'], res['crossing_idx']):
        vals = [interp_at_point(Pc, speed, theta, pt, int(k)) for k in idx]
        crossing.append(dict(point=pt, times=t[idx], speeds=np.array([v[1] for v in vals]),
                             angles=np.array([v[2] for v in vals])))
    fr = res['F_ratio'][:-1]
    stats = dict(min=float(speed.min()), mean=float(speed.mean()), max=float(speed.max()))
    out = dict(dimensions=dims, result=res, speed_stats=stats, crossing=crossing,
               F_ratio_range=(float(fr.min()), float(fr.max())))

    if verbose:
        print(f'Parameters from {csv_path}  (T={T} s, N={N})')
        for name, v in zip(DIM_NAMES, dims):
            extra = f'  ({np.rad2deg(v):.2f} deg)' if name.startswith('phi') else ''
            print(f'  {name:8s} {v:10.4f}{extra}')
        print(f"G speed: min {stats['min']:.3f}  mean {stats['mean']:.3f}  max {stats['max']:.3f}")
        print(f"F position along slider D->G (0=D, 1=G): {fr.min():.3f} .. {fr.max():.3f}")
        if not crossing:
            print('no self-intersection found on the traced path')
        for c in crossing:
            print(f"Intersection at ({c['point'][0]:.3f}, {c['point'][1]:.3f}):")
            for tt, v, a in zip(c['times'], c['speeds'], c['angles']):
                print(f'   t = {tt:5.2f} s   speed = {v:8.3f}   slider angle = {a:8.2f} deg')

    plot_results(res, animate=animate, target=target,
                 title=f'8-bar Slider ({csv_path})', save_path=save_path, show=show)
    return out


# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------
def optimize(target, gate=None, angle_gate=None, min_angle_sep=None, gate_point=None,
             bounds=None, x0=None, T=10.0, N=100, M=400, pin_tol=0.005,
             correspondence='track', phi=None, slider_margin=0.02, maxiter=200, verbose=True,
             csv_path='optimized_parameters.csv', plot=True,
             plot_path='optimized_solution.png'):
    """Fit linkage lengths to `target` with a single SLSQP run from x0.

    target          (M,2) closed target path for G (one crank revolution)
    gate            (v_min, v_max) speed window for G at the target's crossing
                    (use -inf / inf for one-sided; interpolated AT the crossing)
    angle_gate      (a_min, a_max) slider angle window (deg) at each pass
    min_angle_sep   minimum angle difference (deg) between the passes
    gate_point      (x, y): gate only the target crossing nearest this point
    pin_tol         how close G must pass to the crossing (fraction of the target's
                    bounding-box diagonal)
    correspondence  'track' (target untimed) or 'index' (one target point per crank sample)
    x0              starting link lengths (9 free variables); default = DEFAULT_DIMENSIONS
    phi             fixed (phi1, phi2) in rad; default = DEFAULT_DIMENSIONS[10:12]
    slider_margin   F must stay within [margin, 1-margin] of the slider length D->G for
                    the whole crank cycle (None disables)
    csv_path / plot / plot_path  outputs (None / False to skip)
    """
    bounds = bounds or default_bounds()
    lo, hi = np.array(bounds).T
    x_start = dimensions_to_x(DEFAULT_DIMENSIONS) if x0 is None else np.asarray(x0, float)
    if len(bounds) != len(NAMES) or x_start.shape != (len(NAMES),):
        raise ValueError(f'x0 and bounds must have {len(NAMES)} entries ({", ".join(NAMES)}); '
                         f'got x0 {x_start.shape}, bounds {np.shape(bounds)}. phi1/phi2 are '
                         f'no longer optimized: pass them via phi=(phi1, phi2).')
    x_start = np.clip(x_start, lo, hi)

    phi = DEFAULT_DIMENSIONS[10:12] if phi is None else phi
    prob = PathProblem(target, T, N, M, gate, angle_gate, min_angle_sep, gate_point,
                       pin_tol, correspondence, phi=phi, slider_margin=slider_margin)
    x = prob.fit(x_start, bounds, maxiter, verbose)
    d = None if x is None else prob.diagnostics(x)
    if d is None:
        raise RuntimeError('starting design failed to evaluate; change x0 or the bounds. '
                           f'Last error from analyze(): {prob.last_error}')

    feasible = d['min_constraint'] >= -1e-3
    dims = x_to_dimensions(x, prob.phi)
    result = analyze(dims, T=T, N=N)
    out = dict(dimensions=dims, x=x, result=result, bounds=bounds, feasible=feasible,
               rms_error=d['rms_error'], max_error=d['max_error'],
               gate_speeds=d['speeds'], gate_angles=d['angles'],
               target_crossings=d['crossings'], cost=d['rms_error'],
               F_ratio_range=d['F_ratio_range'])
    if verbose:
        print('\ndesign:')
        for n, v in zip(NAMES, x):
            print(f'  {n:8s} {v:.4f}')
        print(f"rms path error {d['rms_error']:.4g}, max {d['max_error']:.4g}, feasible={feasible}")
        print('F position along slider D->G (0=D, 1=G): '
              f"{d['F_ratio_range'][0]:.3f} .. {d['F_ratio_range'][1]:.3f}")
        if len(d['speeds']):
            print('speed at crossing passes:', np.round(d['speeds'], 3))
            print('slider angle at crossing passes (deg):', np.round(d['angles'], 2))

    if csv_path:
        save_parameters_csv(out, csv_path)
        if verbose:
            print('saved parameters ->', csv_path)
    if plot or plot_path:
        plot_results(result, animate=plot, target=target,
                     title=f"Optimized 8-bar Slider (rms error {d['rms_error']:.3g})",
                     save_path=plot_path, show=plot)
        if verbose and plot_path:
            print('saved plot ->', plot_path)
    return out

if __name__ == '__main__':
    # Self-test: start from a perturbed design and recover the nominal mechanism's
    # own path. Replace `target` and the gate with your real requirements.
    # target = nom['path']

    nom = analyze(DEFAULT_DIMENSIONS)
    t = np.linspace(0, 2*np.pi, 100)
    a, b, c, d, e = 0.6, 5.6, 10.3, -12.5, -1.5
    target = np.vstack([b*np.sin(t)+d, e*np.sin(a*np.sin(2*t)+c)+8]).T
    print(target)

    sp = np.concatenate([nom['speed'][i] for i in nom['crossing_idx']] or [[1.0]])
    x_start = dimensions_to_x(DEFAULT_DIMENSIONS) * 1.1
    out = optimize(target, gate=(0.9 * sp.max(), 1 * sp.max()), x0=x_start)

    # plot_from_csv('8-bar-speedy/optimized_parameters.csv', target=target)