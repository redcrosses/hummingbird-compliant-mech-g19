from collections import Counter

import numpy as np
import shapely
from scipy.signal import argrelmin

from mechanism import Vector, Mechanism, get_joints

DEFAULT_DIMENSIONS = np.array([
    3,      # L21
    4,      # L22
    5.5,    # L3
    12.8,   # L41
    12.8,   # L42
    13,     # L51
    1.41,   # L52
    9.3,    # L6
    4.4,    # L71
    10.5,   # Lslider
    np.deg2rad(30),  # phi1
    np.deg2rad(5),   # phi2
])


def solve(dimensions, T=10.0, N=100):
    """Build and solve the mechanism. Returns (mechanism, joints_dict, t)."""
    O1, O2, A, B, C, D, E, F, G, O3, O4 = get_joints('O1 O2 A B C D E F G O3 O4')
    G.follow = True
    D.follow = True
    F.follow = True  # so F's positions are recorded (needed for the slider-travel check)

    L21, L22, L3, L41, L42, L51, L52, L6, L71, Lslider, phi1, phi2 = dimensions

    # initial guesses
    theta3, theta4, theta5, theta6, theta_out = np.deg2rad(
        [102.764, 165.512, 141.297, 85.482, 77.8787])
    L_out1, L_out2, L_71 = 18.3097, 18.9186, 14.0

    r21 = Vector((O1, A), r=L21)
    r22 = Vector((O1, B), r=L22)
    r3 = Vector((O2, C), r=L3)
    r41 = Vector((A, D), r=L41)
    r42 = Vector((A, C), r=L42)
    r51 = Vector((B, E), r=L51)
    r52 = Vector((E, F), r=L52)
    r6 = Vector((O2, E), r=L6)
    r71 = Vector((D, F))
    rslider = Vector((D, G), r=Lslider)
    r81 = Vector((O2, O4), r=12.66, theta=0, style='ground')
    r82 = Vector((O4, O1), r=2.16, theta=np.pi / 2, style='ground')
    rout1 = Vector((G, O3), theta=0, style='ground')
    rout2 = Vector((O3, O1), theta=3 * np.pi / 2, style='ground')

    # x: 0 theta3, 1 theta4, 2 theta5, 3 theta6, 4 theta_out, 5 L_out1, 6 L_out2, 7 L_71
    def loops(x, inp):
        temp = np.zeros((4, 2))
        temp[0] = r21(inp) + r42(x[1] + phi2) - r3(x[0]) + r81() + r82()
        temp[1] = r21(inp) + r41(x[1]) + rslider(x[4]) + rout1(x[5]) + rout2(x[6])
        temp[2] = r22(inp + phi1) + r51(x[2]) - r6(x[3]) + r81() + r82()
        temp[3] = (r22(inp + phi1) + r51(x[2]) + r52(x[2])
                   + (-r71(x[7], x[4]) + rslider(x[4])) + rout1(x[5]) + rout2(x[6]))
        return temp.flatten()

    omega = 2 * np.pi / T
    t = np.linspace(0, T, N)

    guess_theta = np.array([theta3, theta4, theta5, theta6, theta_out,
                            L_out1, L_out2, L_71])
    guess_omega = np.zeros_like(guess_theta)
    guess_omega[0] = omega

    theta2 = np.pi + omega * t

    mechanism = Mechanism(
        vectors=(r21, r22, r3, r41, r42, r51, r52, r6, r71, rslider,
                 r81, r82, rout1, rout2),
        origin=O1, loops=loops, guess=(guess_theta, guess_omega), pos=theta2)
    mechanism.iterate()

    joints = dict(O1=O1, O2=O2, A=A, B=B, C=C, D=D, E=E, F=F, G=G, O3=O3, O4=O4)
    return mechanism, joints, t


def crossing_passes(path, point):
    """Sample indices at which the path passes through `point` (local minima of distance)."""
    closed = np.allclose(path[0], path[-1])
    pts = path[:-1] if closed else path
    d = np.hypot(*(pts - point).T)
    tol = 2 * np.hypot(*np.diff(pts, axis=0).T).max()
    idx = argrelmin(d, mode='wrap')[0]
    return idx[d[idx] < tol]


def find_crossings(path, t, tol=0.05):
    """Self-intersections of the traced path, with the times they occur at.

    Returns (points (K,2), times: list of K arrays of t-values near each point).
    """
    curve = shapely.LineString(path)
    nodes = shapely.node(curve)
    ends = Counter()
    for seg in nodes.geoms:
        ends[seg.coords[0]] += 1
        ends[seg.coords[-1]] += 1
    pts = np.array([p for p, c in ends.items() if c > 2]).reshape(-1, 2)

    times = [t[crossing_passes(path, p)] for p in pts]
    return pts, times


def analyze(dimensions=DEFAULT_DIMENSIONS, T=10.0, N=100):
    """Run the mechanism and return the path of G plus velocities.

    Returns a dict:
        t              (N,)    time
        path           (N,2)   traced path of G (x, y)
        path_D         (N,2)   path of D (slider pivot)
        path_F         (N,2)   path of F (pin sliding on the slider)
        F_ratio        (N,)    position of F along the slider: 0 at D, 1 at G
        velocity       (N,2)   G velocity (vx, vy)
        speed          (N,)    |v_G|
        slider_theta   (N,)    slider angle in degrees
        slider_omega   (N,)    slider angular velocity in deg/s
        crossings      (K,2)   self-intersection points of G's path
        crossing_times list of K arrays of times at which G is at each crossing
        crossing_idx   list of K index arrays (samples of each pass through a crossing)
        crossing_angles list of K arrays: slider angle (deg) at each pass
        mechanism      the solved Mechanism object (e.g. for animation)
    """
    mechanism, joints, t = solve(dimensions, T=T, N=N)
    G, D = joints['G'], joints['D']

    gx, gy = np.asarray(G.x_positions, float), np.asarray(G.y_positions, float)
    dx, dy = np.asarray(D.x_positions, float), np.asarray(D.y_positions, float)
    F = joints['F']
    fx, fy = np.asarray(F.x_positions, float), np.asarray(F.y_positions, float)
    # where F sits on the slider D->G: 0 at D, 1 at G, outside [0,1] = off the slider
    slider_len = np.hypot(gx - dx, gy - dy)
    F_ratio = ((fx - dx) * (gx - dx) + (fy - dy) * (gy - dy)) / slider_len ** 2

    path = np.stack([gx, gy], axis=1)
    path_D = np.stack([dx, dy], axis=1)

    vx, vy = np.gradient(gx, t), np.gradient(gy, t)

    slider_theta = np.rad2deg(np.arctan2(gy - dy, gx - dx)) - 90
    slider_omega = np.gradient(np.rad2deg(np.unwrap(np.deg2rad(slider_theta))), t)

    try:
        crossings, crossing_times = find_crossings(path, t)
    except Exception:
        crossings, crossing_times = np.empty((0, 2)), []
    crossing_idx = [crossing_passes(path, c) for c in crossings]

    return dict(t=t, path=path, path_D=path_D,
                path_F=np.stack([fx, fy], axis=1), F_ratio=F_ratio,
                velocity=np.stack([vx, vy], axis=1), speed=np.hypot(vx, vy),
                slider_theta=slider_theta, slider_omega=slider_omega,
                crossings=crossings, crossing_times=crossing_times,
                crossing_idx=crossing_idx,
                crossing_angles=[slider_theta[i] for i in crossing_idx],
                mechanism=mechanism)


def plot_results(result, animate=True, target=None, title=None,
                 save_path=None, show=True):
    """Your original figures (linkage animation; G_x / theta_out and G_y vs time),
    plus extras: optional target path overlay, and orange markers for the path
    self-intersection (location lines + each pass, labelled with the slider
    angle and G speed there).

    Returns (ani, fig, fig2); keep `ani` referenced so the animation keeps running.
    save_path saves the time-history figure (fig2) as an image.
    """
    import matplotlib
    if show:
        matplotlib.use("QtAgg")
    from matplotlib import pyplot as plt

    t, path = result['t'], result['path']
    theta, speed = result['slider_theta'], result['speed']
    crossings, passes = result['crossings'], result['crossing_idx']

    ani = fig = None
    if animate:
        ani, fig, ax = result['mechanism'].get_animation(show_joints=True,)
        ax.set_title(title or '8-bar Slider')
        if target is not None:
            tg = np.asarray(target)
            ax.plot(tg[:, 0], tg[:, 1], '--', color='gray', lw=1.5, label='target')
            ax.legend()
        if len(crossings):
            ax.scatter(crossings[:, 0], crossings[:, 1], color='orange', zorder=5)

    fig2, axs = plt.subplots(2, 1)
    axs[0].set_title(r'Analysis of $G$')
    axs[0].plot(t, path[:, 0], color='red')
    axs[0].set_ylabel(r'$G_x$', color='red')
    axs[0].tick_params(axis='y', labelcolor='red')
    axs[0].set_xlabel(r'Time (s)')

    axs2 = axs[0].twinx()
    axs2.set_ylabel(r'$\theta_{out}$', color='blue')
    axs2.tick_params(axis='y', labelcolor='blue')
    axs2.plot(t, theta, color='blue')

    axs[1].plot(t, path[:, 1], color='maroon')
    axs[1].set_ylabel(r'$G_y$')
    axs[1].set_xlabel(r'Time (s)')

    for c, idx in zip(crossings, passes):
        axs[0].hlines(c[0], xmin=t[0], xmax=t[-1], colors='orange')
        axs[1].hlines(c[1], xmin=t[0], xmax=t[-1], colors='orange')
        axs[0].scatter(t[idx], path[idx, 0], color='orange', zorder=3)
        axs[1].scatter(t[idx], path[idx, 1], color='orange', zorder=3)
        axs2.scatter(t[idx], theta[idx], color='orange', zorder=3)
        for i in idx:
            axs2.annotate(f'{theta[i]:.1f}\u00b0, v={speed[i]:.2f}', (t[i], theta[i]),
                          textcoords='offset points', xytext=(6, 6))

    for a_ in (axs[0], axs[1]):
        a_.minorticks_on()
        a_.grid(which='both')
    fig2.tight_layout()
    if save_path:
        fig2.savefig(save_path, dpi=200)
    if show:
        plt.show()
    return ani, fig, fig2


if __name__ == '__main__':
    res = analyze()
    print('G path shape:', res['path'].shape)
    print('Crossing points:\n', res['crossings'])
    plot_results(res)