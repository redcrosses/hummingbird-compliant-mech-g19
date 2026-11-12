from collections import Counter

import numpy as np
import shapely
from scipy.signal import argrelmin

from mechanism import Vector, Mechanism, get_joints

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

def analyze(dimensions, T=10.0, N=100):
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


if __name__ == '__main__':
    res = analyze()
    print('G path shape:', res['path'].shape)
    print('Crossing points:\n', res['crossings'])
    plot_results(res)