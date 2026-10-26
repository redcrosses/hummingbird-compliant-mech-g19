from mechanism import *
import numpy as np
import itertools
import matplotlib
matplotlib.use("TkAgg") 
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon

O2, A, B, O4, C, D, E, F, H = get_joints('O2 A B O4 C D E F H')
H.follow = True  # Traced path joint

# Mechanism vectors
a = Vector((O2, A), r=3.5)      # left actuator body, pin A
b = Vector((O2, B), r=4)        # left actuator body, pin B
c = Vector((O4, C), r=4)        # right actuator body, pin C
d = Vector((O4, D), r=4)        # right actuator body, pin D
e = Vector((A, E), r=23.3)      # L5
f = Vector((C, E), r=23.65)     # L2
g = Vector((B, F), r=20.5)      # L6
h = Vector((D, F), r=20.95)     # L7
k = Vector((F, H), r=9.75)      # L8 (slotted link), H is the traced tip
x = Vector((O2, O4), r=12, theta=0, style='ground')

# Fixed angle between the two pins on each rotary body
phi = np.deg2rad(30)

# 0: th_e, 1: th_f, 2: th_g, 3: th_h, 4: th_k
def loops(t, inp):
    th_b = inp + phi        # left body: B is +30 deg from A
    th_c = -inp             # right actuator: equal speed, opposite direction
    th_d = -inp - phi       # right body: D is -30 deg from C

    # Loop 1: O2 -> A -> E -> C -> O4 -> O2
    l1 = a(inp) + e(t[0]) - f(t[1]) - c(th_c) - x()

    # Loop 2: O2 -> B -> F -> D -> O4 -> O2
    l2 = b(th_b) + g(t[2]) - h(t[3]) - d(th_d) - x()

    # Loop 3: E slides on L8, so vector F->E is parallel to link k (cross product = 0)
    v_FE = a(inp) + e(t[0]) - b(th_b) - g(t[2])
    v_k = k(t[4])
    l3 = v_FE[0] * v_k[1] - v_FE[1] * v_k[0]

    return np.concatenate((l1, l2, [l3]))

# Initial guess: th_e, th_f, th_g, th_h, th_k
guess = np.deg2rad([75.9, 106.8, 55.6, 91.1, 150])

t2 = np.linspace(0, 6*np.pi, 300)
mechanism = Mechanism(
    vectors=(a, b, c, d, e, f, g, h, k, x),
    origin=O2,
    loops=loops,
    pos=t2,
    guess=(guess,)
)
mechanism.iterate()

# ---- Solved joint positions (used for our own colored, labelled animation) ----
def jp(joint, fallback=None):
    try:
        xs = np.array(joint.x_positions, dtype=float)
        ys = np.array(joint.y_positions, dtype=float)
        if xs.shape == t2.shape and not (np.isnan(xs).any() or np.isnan(ys).any()):
            return xs, ys
    except Exception:
        pass
    if fallback is None:
        raise RuntimeError('No solved positions found for joint ' + str(joint))
    return np.full(len(t2), fallback[0]), np.full(len(t2), fallback[1])

P = {
    'O2': jp(O2, (0, 0)), 'O4': jp(O4, (12, 0)),
    'A': jp(A), 'B': jp(B), 'C': jp(C), 'D': jp(D),
    'E': jp(E), 'F': jp(F), 'H': jp(H),
}

# (legend name, joint 1, joint 2, color)
links = [
    ('L1 (ground)',          'O2', 'O4', 'gray'),
    ('L5 (A-E)',             'A',  'E',  'tab:green'),
    ('L2 (C-E)',             'C',  'E',  'tab:olive'),
    ('L6 (B-F)',             'B',  'F',  'tab:cyan'),
    ('L7 (D-F)',             'D',  'F',  'tab:purple'),
    ('L8 (F-H, slotted)',    'F',  'H',  'tab:red'),
]
# (legend name, joints, color) - rigid triangles on the two rotary actuators
bodies = [
    ('L4 (left rotary body)',  ('O2', 'A', 'B'), 'tab:orange'),
    ('L3 (right rotary body)', ('O4', 'C', 'D'), 'tab:blue'),
]

# ---- Target poses (x, y, angle [deg]) in the target frame ----
# Pose 3 is hidden behind pose 7 in your picture, so (0, 0, 45) is assumed.
targets = np.array([
    [-3.0,  0.0,  90],   # Pose 1
    [-1.5,  1.0,  75],   # Pose 2
    [ 0.0,  0.0,  45],   # Pose 3 (assumed)
    [ 1.5, -1.0,  75],   # Pose 4
    [ 3.0,  0.0,  90],   # Pose 5
    [ 1.5,  1.0, 105],   # Pose 6
    [ 0.0,  0.0, 135],   # Pose 7
    [-1.5, -1.0, 105],   # Pose 8
])*5
n_t = len(targets)
t_cols = plt.cm.RdYlBu(np.linspace(0, 1, n_t))   # dark blue (1) -> dark red (8), as in your picture

# Reference point = center of L8 (midpoint of F and H); reference angle = direction of L8 (F -> H)
cx = (P['F'][0] + P['H'][0]) / 2
cy = (P['F'][1] + P['H'][1]) / 2
th_k = np.unwrap(np.arctan2(P['H'][1] - P['F'][1], P['H'][0] - P['F'][0]))

# Placing the target frame in the mechanism frame (EDIT THESE to move/scale the targets)
TARGET_SCALE = 1.0                                   # target units -> cm
TARGET_OFFSET = (0.5 * (cx.max() + cx.min()),        # default: center of the L8 center path
                 0.5 * (cy.max() + cy.min()))
ANGLE_OFFSET = 0.0                                   # deg, added to the L8 angle before comparing

tx = TARGET_OFFSET[0] + TARGET_SCALE * targets[:, 0]
ty = TARGET_OFFSET[1] + TARGET_SCALE * targets[:, 1]
tang = targets[:, 2]

# ---- Errors of the L8 center versus every target pose, over the whole cycle ----
wrap = lambda ang: (ang + 180) % 360 - 180
pos_err = np.array([np.hypot(cx - tx[i], cy - ty[i]) for i in range(n_t)])               # [pose, frame]
ang_err = np.array([wrap(np.rad2deg(th_k) + ANGLE_OFFSET - tang[i]) for i in range(n_t)])  # [pose, frame]
best = pos_err.argmin(axis=1)                      # frame of closest approach per pose
best_pos = pos_err[np.arange(n_t), best]
best_ang = ang_err[np.arange(n_t), best]

fig, ax = plt.subplots(figsize=(8, 8))
fig.subplots_adjust(bottom=0.14)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.set_title('Dual-Actuator Compliant Mechanism')

allx = np.concatenate([P[j][0] for j in P] + [tx])
ally = np.concatenate([P[j][1] for j in P] + [ty])
ax.set_xlim(allx.min() - 3, allx.max() + 3)
ax.set_ylim(ally.min() - 3, ally.max() + 3)

# Full tip path (static) and current tip position
ax.plot(P['H'][0], P['H'][1], color='lightskyblue', lw=1.2, label='Tip path (H)')
ax.plot(cx, cy, color='tab:red', lw=1.0, ls='--', alpha=0.6, label='L8 center path')
tip_dot, = ax.plot([], [], 'o', color='tab:red', ms=6)
center_dot, = ax.plot([], [], 'D', color='black', ms=6, label='L8 center (reference)')

# Target poses: rectangle (long side along the pose angle) + arrow + label
rect_len, rect_wid = 1.0 * TARGET_SCALE, 0.4 * TARGET_SCALE
for i in range(n_t):
    ang = np.deg2rad(tang[i])
    u = np.array([np.cos(ang), np.sin(ang)])
    v = np.array([-np.sin(ang), np.cos(ang)])
    ctr = np.array([tx[i], ty[i]])
    corners = [ctr + s1 * rect_len / 2 * u + s2 * rect_wid / 2 * v
               for s1, s2 in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    ax.add_patch(Polygon(corners, closed=True, facecolor=t_cols[i], edgecolor=t_cols[i],
                         alpha=0.45, label='Target poses' if i == 0 else None))
    ax.annotate('', xy=ctr + 0.6 * rect_len * u, xytext=ctr,
                arrowprops=dict(arrowstyle='->', color=t_cols[i], lw=1.5))
    ax.plot(tx[i], ty[i], 'o', color=t_cols[i], ms=5)
    ax.text(tx[i] + 0.3 * TARGET_SCALE, ty[i] + 0.3 * TARGET_SCALE, 'P' + str(i + 1),
            color=t_cols[i] if i not in (3, 4) else 'dimgray', fontsize=9, fontweight='bold')
# Closest approach of the L8 center to every target
ax.plot(cx[best], cy[best], 'x', color='black', ms=7, label='Closest approach')
for i in range(n_t):
    ax.plot([tx[i], cx[best[i]]], [ty[i], cy[best[i]]], ':', color=t_cols[i], lw=1)

body_patches = []
for name, js, col in bodies:
    poly = Polygon(np.zeros((3, 2)), closed=True, facecolor=col, edgecolor=col, alpha=0.5, label=name)
    ax.add_patch(poly)
    body_patches.append(poly)

link_lines, link_texts = [], []
for name, j1, j2, col in links:
    line, = ax.plot([], [], '-', color=col, lw=4, solid_capstyle='round', label=name)
    txt = ax.text(0, 0, name.split()[0], color='black', fontsize=9, ha='center', va='center',
                  bbox=dict(boxstyle='round,pad=0.15', fc='white', ec=col, alpha=0.85))
    link_lines.append(line)
    link_texts.append(txt)

joint_dots, = ax.plot([], [], 'o', color='black', ms=4)
ax.legend(loc='upper right', fontsize=7)

def draw(i):
    for line, txt, (name, j1, j2, col) in zip(link_lines, link_texts, links):
        x1, y1 = P[j1][0][i], P[j1][1][i]
        x2, y2 = P[j2][0][i], P[j2][1][i]
        line.set_data([x1, x2], [y1, y2])
        txt.set_position(((x1 + x2) / 2, (y1 + y2) / 2))
    for poly, (name, js, col) in zip(body_patches, bodies):
        poly.set_xy(np.array([[P[j][0][i], P[j][1][i]] for j in js]))
    tip_dot.set_data([P['H'][0][i]], [P['H'][1][i]])
    center_dot.set_data([cx[i]], [cy[i]])
    joint_dots.set_data([P[j][0][i] for j in P], [P[j][1][i] for j in P])

# ---- Animation driven by the frequency slider ----
f0 = 1.0                                   # actuator frequency [Hz] (revolutions per second)
state = {'f': f0, 'phase': 0.0}            # phase = left actuator input angle [rad]
frame_dt = 0.03                            # seconds per displayed frame
dtheta = t2[1] - t2[0]

def step(_):
    state['phase'] += 2 * np.pi * state['f'] * frame_dt
    draw(int(state['phase'] / dtheta) % len(t2))
    return []

ani = FuncAnimation(fig, step, frames=itertools.count(), interval=int(frame_dt * 1000),
                    blit=False, cache_frame_data=False)

# ---- Slotted link L8 (k): angle and angular speed ----
dth_dinp = np.gradient(th_k, t2)   # rad of L8 per rad of actuator input

fig2, (ax_ang, ax_spd) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
line_ang, = ax_ang.plot(t2 / (2 * np.pi * f0), np.rad2deg(th_k), color='tab:red', label='L8 angle')
line_spd, = ax_spd.plot(t2 / (2 * np.pi * f0), dth_dinp * 2 * np.pi * f0, color='tab:red',
                        label='L8 angular speed')
# Target angles as reference lines (shifted by 360 deg multiples would be needed if L8 wraps)
for i in range(n_t):
    ax_ang.axhline(tang[i] - ANGLE_OFFSET, color=t_cols[i], lw=0.8, ls=':')
ax_ang.set_ylabel('L8 angle [deg]')
ax_spd.set_ylabel('L8 angular speed [rad/s]')
ax_spd.set_xlabel('Time [s]')
for axx in (ax_ang, ax_spd):
    axx.grid(True)
    axx.legend(loc='upper right')

# ---- Error graph: how far off is the L8 center from each target pose ----
fig3, (ax_pe, ax_ae) = plt.subplots(1, 2, figsize=(11, 4.5))
labels = ['P' + str(i + 1) for i in range(n_t)]
bp = ax_pe.bar(labels, best_pos, color=t_cols, edgecolor='black')
ba = ax_ae.bar(labels, np.abs(best_ang), color=t_cols, edgecolor='black')
ax_pe.set_ylabel('Position error [cm]')
ax_pe.set_title('Closest approach of L8 center to each pose')
ax_ae.set_ylabel('|Angle error| at that moment [deg]')
ax_ae.set_title('L8 angle error at closest approach')
for axx in (ax_pe, ax_ae):
    axx.grid(True, axis='y', alpha=0.3)
for rect, val in zip(bp, best_pos):
    ax_pe.text(rect.get_x() + rect.get_width() / 2, val, '%.1f' % val, ha='center', va='bottom', fontsize=8)
for rect, val in zip(ba, np.abs(best_ang)):
    ax_ae.text(rect.get_x() + rect.get_width() / 2, val, '%.0f' % val, ha='center', va='bottom', fontsize=8)
fig3.tight_layout()

print('Pose  pos err [cm]  angle err [deg]')
for i in range(n_t):
    print('P%d    %8.2f      %8.1f' % (i + 1, best_pos[i], best_ang[i]))
print('Mean position error: %.2f cm' % best_pos.mean())

# One slider (in the animation window) controls the animation AND the plots.
# Left actuator frequency; the right one follows at the same speed, opposite direction.
ax_freq = fig.add_axes([0.2, 0.04, 0.6, 0.03])
freq_slider = Slider(ax_freq, 'Actuator f [Hz]', 0.05, 2.0, valinit=f0)

def update(val):
    fr = freq_slider.val
    state['f'] = fr
    line_ang.set_xdata(t2 / (2 * np.pi * fr))
    line_spd.set_xdata(t2 / (2 * np.pi * fr))
    line_spd.set_ydata(dth_dinp * 2 * np.pi * fr)
    for axx in (ax_ang, ax_spd):
        axx.relim()
        axx.autoscale_view()
    fig2.canvas.draw_idle()

freq_slider.on_changed(update)

plt.show()

print(np.max(H.y_positions))
print(np.min(H.y_positions))
print(np.max(H.y_positions) - np.min(H.y_positions))