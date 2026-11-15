#6-bar mechanism: four-bar O2-A-B-O4 with coupler point P, wing link P-C, and link 6 C-D.
#D sits on the crank, the rocker or the body (as found by six_bar_wing.py). The wing chord
#is fixed to link P-C at an offset angle. The design is read from six_bar_design.json,
#which six_bar_wing.py writes. The crank turns backwards (clockwise), so the downstroke
#runs towards -x.
from mechanism import *
import json
import os
import numpy as np
import matplotlib
matplotlib.use("QtAgg")            # animation in its own window
from matplotlib import pyplot as plt

# black and white figures with LaTeX-style text (Computer Modern)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['cmr10'],
    'mathtext.fontset': 'cm',
    'axes.formatter.use_mathtext': True,
    'axes.unicode_minus': False,
    'font.size': 11,
    'axes.prop_cycle': plt.cycler(color=['k']),
})

# Design from six_bar_wing.py (lengths in mm, angles in rad), placed coordinates, O2 at (0, 0)
DESIGN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'six_bar_design.json')
with open(DESIGN_FILE) as f:
    d = json.load(f)
Z1, Z2, Z3, Z4, Z5, Z6 = d['Z']
th1 = d['th1']           # placed ground angle O2->O4
psi = d['psi']           # crank angle at precision point 0
WHERE = d['where']       # link 6 attached to the 'crank', 'rocker' or 'ground'
L_D, beta = d['L_D'], d['beta']   # crank: D from O2, angle from the crank O2->A
                                  # rocker: D from O4, angle from the rocker O4->B
                                  # ground: D (= O6) from O2, angle from the x-axis
L_PC = d['L_PC']                  # wing link P-C
L_DC = d['L_DC']                  # link 6, D -> C
OFFSET = d['offset']              # wing chord = direction P->C turned by OFFSET (counter-clockwise)
guess = np.deg2rad(d['guess_deg'])   # angles theta3 ... theta8 at the first input angle
print(f"Design: link 6 on the {WHERE}, path RMS {d['path_rms_mm']:.2f} mm, "
      f"wing angle RMS {d['wing_rms_deg']:.1f} deg (optimiser)")
CHORD = 40                       # chord length drawn in the animation (mm), only for the picture

AOA = 45                         # target angle of attack on both strokes (deg)

O2, O4, A, B, P, C, D = get_joints('O2 O4 A B P C D')
P.follow = True
C.follow = True

r1 = Vector((O2, O4), r=Z1, theta=th1, style='ground')
r2 = Vector((O2, A), r=Z2)        # crank
if WHERE == 'crank':
    r10 = Vector((O2, D), r=L_D)                          # second point on the crank
elif WHERE == 'rocker':
    r10 = Vector((O4, D), r=L_D)                          # second point on the rocker
else:
    r10 = Vector((O2, D), r=L_D, theta=beta, style='ground')   # fixed pivot O6 = D
r3 = Vector((A, B), r=Z3)
r4 = Vector((B, O4), r=Z4)
r5 = Vector((A, P), r=Z5)
r6 = Vector((P, B), r=Z6)
r7 = Vector((P, C), r=L_PC)       # wing
r8 = Vector((D, C), r=L_DC)       # link 6

def O2_to_D(t, inp):
    """Vector from O2 to D."""
    if WHERE == 'crank':
        return r10(inp + beta)
    if WHERE == 'rocker':                     # rocker O4->B points along t[1] + pi (r4 runs B->O4)
        return r1() + r10(t[1] + np.pi + beta)
    return r10()


# 0: theta3, 1: theta4, 2: theta5, 3: theta6, 4: theta7 (wing), 5: theta8 (link 6)
def loops(t, inp):
    temp = np.zeros((3, 2))
    temp[0] = r2(inp) + r3(t[0]) + r4(t[1]) - r1()                      # four-bar loop
    temp[1] = r5(t[2]) + r6(t[3]) - r3(t[0])                            # coupler triangle
    temp[2] = O2_to_D(t, inp) + r8(t[5]) - r2(inp) - r5(t[2]) - r7(t[4])  # wing loop O2-D-C-P-A-O2
    return temp.flatten()



T = 5
omega2 = 2*np.pi/T
t = np.linspace(0, 3*T, 300)
t2 = psi - omega2*t      # crank turns backwards: downstroke towards -x

mechanism = Mechanism(vectors=(r1, r2, r10, r3, r4, r5, r6, r7, r8), origin=O2, loops=loops,
                      pos=t2, vel=-omega2*np.ones_like(t), guess=(guess, np.zeros(6)))
mechanism.iterate()
ani, fig, ax = mechanism.get_animation()

ax.set_title(r'Six-bar mechanism')

# Position & velocity analysis
fig2, axs = plt.subplots(2, 1)
axs[0].set_title(r'Analysis of $P$')

axs[0].plot(t, P.x_positions, color='k')
axs[0].set_ylabel(r'$P_x$ (mm)')
v0 = axs[0].twinx()
v0.plot(t, P.x_velocities, color='k', linestyle='--')
v0.set_ylabel(r'$v_{P,x}$ (mm/s)')

axs[1].plot(t, P.y_positions, color='k')
axs[1].set_ylabel(r'$P_y$ (mm)')
v1 = axs[1].twinx()
v1.plot(t, P.y_velocities, color='k', linestyle='--')
v1.set_ylabel(r'$v_{P,y}$ (mm/s)')
axs[1].set_xlabel(r'Time $t$ (s)')

for a in (axs[0], axs[1]):
    a.minorticks_on()
    a.grid(which='both')
fig2.tight_layout()

# one flap cycle (one crank turn)
n = np.sum(t2 >= psi - 2*np.pi)
th2 = t2[:n]
th3 = np.unwrap(r3.pos.thetas[:n])
th4 = np.unwrap(r4.pos.thetas[:n])
th6 = np.unwrap(r6.pos.thetas[:n])
th7 = np.unwrap(r7.pos.thetas[:n])
th8 = np.unwrap(r8.pos.thetas[:n])

# % of the flap cycle, following the motion of P:
# downstroke = P moves towards -x (0-50 %), upstroke = P moves towards +x (50-100 %)
x = P.x_positions[:n]
down = np.diff(np.r_[x, x[0]]) < 0                    # step i -> i+1 goes towards -x
i0 = np.where(down & ~np.roll(down, 1))[0][0]         # start of the downstroke
i1 = np.where(~down & np.roll(down, 1))[0][0]         # start of the upstroke
n_down = (i1 - i0) % n
phase = np.empty(n)
for k in range(n):
    i = (i0 + k) % n
    phase[i] = 50*k/n_down if k < n_down else 50 + 50*(k - n_down)/(n - n_down)
order = np.argsort(phase)

# flexure angles: change of the angle between the two links at each joint
joint_B  = np.rad2deg(th4 - th3)                      # coupler - rocker
joint_O4 = np.rad2deg(th4 - th1)                      # rocker - body
joint_P  = np.rad2deg(th7 - th6)                      # coupler - wing
joint_C  = np.rad2deg(th8 - th7)                      # wing - link 6
if WHERE == 'crank':
    joint_D = np.rad2deg(th8 - (th2 + beta))          # crank - link 6 (turns fully: bearing)
elif WHERE == 'rocker':
    joint_D = np.rad2deg(th8 - (th4 + np.pi + beta))  # rocker - link 6
else:
    joint_D = np.rad2deg(th8 - beta)                  # body - link 6 (change only matters)

# compliant joints (D turns fully when it sits on the crank, so it is left out then)
flex = [(r'Joint $B$', joint_B, '-'), (r'Joint $O_4$', joint_O4, '--'),
        (r'Joint $P$', joint_P, '-.'), (r'Joint $C$', joint_C, ':')]
if WHERE != 'crank':
    flex.append((r'Joint $D$', joint_D, (0, (5, 1, 1, 1, 1, 1))))

fig3, ax3 = plt.subplots()
for name, v, ls in flex:
    ax3.plot(phase[order], (v - v[i0])[order], color='k', linestyle=ls, label=name)
ax3.axvspan(0, 50, color='0.9', label='Downstroke')
ax3.set_xlim(0, 100)
ax3.set_xticks(np.arange(0, 101, 10))
ax3.set_title(r'Joint rotations')
ax3.set_xlabel(r'Flap cycle ($\%$)')
ax3.set_ylabel(r'Joint rotation $\Delta\phi$ ($^\circ$)')
ax3.legend(); ax3.minorticks_on(); ax3.grid(which='both')
fig3.tight_layout()

# wing angle: chord C -> P, measured from the +y-axis, positive towards +x (clockwise)
wing_x = np.rad2deg(th7 + OFFSET)                     # chord = P -> C turned by OFFSET
wing_y = (90 - wing_x + 180) % 360 - 180

# target wing angle, x = % of the flap cycle
x_cycle = np.linspace(0, 100, 201)
target = -np.sin(2*np.pi*x_cycle/100)*(90 - AOA)

fig5, ax5 = plt.subplots()
ax5.axvspan(0, 50, color='0.9', label='Downstroke')
ax5.plot(phase[order], wing_y[order], color='k', label=r'Wing chord')
ax5.plot(x_cycle, target, color='k', linestyle='--', label='Target')
ax5.set_xlim(0, 100)
ax5.set_xticks(np.arange(0, 101, 10))
ax5.set_title(r'Wing angle relative to the $y$-axis')
ax5.set_xlabel(r'Flap cycle ($\%$)')
ax5.set_ylabel(r'Wing angle $\theta_w$ ($^\circ$)')
ax5.legend(); ax5.minorticks_on(); ax5.grid(which='both')
fig5.tight_layout()

# ---------------------------------------------------------------------------
# Animation: mechanism + wing angle + joint rotations, with a marker showing where you are in the cycle
# ---------------------------------------------------------------------------
from matplotlib.animation import FuncAnimation

def xy(J):                                   # positions of a joint over one cycle
    return np.asarray(J.x_positions[:n]), np.asarray(J.y_positions[:n])

pts = {name: xy(J) for name, J in (('A', A), ('B', B), ('P', P), ('C', C), ('D', D))}
pts['O2'] = (np.zeros(n), np.zeros(n))
pts['O4'] = (np.full(n, Z1*np.cos(th1)), np.full(n, Z1*np.sin(th1)))

links = [('O2', 'A', 1.5), ('A', 'B', 1.5), ('B', 'O4', 1.5),
         ('A', 'P', 1.5), ('P', 'B', 1.5), ('O4' if WHERE == 'rocker' else 'O2', 'D', 1.5),
         ('D', 'C', 1.5), ('P', 'C', 1.5)]
# wing chord: through P, at the angle wing_x (from +x), drawn thick
chord_dir = np.exp(1j*np.deg2rad(wing_x))

fig6 = plt.figure(figsize=(14, 7))
gs = fig6.add_gridspec(2, 2, width_ratios=[1.1, 1])
axm = fig6.add_subplot(gs[:, 0])        # mechanism (left, full height)
axw = fig6.add_subplot(gs[0, 1])        # wing angle (top right)
axj = fig6.add_subplot(gs[1, 1], sharex=axw)   # joint rotations (bottom right)

# left: mechanism
allx = np.concatenate([v[0] for v in pts.values()])
ally = np.concatenate([v[1] for v in pts.values()])
axm.set_xlim(allx.min() - CHORD/2 - 5, allx.max() + CHORD/2 + 5)
axm.set_ylim(ally.min() - CHORD/2 - 5, ally.max() + CHORD/2 + 5)
axm.set_aspect('equal'); axm.grid()
axm.set_xlabel(r'$x$ (mm)'); axm.set_ylabel(r'$y$ (mm)')
axm.plot([0, pts['O4'][0][0]], [0, pts['O4'][1][0]], 'k--')       # ground
axm.plot(*pts['P'], color='0.75', linestyle=':')                   # full path of P
trace, = axm.plot([], [], color='0.4', lw=1.5)                     # part already travelled
lines = {(a, b): axm.plot([], [], '-o', color='k', lw=w, ms=4, mfc='w')[0]
         for a, b, w in links}
labels = {k: axm.annotate(f'${k}$' if len(k) == 1 else f'${k[0]}_{k[1]}$', (0, 0),
                          textcoords='offset points', xytext=(5, 5), fontsize=10)
          for k in pts}
chord, = axm.plot([], [], color='k', lw=5, solid_capstyle='round')   # the wing itself
title = axm.set_title('')

# top right: wing angle vs % of flap cycle
axw.axvspan(0, 50, color='0.9', label='Downstroke')
axw.plot(phase[order], wing_y[order], color='k', label=r'Wing chord')
axw.plot(x_cycle, target, color='k', linestyle='--', label='Target')
vline_w = axw.axvline(0, color='0.5', lw=1)
dot_w, = axw.plot([], [], 'o', color='k', ms=7)
axw.set_xlim(0, 100); axw.set_xticks(np.arange(0, 101, 10))
axw.set_ylabel(r'Wing angle $\theta_w$ ($^\circ$)')
axw.set_title(r'Wing angle relative to the $y$-axis')
axw.legend(loc='lower right', fontsize=9); axw.grid()

# bottom right: joint rotations vs % of flap cycle
joints = flex
axj.axvspan(0, 50, color='0.9')
dots_j = []
for name, v, ls in joints:
    rot = v - v[i0]
    axj.plot(phase[order], rot[order], color='k', linestyle=ls, label=name)
    dots_j.append((axj.plot([], [], 'o', color='k', ms=5)[0], rot))
vline_j = axj.axvline(0, color='0.5', lw=1)
axj.set_xlabel(r'Flap cycle ($\%$)')
axj.set_ylabel(r'Joint rotation $\Delta\phi$ ($^\circ$)')
axj.set_title(r'Joint rotations')
axj.legend(loc='upper right', fontsize=9); axj.grid()
fig6.tight_layout()

frames = [(i0 + k) % n for k in range(int(n))]          # start the animation at the start of the downstroke

def update(k):
    i = frames[k]
    for (a, b), ln in lines.items():
        ln.set_data([pts[a][0][i], pts[b][0][i]], [pts[a][1][i], pts[b][1][i]])
    for name, lab in labels.items():
        lab.xy = (pts[name][0][i], pts[name][1][i])
    done = frames[:k + 1]
    trace.set_data(pts['P'][0][done], pts['P'][1][done])
    Pi = pts['P'][0][i] + 1j*pts['P'][1][i]
    ends = Pi + np.array([-0.5, 0.5])*CHORD*chord_dir[i]
    chord.set_data(ends.real, ends.imag)
    vline_w.set_xdata([phase[i], phase[i]])
    vline_j.set_xdata([phase[i], phase[i]])
    dot_w.set_data([phase[i]], [wing_y[i]])
    for d, rot in dots_j:
        d.set_data([phase[i]], [rot[i]])
    stroke = 'Downstroke' if phase[i] < 50 else 'Upstroke'
    title.set_text(f'{stroke}: ${phase[i]:.0f}\\%$ of flap cycle')
    return (*lines.values(), *labels.values(), trace, chord, vline_w, vline_j, dot_w,
            *[d for d, _ in dots_j], title)

ani2 = FuncAnimation(fig6, update, frames=int(n), interval=50, blit=False)
ani2.save('MotionGIF.gif', writer='pillow', fps=20, dpi=100)   # <- add this line

plt.show()

# ani.save('../animations/sixbar.mp4', dpi=300)

tgt = -np.sin(2*np.pi*phase/100)*(90 - AOA)
err = (wing_y - tgt + 180) % 360 - 180
print(f"wing angle RMS error {np.sqrt(np.mean(err**2)):.1f} deg")
for name, v in (('B', joint_B), ('O4', joint_O4), ('P', joint_P), ('C', joint_C), ('D', joint_D)):
    print(f"joint {name}: total rotation {v.max() - v.min():.1f} deg")
print(np.max(P.y_positions) - np.min(P.y_positions))
err = (wing_y - tgt + 180) % 360 - 180
print(f"wing angle RMS error {np.sqrt(np.mean(err**2)):.1f} deg")
for name, v in (('B', joint_B), ('O4', joint_O4), ('P', joint_P), ('C', joint_C), ('D', joint_D)):
    print(f"joint {name}: total rotation {v.max() - v.min():.1f} deg")
print(np.max(P.y_positions) - np.min(P.y_positions))
