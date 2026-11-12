#6-bar mechanism: four-bar O2-A-B-O4 with coupler point P, wing P-C, and link 6 C-D
#driven by a point D on the crank. The crank turns backwards (clockwise), so the
#downstroke runs towards -x.
from mechanism import *
import numpy as np
import matplotlib
matplotlib.use("QtAgg")            # animation in its own window
from matplotlib import pyplot as plt

# Result of six_bar_wing.py (lengths in mm, angles in rad), placed coordinates with O2 at (0, 0)
# Four-bar (step 1). th1 is the placed ground angle O2->O4.
Z1, Z2, Z3, Z4, Z5, Z6, th1 = 100.301, 44.179, 118.632, 80.475, 108.824, 114.984, -0.1001
psi = -2.0102            # crank angle at precision point 0
# Wing dyad (step 2), link 6 on the crank
L_D, beta = 42.765, 0.3206       # D on the crank: distance from O2, angle from the crank O2->A
L_PC = 46.444                    # wing P-C (the wing chord runs along C -> P)
L_DC = 133.331                   # link 6, D -> C

AOA = 45                         # target angle of attack on both strokes (deg)

O2, O4, A, B, P, C, D = get_joints('O2 O4 A B P C D')
P.follow = True
C.follow = True

r1 = Vector((O2, O4), r=Z1, theta=th1, style='ground')
r2 = Vector((O2, A), r=Z2)        # crank
r10 = Vector((O2, D), r=L_D)      # second point on the crank
r3 = Vector((A, B), r=Z3)
r4 = Vector((B, O4), r=Z4)
r5 = Vector((A, P), r=Z5)
r6 = Vector((P, B), r=Z6)
r7 = Vector((P, C), r=L_PC)       # wing
r8 = Vector((D, C), r=L_DC)       # link 6

# 0: theta3, 1: theta4, 2: theta5, 3: theta6, 4: theta7 (wing), 5: theta8 (link 6)
def loops(t, inp):
    temp = np.zeros((3, 2))
    temp[0] = r2(inp) + r3(t[0]) + r4(t[1]) - r1()                      # four-bar loop
    temp[1] = r5(t[2]) + r6(t[3]) - r3(t[0])                            # coupler triangle
    temp[2] = r10(inp + beta) + r8(t[5]) - r2(inp) - r5(t[2]) - r7(t[4])  # wing loop O2-D-C-P-A-O2
    return temp.flatten()


guess = np.deg2rad([53.16, 306.15, 352.61, 108.66, 311.66, 339.72])   # angles at the first input angle

T = 5
omega2 = 2*np.pi/T
t = np.linspace(0, 3*T, 300)
t2 = psi - omega2*t      # crank turns backwards: downstroke towards -x

mechanism = Mechanism(vectors=(r1, r2, r10, r3, r4, r5, r6, r7, r8), origin=O2, loops=loops,
                      pos=t2, vel=-omega2*np.ones_like(t), guess=(guess, np.zeros(6)))
mechanism.iterate()
ani, fig, ax = mechanism.get_animation()

ax.set_title('6-bar')

# Position & velocity analysis
fig2, axs = plt.subplots(2, 1)
axs[0].set_title(r'Analysis of $P$')

axs[0].plot(t, P.x_positions, color='maroon')
axs[0].set_ylabel(r'$P_x$ (mm)')
v0 = axs[0].twinx()
v0.plot(t, P.x_velocities, color='navy', linestyle='--')
v0.set_ylabel(r'$v_{P,x}$ (mm/s)')

axs[1].plot(t, P.y_positions, color='maroon')
axs[1].set_ylabel(r'$P_y$ (mm)')
v1 = axs[1].twinx()
v1.plot(t, P.y_velocities, color='navy', linestyle='--')
v1.set_ylabel(r'$v_{P,y}$ (mm/s)')
axs[1].set_xlabel('Time (s)')

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
joint_D  = np.rad2deg(th8 - (th2 + beta))             # crank - link 6 (turns fully: bearing)

fig3, ax3 = plt.subplots()
for name, v in (('joint B', joint_B), ('joint O4', joint_O4), ('joint P', joint_P),
                ('joint C', joint_C)):
    ax3.plot(phase[order], (v - v[i0])[order], label=name)
ax3.axvspan(0, 50, color='tab:blue', alpha=0.08)
ax3.set_xlim(0, 100)
ax3.set_xticks(np.arange(0, 101, 10))
ax3.set_title(r'Joint rotations')
ax3.set_xlabel('% of flap cycle (0-50 downstroke, 50-100 upstroke)')
ax3.set_ylabel(r'Joint rotation from start of downstroke (deg)')
ax3.legend(); ax3.minorticks_on(); ax3.grid(which='both')
fig3.tight_layout()

# wing angle: chord C -> P, measured from the +y-axis, positive towards +x (clockwise)
wing_x = np.rad2deg(th7) + 180                        # C -> P is P -> C turned by 180 deg
wing_y = (90 - wing_x + 180) % 360 - 180

# target wing angle, x = % of the flap cycle
x_cycle = np.linspace(0, 100, 201)
target = -np.sin(2*np.pi*x_cycle/100)*(90 - AOA)

fig5, ax5 = plt.subplots()
ax5.axvspan(0, 50, color='tab:blue', alpha=0.08, label='Downstroke')
ax5.plot(phase[order], wing_y[order], color='maroon', label=r'Wing (C$\rightarrow$P)')
ax5.plot(x_cycle, target, color='navy', linestyle='--', label='target')
ax5.set_xlim(0, 100)
ax5.set_xticks(np.arange(0, 101, 10))
ax5.set_title(r'Wing angle relative to the y-axis')
ax5.set_xlabel('% of flap cycle (0-50 downstroke, 50-100 upstroke)')
ax5.set_ylabel('Angle from y-axis (deg)')
ax5.legend(); ax5.minorticks_on(); ax5.grid(which='both')
fig5.tight_layout()

plt.show()

# ani.save('../animations/sixbar.mp4', dpi=300)

tgt = -np.sin(2*np.pi*phase/100)*(90 - AOA)
err = (wing_y - tgt + 180) % 360 - 180
print(f"wing angle RMS error {np.sqrt(np.mean(err**2)):.1f} deg")
for name, v in (('B', joint_B), ('O4', joint_O4), ('P', joint_P), ('C', joint_C), ('D', joint_D)):
    print(f"joint {name}: total rotation {v.max() - v.min():.1f} deg")
print(np.max(P.y_positions) - np.min(P.y_positions))