
#4-bar mechanism with coupler point P
from mechanism import *
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

# I used Z for lengths instead of L, because the paper I used for the optimisation also used Z
Z1, Z2, Z3, Z4, Z5, Z6, th1 = 98.542, 46.651, 58.395, 94.440, 88.002, 95.110, 2.3124

O2, O4, A, B, P = get_joints('O2 O4 A B P')
B.follow = True
P.follow = True

r1 = Vector((O2, O4), r=Z1, theta=th1, style='ground')
r2 = Vector((O2, A), r=Z2)
r3 = Vector((A, B), r=Z3)
r4 = Vector((B, O4), r=Z4)
r5 = Vector((A, P), r=Z5)
r6 = Vector((P, B), r=Z6)

# 0: theta3, 1: theta4, 2: theta5, 3: theta6
def loops(t, inp):
    temp = np.zeros((2, 2))
    temp[0] = r2(inp) + r3(t[0]) + r4(t[1]) - r1() #vector loop 1
    temp[1] = r5(t[2]) + r6(t[3]) - r3(t[0]) #vector loop triangle
    return temp.flatten()


guess = np.deg2rad([184.1, 125.6, 105.9, 249.0])

T = 5
omega2 = 2*np.pi/T
t = np.linspace(0, 3*T, 300)
t2 = omega2*t

mechanism = Mechanism(vectors=(r1, r2, r3, r4, r5, r6), origin=O2, loops=loops, pos=t2, vel=omega2*np.ones_like(t), guess=(guess, np.zeros(4)))
mechanism.iterate()
ani, fig, ax = mechanism.get_animation()

ax.set_title('4-bar')

# Position & velocity analysis
fig2, axs = plt.subplots(2, 1)
axs[0].set_title(r'Analysis of $P$')

axs[0].plot(t2, P.x_positions, color='maroon')
axs[0].set_ylabel(r'$P_x$ (mm)')
v0 = axs[0].twinx()
v0.plot(t2, P.x_velocities, color='navy', linestyle='--')
v0.set_ylabel(r'$v_{P,x}$ (mm/s)')

axs[1].plot(t2, P.y_positions, color='maroon')
axs[1].set_ylabel(r'$P_y$')
v1 = axs[1].twinx()
v1.plot(t2, P.y_velocities, color='navy', linestyle='--')
v1.set_ylabel(r'$v_{P,y}$ (mm/s)')

for a in (axs[0], axs[1]):
    a.minorticks_on()
    a.grid(which='both')
fig2.tight_layout()

# flexures angles plotted against input angle
t3 = t2[t2 <= 2*np.pi]
n = len(t3)

th2 = t3
theta2_deg = np.rad2deg(t3)
th3 = np.unwrap(r3.pos.thetas[:n])
th4 = np.unwrap(r4.pos.thetas[:n])

joint_O2 = np.rad2deg(th2 - th1)
joint_A  = np.rad2deg(np.unwrap(th3 - th2))
joint_B  = np.rad2deg(np.unwrap(th4 - th3))
joint_O4 = np.rad2deg(th4 - th1)

fig3, ax3 = plt.subplots()
# ax3.plot(theta2_deg, joint_A, label=r'joint A') # A is the crank, so will always go 360 degrees
ax3.plot(theta2_deg, joint_B, label=r'joint B')
ax3.plot(theta2_deg, joint_O4, label=r'joint O4')
ax3.set_xlim(0, 360)
ax3.set_xticks(np.arange(0, 361, 60))
ax3.set_title(r'Joint angles')
ax3.set_xlabel(r'Input angle $\theta_2$ $\in [0^\circ, 360^\circ]$')
ax3.set_ylabel(r'Joint angles')
ax3.legend(); ax3.minorticks_on(); ax3.grid(which='both')
fig3.tight_layout()
plt.show()

# ani.save('../animations/crankslider.mp4', dpi=300)

print(np.max(P.y_positions))
print(np.min(P.y_positions))
print(np.max(P.y_positions) - np.min(P.y_positions))
