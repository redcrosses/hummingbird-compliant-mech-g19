import mechanism as msm
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

O, A, B, C, D, E, F, G = msm.get_joints('O A B C D E F G')

G.follow = True
F.follow = True

L_OA = 4
L_OB = 8
L_OE = 2

L_AD = 3
L_BC = 4

# Ternary 1
L_DE = 4
L_EF = 4
alpha = 0.5
L_DF = np.sqrt(L_DE**2 + L_EF**2 - 2*L_DE*L_EF*np.cos(alpha))
delta = np.asin(np.sin(alpha)*L_EF/L_DE)

# Ternary 2
L_CF = 5
L_FG = 5
beta = 0.7
L_CG = np.sqrt(L_CF**2 + L_FG**2 - 2*L_CF*L_FG*np.cos(beta))
gamma = np.asin(np.sin(beta)*L_FG/L_CG)
print(beta)

# Loop 1
r_OA = msm.Vector((O, A), r=L_OA, theta=-0.3, style='ground')
r_AD = msm.Vector((A, D), r=L_AD)
r_DE = msm.Vector((D, E), r=L_DE)
r_OE = msm.Vector((O, E), r=L_OE)

# Loop 2
r_OB = msm.Vector((O, B), r=L_OB, theta=-0.3 , style='ground')
r_BC = msm.Vector((B, C), r=L_BC)
r_CF = msm.Vector((C, F), r=L_CF)
r_EF = msm.Vector((E, F), r= L_EF)

# stray
r_DF = msm.Vector((D,F), r=L_DF)
r_CG = msm.Vector((C,G), r=L_CG)
r_FG = msm.Vector((F,G), r=L_FG)

# Loop definitions
# x:
#   0 - theta_AD
#   1 - theta_DE
#   2 - theta_BC
#   3 - theta_CF
def loops(x, inp):
    temp = np.zeros((2,2))
    temp[0] = r_OA() + r_AD(x[0]) + r_DE(x[1]) - r_OE(inp)
    temp[1] = r_OB() + r_BC(x[2]) + r_CF(x[3]) - r_EF(x[1] - np.pi + alpha) - r_OE(inp)
    r_FG(x[3] + beta - np.pi)
    r_CG(x[3] - gamma)
    r_DF(x[1] - delta)
    return temp.flatten()

# initial guess
guess = np.deg2rad([90, 195, 90, 165])

T = 10 #s
omega = 2*np.pi/T
t = np.linspace(0, 2*T, 100)
theta_OE = omega * t


## Construct the mechanism
mechanism = msm.Mechanism(
    vectors = (r_OA, r_AD, r_DE, r_OE, r_OB, r_BC, r_CF, r_EF, r_FG, r_CG, r_DF),
    origin = O,
    guess = (guess, np.zeros(4)),
    pos = theta_OE,
    loops = loops
)

mechanism.iterate()
ani, fig, ax = mechanism.get_animation()
plt.show()
ax.set_title('6-bar')

## Optimisation

## Output results


fig, axs = plt.subplots(2, 1)
axs[0].set_title(r'Analysis of $G$')
axs[0].plot(t, G.x_positions, color='maroon')
axs[0].set_ylabel(r'$G_x$')
axs[0].set_xlabel(r'Time (s)')

axs[1].plot(t, G.y_positions, color='maroon')
axs[1].set_ylabel(r'$G_y$')
axs[1].set_xlabel(r'Time (s)')

# Velocities
v0 = axs[0].twinx()
v0.plot(t, G.x_velocities)

v1 = axs[1].twinx()
v1.plot(t, G.y_velocities)
for a in (axs[0], axs[1]):
    a.minorticks_on()
    a.grid(which='both')
fig.tight_layout()

# Link angles at joints
theta_OE = r_OE.pos.thetas
theta_AD = r_AD.pos.thetas
theta_BC = r_BC.pos.thetas
theta_BCF = r_CF.pos.thetas - r_BC.pos.thetas
theta_DFC = r_DF.pos.thetas - r_CF.pos.thetas
theta_ADE = r_AD.pos.thetas - (r_DE.pos.thetas - np.pi)


theta_AD = np.rad2deg(theta_AD)
theta_BC = np.rad2deg(theta_BC)
theta_BCF = np.rad2deg(theta_BCF)
theta_DFC = np.rad2deg(theta_DFC)
theta_ADE = np.rad2deg(theta_ADE)

fig_angles, ax_angles = plt.subplots()
ax_angles.plot(t, theta_AD, label="$theta_{AD}$")
ax_angles.plot(t, theta_BC, label="$theta_{BC}$")
ax_angles.plot(t, theta_BCF, label="$theta_{BCF}$")
ax_angles.plot(t, theta_DFC, label="$theta_{DFC}$")
ax_angles.plot(t, theta_ADE, label="$theta_{ADE}$")

ax_angles.legend()

# angle changes
def varia(angle):
    return np.max(angle) - np.min(angle)

d_theta_AD = varia(theta_AD)
d_theta_BC = varia(theta_BC)
d_theta_BCF = varia(theta_BCF)
d_theta_DFC = varia(theta_DFC)
d_theta_ADE = varia(theta_ADE)

print(d_theta_AD, d_theta_ADE, d_theta_BC, d_theta_BCF, d_theta_DFC)



plt.show()