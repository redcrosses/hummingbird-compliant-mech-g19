from mechanism import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import fsolve

O2, O4, A, B, C, D, P = get_joints('O2 O4 A B C D P')
P.follow = True
params = pd.read_csv('final_design/six_bar_loop_parameters.csv')
lengths = params[['Value']].to_numpy().flatten()

#TODO adjust these according to PRBM
gammas = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
L1, L2, L3, L4, L5, L6, L7, L8, L9, theta1, psi, alpha, beta, gamma = lengths

L1, L2, L3, L4, L5, L6, L7, L8, L9 = gammas * np.array([L1, L2, L3, L4, L5, L6, L7, L8, L9])
theta1, psi, alpha, beta, gamma = np.deg2rad([theta1, psi, alpha, beta, gamma])

r1 = Vector((O2, O4), r=L1, theta=theta1, style='ground')
r2 = Vector((O2, A), r=L2) #input
r3 = Vector((A, B), r=L3)
r4 = Vector((P, B), r=L4)
r5 = Vector((P, A), r=L5)
r6 = Vector((O4, B), r=L6)
r7 = Vector((O4, D), r=L7)
r8 = Vector((C, D), r=L8)
r9 = Vector((C, P), r=L9)

#x: theta3 theta6 theta8 theta9
def loops(x, inp):
    temp = np.zeros((2, 2))
    temp[0] = r2(inp) + r3(x[0]) - r1() - r6(x[1])
    temp[1] = r2(inp) + r8(x[2]) - r5(x[0] - alpha + np.pi) - r9(x[3]) - r7(x[1] - beta) - r1()
    return temp.flatten()

T = 1
N = 100
omega = 2 * np.pi / T
t = np.linspace(0, T, N, endpoint=False)
theta2 = psi - omega*t

# initial guess; roughly
guess = np.deg2rad([20, 120, 60, 180])

mechanism = Mechanism(vectors=(r1, r2, r3, r5, r6, r7, r8, r9), origin=O2, loops=loops, guess=(guess, ), pos=theta2,) #vel=omega*np.ones_like(t),) #acc=np.zeros_like(t))
mechanism.iterate()
ani, fig1, ax1 = mechanism.get_animation(show_joints=True)

sol = np.zeros((N, 4))
x0 = guess
for i in range(N):
    x, info, ier, msg = fsolve(loops, x0, args=(theta2[i],), full_output=True)
    if ier != 1:
        print(f"step {i} did not converge: {msg}")
    sol[i] = x
    x0 = x

th3, th6, th8, th9 = sol.T
th5 = th3 - alpha + np.pi
th4 = th5 + gamma
th7 = th6 - beta

#TODO fill in values
fabr_state = np.deg2rad([0,0,0,0,0,0,0,0])
state = np.column_stack([theta2, th3, th4, th5, th6, th7, th8, th9])
deflections = state - np.array([fabr_state]*t.shape[0])

#phiB, phi6, phi7, phiC, phiP
phis = np.array([deflections[:,4] - deflections[:,2],
                 deflections[:,4],
                 deflections[:,5],
                 deflections[:,7] - deflections[:,6],
                 deflections[:,2]]).T

E = 1
I = 1
#torsional springs corresponding to: phiB, phi6, phi7, phiC, phiP; TODO fill in values
Ks_PRBM = np.array([[1, 1, 1, 1, 1]]*t.shape[0])

# velocities and accelerations by finite differences
phis_d  = np.gradient(phis, t, axis=0)
omega_inp = np.gradient(theta2, t, axis=0)

#torque-deflection
torque = np.sum(Ks_PRBM * phis * phis_d, axis=1) / omega_inp

fig2, ax2 = plt.subplots()
theta2_plt = np.rad2deg(theta2)
ax2.plot(theta2_plt, torque)
ax2.set_title("Torque-Deflection Relationship")
ax2.set_xlim(max(theta2_plt), min(theta2_plt))
ax2.set_xlabel(r'$\theta_{inp}$ [deg]'); ax2.set_ylabel(r'Torque [Nmm]')
ax2.grid(which = 'both')
ax2.minorticks_on()
print(f'Min:{np.min(torque):.3f}[Nmm], Max:{np.max(torque):.3f}[Nmm]')

plt.show()