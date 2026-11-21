import os
from mechanism import *
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import pandas as pd
from scipy.optimize import fsolve
from scipy.interpolate import CubicSpline

from ansys.mapdl.core import launch_mapdl

O2, O4, A, B, C, D, P = get_joints('O2 O4 A B C D P')
P.follow = True
params = pd.read_csv('final_design/six_bar_loop_parameters.csv')
lengths = params[['Value']].to_numpy().flatten()

fabr_state = pd.read_csv('final_design/fab_angles.csv')[['Value']].to_numpy().flatten() * np.pi/180

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
t = np.linspace(0, T, N, endpoint=True)
theta2 = fabr_state[0] - omega*t
theta2_deg = theta2*180/np.pi

# initial guess; roughly
guess = np.array([fabr_state[1], fabr_state[4], fabr_state[6], fabr_state[7]])

mechanism = Mechanism(vectors=(r1, r2, r3, r5, r6, r7, r8, r9), origin=O2, loops=loops, guess=(guess, ), pos=theta2,) #vel=omega*np.ones_like(t),) #acc=np.zeros_like(t))
mechanism.iterate()
ani, fig1, ax1 = mechanism.get_animation(show_joints=True)

# make room for the slider and add it to the animation figure
fig1.subplots_adjust(bottom=0.18)
ax_s = fig1.add_axes([0.15, 0.05, 0.7, 0.04])
slider = Slider(ax_s, r'$\theta_2$ [deg]', theta2_deg.min(), theta2_deg.max(),
                valinit=theta2_deg[0])
ax_s.invert_xaxis()

def on_slide(val):
    i = int(np.argmin(np.abs(theta2_deg - val)))   # nearest solved frame
    ani.pause(); ani.going = False                 # stop playback while scrubbing
    ani.i = i
    ani.func(i)                                    # redraws links, joint labels and the stamp
    fig1.canvas.draw_idle()

slider.on_changed(on_slide)

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
th4 = th5 - (fabr_state[3] - fabr_state[2])
th7 = th6 - beta

state = np.column_stack([theta2, th3, th4, th5, th6, th7, th8, th9])
deflections = state - np.array([fabr_state]*t.shape[0])
# print(np.rad2deg(deflections[0,:]))

#phiB, phi6, phi7, phiC, phiP
phis = np.array([deflections[:,4] - deflections[:,2], #phiB
                 deflections[:,4], #phi6
                 deflections[:,5] - deflections[:,6], #phiD
                 deflections[:,7] - deflections[:,6], #phiC
                 deflections[:,2] - deflections[:,7]]).T #phiP

#TODO try to set up a fully plastic version 
E = 185e3 #[N/mm^2] steel 1.4310, from datasheet, h+s
yield_str = 1000 #[N/mm^2] from virgamet 
endurance_lim = 550 #[N/mm^2] from datasheet, h+s
b = 10 #[mm] #fixed for all metal flexure options
thicknesses = 1*np.array([0.1, 0.1, 0.1, 0.1, 0.1]) #[mm], sprB, spr6, sprD, sprC, sprP. options: 0.1, 0.2, 0.3, 0.4
l = np.array([4.444*2, 4.444*6, 12.33, 2.27*4, 2.27*4,]) #[mm], lengths of the metal flexure(s)

I = b * thicknesses**3 / 12 #with b the depth of the flexure and t the thickness of the flexure

#torsional springs corresponding to: phiB, phi6, phiD, phiC, phiP; 
Ks_PRBM = E * np.array([I/l]*t.shape[0])

# velocities and accelerations by finite differences
phis_d  = np.gradient(phis, t, axis=0)
omega_inp = np.gradient(theta2, t, axis=0)

#torque-deflection
torque = np.sum(Ks_PRBM * phis * phis_d, axis=1) / omega_inp
theta_torque_zero = CubicSpline(np.rad2deg(theta2)[::-1], torque[::-1], bc_type='not-a-knot').roots(extrapolate=False)
print(f'Zero-torque occurs at: {theta_torque_zero} [deg]')

ax1.lines[-1].set_label('PRBM')
fig2, (ax2, ax3, ax4) = plt.subplots(3, 1, sharex='col')

#get ansys results TODO run ansys with the provided parameters from this script
try:
    imp = np.loadtxt('final_design/results_post26(v2).txt', skiprows=8)
    theta2_ansys_deg = imp[:,2]
    torque_ansys = imp[:,1]
    ax1.plot(imp[:,3], imp[:,4], c='green', label='ANSYS FEM results')
    ax2.plot(theta2_ansys_deg, torque_ansys, c='cadetblue', label='ANSYS FEM results')
    ax4.plot(imp[:,2], imp[:,6])
except:
    print('Could not import ANSYS APDL results :(')

ax2.plot(theta2_deg, torque, c='maroon', label='PRBM')
for thet in theta_torque_zero:
    ax2.axvline(thet, linestyle='--', c='red')
    ax3.axvline(thet, linestyle='--', c='red')
ax2.set_title("Torque-Deflection Relationship")
ax2.set_xlim(max(theta2_deg), min(theta2_deg))
ax2.set_xlabel(r'$\theta_{inp}$ [deg]'); ax2.set_ylabel(r'Torque [Nmm]')
ax2.grid(which = 'both')
ax2.minorticks_on()
ax1.legend()
ax2.legend()
print(f'Min:{np.min(torque):.3f}[Nmm], Max:{np.max(torque):.3f}[Nmm]')

ax3.plot(theta2_deg, np.rad2deg(phis), label=[r'$\phi_B$', r'$\phi_6$', r'$\phi_7$', r'$\phi_C$', r'$\phi_P$'])
ax3.legend()
ax3.set_xlim(max(theta2_deg), min(theta2_deg))
ax3.set_xlabel(r'$\theta_{inp}$ [deg]'); ax3.set_ylabel(r'Deflection [deg]')
ax3.grid(which = 'both')
ax3.minorticks_on()

ax4.set_title('Maximum Stresses over Rotation [MPa]')
ax4.axhline(endurance_lim, linestyle='--', c='orange', label='Endurance Limit')
ax4.axhline(yield_str, linestyle='--', c='red', label='Tensile Strength')
ax4.set_xlabel(r'$\theta_{inp}$ [deg]'); ax4.set_ylabel(r'Max von Mises Stress [MPa]')
ax4.set_xlim(max(theta2_deg), min(theta2_deg))
ax4.grid(which = 'both')
ax4.minorticks_on()
ax4.legend()

fig2.tight_layout()

np.savetxt('final_design/PRBM_torque_deflection.csv', np.column_stack([theta2_deg, torque, np.rad2deg(phis)]), delimiter=',', header='theta2_deg, torque, phiB, phi6, phi7, phiC, phiP', comments='')

plt.show()