from mechanism import *
import numpy as np
import matplotlib
matplotlib.use("TkAgg") 
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

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
ani, fig, ax = mechanism.get_animation()

ax.set_title('Dual-Actuator Compliant Mechanism')

# ---- Slotted link L8 (k): angle and angular speed ----
# Angle of L8 taken from the solved joint positions (F -> H), unwrapped so it is continuous
th_k = np.unwrap(np.arctan2(np.array(H.y_positions) - np.array(F.y_positions),
                            np.array(H.x_positions) - np.array(F.x_positions)))
dth_dinp = np.gradient(th_k, t2)   # rad of L8 per rad of actuator input

# Actuator frequency (Hz): input angle = 2*pi*f*t, the second actuator follows at -inp
# Actuator frequency (Hz): input angle = 2*pi*f*t, the second actuator follows at -inp
f0 = 1.0
frames_per_rev = len(t2) / (t2[-1] - t2[0]) * 2 * np.pi

time = t2 / (2 * np.pi * f0)

# Linear velocity of point H on bar L8
vx_H = np.gradient(H.x_positions, time)
vy_H = np.gradient(H.y_positions, time)
v_H = np.sqrt(vx_H**2 + vy_H**2)

fig2, (ax_ang, ax_spd) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

line_ang, = ax_ang.plot(time, np.rad2deg(th_k))
line_spd, = ax_spd.plot(time, v_H)

ax_ang.set_ylabel('L8 angle [deg]')
ax_spd.set_ylabel('L8 linear speed [units/s]')
ax_spd.set_xlabel('Time [s]')
ax_ang.grid(True)
ax_spd.grid(True)


plt.show()

print(np.max(H.y_positions))
print(np.min(H.y_positions))
print(np.max(H.y_positions) - np.min(H.y_positions))