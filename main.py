from mechanism import *
import numpy as np
import matplotlib
matplotlib.use("QtAgg") 
from matplotlib import pyplot as plt

O1, O2, A, B, C, D, E, F, G, O3, O4 = get_joints('O1 O2 A B C D E F G O3 O4')
G.follow = True
D.follow = True
# c = Vector((O2, B), theta=0, style='ground')

L21 = 5
L22 = 7
L23 = 3.657
L3 = 8
L41 = 22.157
L42 = L41
L43 = 1.933
L51 = 30
L52 = 5
L6 = 20.0665
L71 = 14
Lslider = 19
phi1 = np.deg2rad(30)
phi2 = np.deg2rad(5)

r21 = Vector((O1, A), r=L21)
r22 = Vector((O1, B), r=L22)
# r23 = Vector((A, B), r=L23) #to constrain the rigid body

r3 = Vector((O2, C), r=L3)

r41 = Vector((A, D), r=L41)
r42 = Vector((A, C), r=L42)
# r43 = Vector((C, D), r=L43) #to constrain the rigid body

r51 = Vector((B, E), r=L51)
r52 = Vector((E, F), r=L52)
r6 = Vector((O2, E), r=L6)

r71 = Vector((D, F))
# r72 = Vector((F, G))
rslider = Vector((D, G), r=Lslider)

r81 = Vector((O2, O4), r=20, theta=0, style='ground')
r82 = Vector((O4, O1), r=5, theta=np.pi/2, style='ground')
rout1 = Vector((G, O3), theta=0, style='ground')
rout2 = Vector((O3, O1), theta=3*np.pi/2, style='ground')


#x - 0:theta3. 1:theta4, 2:theta5, 3:theta6, 4:theta_out, 5:L_out1, 6: L_out2, 7: L_71
def loops(x, inp):
    temp = np.zeros((4, 2))
    temp[0] = r21(inp) + r42(x[1]+phi2) - r3(x[0]) + r81() + r82()
    temp[1] = r21(inp) + r41(x[1]) + rslider(x[4]) + rout1(x[5]) + rout2(x[6])
    temp[2] = r22(inp+phi1) + r51(x[2]) - r6(x[3]) + r81() + r82()
    temp[3] = r22(inp+phi1) + r51(x[2]) + r52(x[2]) + (-r71(x[7], x[4]) + rslider(x[4])) + rout1(x[5]) + rout2(x[6])
    return temp.flatten()


guess = np.concatenate((np.deg2rad([152.822, 178.483, 143.754, 110.922, 102.822]), np.array([31.3663, 19.113, 14])))

T = 5 #s
omega = 2*np.pi/T
t = np.linspace(0, 3*T, 100)
theta2_0 = 0 + omega*t


mechanism = Mechanism(vectors=(r21, r22, r3, r41, r42, r51, r52, r6, r71, rslider, r81, r82, rout1, rout2), origin=O1, loops=loops, guess=(guess, ), pos=theta2_0) #, vel=omega*np.ones_like(t), acc=np.zeros_like(t))
mechanism.iterate()
ani, fig, ax = mechanism.get_animation()

ax.set_title('8-bar Slider')

fig2, axs = plt.subplots(2, 1)
axs[0].set_title(r'Analysis of $G$')
axs[0].plot(t, G.x_positions, color='maroon')
axs[0].set_ylabel(r'$G_x$')
axs[0].set_xlabel(r'Time (s)')

axs[1].plot(t, G.y_positions, color='maroon')
axs[1].set_ylabel(r'$G_y$')
axs[1].set_xlabel(r'Time (s)')

for a in (axs[0], axs[1]):
    a.minorticks_on()
    a.grid(which='both')
fig2.tight_layout()

plt.show()

# ani.save('../animations/crankslider.mp4', dpi=300)