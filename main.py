from mechanism import *
import numpy as np
import matplotlib
matplotlib.use("QtAgg") 
from matplotlib import pyplot as plt
import shapely
from intersect import intersection

def solve(dimensions):
    O1, O2, A, B, C, D, E, F, G, O3, O4 = get_joints('O1 O2 A B C D E F G O3 O4')
    G.follow = True
    D.follow = True
    # c = Vector((O2, B), theta=0, style='ground')

    L21, L22, L3, L41, L42, L51, L52, L6, L71, Lslider, phi1, phi2 = dimensions

    #initial conditions
    theta3, theta4, theta5, theta6, theta_out = np.deg2rad([102.764, 165.512, 141.297, 85.482, 77.8787])
    L_out1, L_out2, L_71 = np.array([18.3097, 18.9186, 14])

    r21 = Vector((O1, A), r=L21)
    r22 = Vector((O1, B), r=L22)
    r3 = Vector((O2, C), r=L3)
    r41 = Vector((A, D), r=L41)
    r42 = Vector((A, C), r=L42)
    r51 = Vector((B, E), r=L51)
    r52 = Vector((E, F), r=L52)
    r6 = Vector((O2, E), r=L6)
    r71 = Vector((D, F))
    # r72 = Vector((F, G))
    rslider = Vector((D, G), r=Lslider)
    r81 = Vector((O2, O4), r=12.66, theta=0, style='ground')
    r82 = Vector((O4, O1), r=2.16, theta=np.pi/2, style='ground')
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

    T = 10 #s
    N = 100
    omega = 2*np.pi/T
    t = np.linspace(0, T, N)

    guess_theta = np.array([theta3, theta4, theta5, theta6, theta_out, L_out1, L_out2, L_71])
    guess_omega = np.zeros_like(guess_theta)
    guess_omega[0] = omega

    # print(t)
    theta2_0 = np.pi + omega*t

    mechanism = Mechanism(vectors=(r21, r22, r3, r41, r42, r51, r52, r6, r71, rslider, r81, r82, rout1, rout2), origin=O1, loops=loops, guess=(guess_theta, guess_omega), pos=theta2_0,) #vel=omega*np.ones_like(t),) #acc=np.zeros_like(t))
    mechanism.iterate()
    return mechanism


L21 = 3 #need to mess around with these a bit to reach desired motion
L22 = 4
L3 = 5.5
L41 = 12.8
L42 = L41
L51 = 13
L52 = 1.41
L6 = 9.3
L71 = 4.4
Lslider = 10.5
phi1 = np.deg2rad(30)
phi2 = np.deg2rad(5)

dimensions = np.array([L21, L22, L3, L41, L42, L51, L52, L6, L71, Lslider, phi1, phi2])

slider_theta = 180/np.pi * np.arctan2((G.y_positions - D.y_positions), (G.x_positions - D.x_positions)) - 90

curve = shapely.LineString(np.stack([G.x_positions, G.y_positions], axis=1))
nodes = shapely.node(curve)

ani, fig, ax = mechanism.get_animation(show_joints=True,)
ax.set_title('8-bar Slider')

fig2, axs = plt.subplots(2, 1)
axs[0].set_title(r'Analysis of $G$')
axs[0].plot(t, G.x_positions, color='red')
axs[0].set_ylabel(r'$G_x$', color='red')
axs[0].tick_params(axis='y', labelcolor='red')
axs[0].set_xlabel(r'Time (s)')

axs2 = axs[0].twinx()
axs2.set_ylabel(r'$\theta_{out}$', color='blue')
axs2.tick_params(axis='y', labelcolor='blue')
axs2.plot(t, slider_theta, color='blue')

try: 
    from collections import Counter
    ends = Counter()
    for seg in nodes.geoms:
        ends[seg.coords[0]] += 1
        ends[seg.coords[-1]] += 1
    crossings = np.array([p for p, c in ends.items() if c > 2])
    ax.scatter(crossings[:,0], crossings[:,1])
    axs[0].hlines(crossings[:,0], xmin=t[0], xmax=t[-1], colors='orange')
    # print(crossings[:,0], G.x_positions)   
    # intersect = intersection(t, np.ones_like(t)*crossings[0,0], t, G.x_positions)
    # print(intersect)
    # intersect = np.array([*intersect])
    # print(intersect)
    intersect_i = np.isclose(crossings[:,0], G.x_positions, atol=0.7)
    print(np.vstack([t[intersect_i], slider_theta[intersect_i]]).T)
    axs[0].scatter(t[intersect_i], G.x_positions[intersect_i])
    axs2.scatter(t[intersect_i], slider_theta[intersect_i])
except:
    print('Intersections ate shit')

axs[1].plot(t, G.y_positions, color='maroon')
axs[1].set_ylabel(r'$G_y$')
axs[1].set_xlabel(r'Time (s)')
# axs[1].scatter(t[crossings[1]], G.y_positions[crossings[1]])

for a in (axs[0], axs[1]):
    a.minorticks_on()
    a.grid(which='both')
fig2.tight_layout()

plt.show()
