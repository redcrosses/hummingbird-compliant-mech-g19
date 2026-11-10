import mechanism as msm
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

O, A, B, C, D, E, F, G = msm.get_joints('O A B C D E F G')

E.follow = True

# Base 4-bar
L_OC = 11
L_OA = 1.41
L_AB = 8.9
L_BC = 4

# ternary
L_AD = 4
alpha = 0
L_BD = np.sqrt(L_AD**2 + L_AB**2 - L_AD*L_AB*np.cos(alpha))

# dyad DFG
L_OG = 2
L_DF = 4
L_FG = 2.5

# dyad BEF
L_BE = 3.5
L_EF = 5.5

# Loop 1 - base 4-bar
r_OA = msm.Vector((O, A), r=L_OA)
r_AB = msm.Vector((A, B), r=L_AB)
r_BC = msm.Vector((B, C), r=L_BC)
r_OC = msm.Vector((O, C), r=L_OC, theta=0 , style='ground')

# ternary
r_AD = msm.Vector((A,D), r=L_AD)
r_BD = msm.Vector((B,D), r=L_BD)

# dyad DFG
r_DF = msm.Vector((D, F), r=L_DF)
r_FG = msm.Vector((F, G), r=L_FG)
r_OG = msm.Vector((O, G), r= L_OG, theta=np.pi/2*0.6 , style='ground')

# dyad BEF
r_BE = msm.Vector((B, E), r=L_BE)
r_EF = msm.Vector((E, F), r=L_EF)

# Loop definitions
# x:
#   0 - theta_AB
#   1 - theta_BC
#   2 - theta_DF
#   3 - theta_FG
#   4 - theta_BE
#   5 - theta_EF
def loops(x, inp):
    temp = np.zeros((3,2))
    temp[0] = r_OA(inp) + r_AB(x[0]) + r_BC(x[1]) - r_OC()
    temp[1] = r_OA(inp) + r_AD(x[0] - alpha) + r_DF(x[2]) + r_FG(x[3]) - r_OG()
    temp[2] = r_BE(x[4]) + r_EF(x[5]) + r_FG(x[3]) - r_OG() + r_OA(inp) + r_AB(x[0])
    return temp.flatten()

# initial guess
guess = np.deg2rad([0,0,0,0,0,0])

T = 5 #s
omega = 2*np.pi/T
t = np.linspace(0, 4*T, 100)
theta_OA = omega * t + 1.5*np.pi


## Construct the mechanism
mechanism = msm.Mechanism(
    vectors = (r_OA, r_AB, r_BC, r_OC, r_AD, r_DF, r_FG, r_OG, r_BE, r_EF),
    origin = O,
    guess = (guess, np.zeros(6)),
    pos = theta_OA,
    vel=omega*np.ones_like(t),
    loops = loops
      
)

mechanism.iterate()
ani, fig, ax = mechanism.get_animation()
plt.show()

## Optimisation

## Output results
