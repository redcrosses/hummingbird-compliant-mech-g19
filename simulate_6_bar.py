import mechanism as msm

import mechanism as msm
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

def simulate_6bar(t, omega, **kwargs):
    ## Joints
    joint_names = ['O', 'A', 'B', 'C', 'D', 'E', 'F', 'G']
    joints = dict(zip(joint_names, msm.get_joints(' '.join(joint_names))))

    #G.follow = True
    #F.follow = True

    # Get lengths from kwargs
    L_OA = kwargs.get('L_OA', 4)
    L_OB = kwargs.get('L_OB', 8)
    L_OE = kwargs.get('L_OE', 2)
    L_AD = kwargs.get('L_AD', 3)
    L_BC = kwargs.get('L_BC', 4)
    L_DE = kwargs.get('L_DE', 4)
    L_EF = kwargs.get('L_EF', 4)
    alpha = kwargs.get('alpha', 0.5)
    L_CF = kwargs.get('L_CF', 5)
    L_FG = kwargs.get('L_FG', 5)
    beta = kwargs.get('beta', 0.7)

    # Ternary 1
    # DE - EF - DF - alpha - alpha
    L_DF = np.sqrt(L_DE**2 + L_EF**2 - 2*L_DE*L_EF*np.cos(alpha))
    phi = np.asin(np.sin(alpha)*L_EF/L_DE)

    # Ternary 2
    # CF - FG - CG - beta - gamma
    L_CG = np.sqrt(L_CF**2 + L_FG**2 - 2*L_CF*L_FG*np.cos(beta))
    gamma = np.asin(np.sin(beta)*L_FG/L_CG)

    ## Vectors
    #TODO Consider adding thetas to kwargs
    theta_OA = -.3
    theta_OB = -.3

    vecs = {
        'OA': msm.Vector((joints['O'], joints['A']), r=L_OA, theta=theta_OA, style='ground'),
        'AD': msm.Vector((joints['A'], joints['D']), r=L_AD),
        'DE': msm.Vector((joints['D'], joints['E']), r=L_DE),
        'OE': msm.Vector((joints['O'], joints['E']), r=L_OE),
        
        'OB': msm.Vector((joints['O'], joints['B']), r=L_OB, theta=theta_OB, style='ground'),
        'BC': msm.Vector((joints['B'], joints['C']), r=L_BC),
        'CF': msm.Vector((joints['C'], joints['F']), r=L_CF),
        'EF': msm.Vector((joints['E'], joints['F']), r=L_EF),
        
        'DF': msm.Vector((joints['D'], joints['F']), r=L_DF),
        'CG': msm.Vector((joints['C'], joints['G']), r=L_CG),
        'FG': msm.Vector((joints['F'], joints['G']), r=L_FG)
    }

    # Loop definitions
    # x:
    #   0 - theta_AD
    #   1 - theta_DE
    #   2 - theta_BC
    #   3 - theta_CF
    def loops(x, inp):
        temp = np.zeros((2,2))
        temp[0] = vecs['OA']() + vecs['AD'](x[0]) + vecs['DE'](x[1]) - vecs['OE'](inp)
        temp[1] = vecs['OB']() + vecs['BC'](x[2]) + vecs['CF'](x[3]) - vecs['EF'](x[1] - np.pi + alpha) - vecs['OE'](inp)
        vecs['FG'](x[3] + beta - np.pi)
        vecs['CG'](x[3] - gamma)
        vecs['DF'](x[1] - phi)
        return temp.flatten()

    # initial guess
    # TODO: For the optimiser this initial guess might not be perfect!
    guess = np.deg2rad([90, 195, 90, 165])

    theta_OE = omega * t


## Construct the mechanism
    try:
        mechanism = msm.Mechanism(
            vectors = tuple(vecs.values()),
            origin = joints['O'],
            guess = (guess, np.zeros(4)),
            pos = theta_OE,
            loops = loops
        )

        #TODO Enforce constant length of links (in some cases solver allows links to change length)
        # Possibly through calculating the resulting lengths, comparing them with current ones and returning None
        # if difference larger than specified value (what value?).

        mechanism.iterate()
        return mechanism, vecs, joints
    except Exception as e:
        # print(f"KINEMATICS ERROR: {e}") # for debugging
        return None
