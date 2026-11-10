from simulate_6_bar import simulate_6bar
import mechanism as msm
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize

param_names = ['L_OA', 'L_OB', 'L_OE', 'L_AD', 'L_BC', 'L_DE', 'L_EF', 'alpha', 'L_CF', 'L_FG', 'beta']


def target_angle(omega, t):
    return -np.sin(omega*t)*64.5 + 28

def target_path(omega, t):
    # TODO find a way to fit it nicer to the traced path
    A = 5 #10.0
    B = 3 #1.5
    x_target = A/2*np.sin(omega*t) + 7
    y_target = B/2*np.sin(2*omega*t) + 5
    return x_target, y_target

ev_count = 0
def objective(params, omega, t):
    kwargs = dict(zip(param_names, params))

    res = simulate_6bar(t, omega, **kwargs)

    global ev_count
    ev_count = ev_count + 1

    if res is None:
        # high penalty for non-assemblabe linkages
        return 1e6

    mechanism, vecs, joints = res

    x_target, y_target = target_path(omega, t)
    path_error = np.mean(
        (joints['G'].x_positions - x_target)**2 +
        (joints['G'].y_positions - y_target)**2
    )

    target_angles = np.deg2rad(target_angle(omega, t))
    angle_error = np.mean((vecs['FG'].pos.thetas - target_angles)**2)

    # angular displacement penalty
    def angle_range(angle):
        return np.max(angle) - np.min(angle)

    d_theta_AD = angle_range(vecs['AD'].pos.thetas)
    d_theta_BC = angle_range(vecs['BC'].pos.thetas)
    d_theta_BCF = angle_range(vecs['CF'].pos.thetas - vecs['BC'].pos.thetas)
    d_theta_DFC = angle_range(vecs['DF'].pos.thetas - vecs['CF'].pos.thetas)
    d_theta_ADE = angle_range(vecs['AD'].pos.thetas - (vecs['DF'].pos.thetas + np.pi))

    angle_var = d_theta_AD + d_theta_BC + d_theta_BCF + d_theta_DFC + d_theta_ADE

    # Weights
    W_path = 1 
    W_angle = 5
    W_angle_var = 0

    cost = W_path * path_error + W_angle * angle_error + W_angle_var * angle_var

    if ev_count %50 == 0:
        print("Current cost:", cost)

    return cost

if __name__ == "__main__":
    T = 10 #s
    omega = 2*np.pi/T

    t = np.linspace(0, 2*T, 100)

    x0 = [4, 8, 2, 3, 4, 4, 4, 0.5, 5, 5, 0.7]

    bounds = [
        (1,200), (1,20), (1,20), (1,20), (1,20), (1,20), (1,20), (0, np.pi), (1,20), (1,20), (0,np.pi/2)
    ]

    print("Optimiser start")

    result = minimize(
        objective,
        x0,
        args = (t,omega),
        method= 'Nelder-Mead',
        bounds=bounds,
        options={'disp': True, 'maxiter': 500}
    )

    print("\nOptimized Parameters:")
    for name, val in zip(param_names, result.x):
        print(f"  {name}: {val:.3f}")
    
    # Re-run simulation with optimized parameters for visualization
    opt_kwargs = dict(zip(param_names, result.x))
    opt_res = simulate_6bar(t, omega, **opt_kwargs)
    
    if opt_res:
        opt_mech, opt_vecs, opt_joints = opt_res
        
        fig, ax = plt.subplots(figsize=(8, 6))
        xt, yt = target_path(omega, t)
        ax.plot(xt, yt, 'k--', label='Target Path')
        ax.plot(opt_joints['G'].x_positions, opt_joints['G'].y_positions, 'r-', label='Optimized Path G')
        
        ax.set_aspect('equal')
        ax.legend()
        ax.set_title('Mechanism G-Node Path Tracking')
        ax.grid(True)
        plt.show()

        ani, fig, ax = opt_mech.get_animation()
        plt.show()
        ax.set_title('6-bar')

def preview_path(omega, t):
    fig, ax = plt.subplots(figsize=(8, 6))
    xt, yt = target_path(omega, t)
    ax.plot(xt, yt, 'k--', label='Target Path')
    plt.show()
    exit()
