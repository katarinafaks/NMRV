import numpy as np
import math
import matplotlib.pyplot as plt
from ex4_utils import kalman_step

def get_matrices(model, dt, q):
    """Returna A (Phi), C (H), and Q za dan model."""
    if model == 'RW':
        A = np.eye(2, dtype=np.float32)
        C = np.eye(2, dtype=np.float32)
        Q = q * np.eye(2, dtype=np.float32) * dt
        return A, C, Q
        
    elif model == 'NCV':
        A = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=np.float32)
        C = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]], dtype=np.float32)
        Q = q * np.array([[(dt**3)/3, 0, (dt**2)/2, 0],
                          [0, (dt**3)/3, 0, (dt**2)/2],
                          [(dt**2)/2, 0, dt, 0],
                          [0, (dt**2)/2, 0, dt]], dtype=np.float32)
        return A, C, Q
        
    elif model == 'NCA':
        A = np.array([[1, 0, dt, 0, (dt**2)/2, 0],
                      [0, 1, 0, dt, 0, (dt**2)/2],
                      [0, 0, 1, 0, dt, 0],
                      [0, 0, 0, 1, 0, dt],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1]], dtype=np.float32)
        C = np.array([[1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0]], dtype=np.float32)
        Q = q * np.array([[(dt**5)/20, 0, (dt**4)/8, 0, (dt**3)/6, 0],
                          [0, (dt**5)/20, 0, (dt**4)/8, 0, (dt**3)/6],
                          [(dt**4)/8, 0, (dt**3)/3, 0, (dt**2)/2, 0],
                          [0, (dt**4)/8, 0, (dt**3)/3, 0, (dt**2)/2],
                          [(dt**3)/6, 0, (dt**2)/2, 0, dt, 0],
                          [0, (dt**3)/6, 0, (dt**2)/2, 0, dt]], dtype=np.float32)
        return A, C, Q

def generate_trajectory(type='spiral'):
    N = 40
    if type == 'spiral':
        v = np.linspace(5 * math.pi, 0, N)
        x = np.cos(v) * v
        y = np.sin(v) * v
    elif type == 'rectangle':
        x = np.concatenate([np.linspace(0, 10, 10), np.ones(10)*10, np.linspace(10, 0, 10), np.zeros(10)])
        y = np.concatenate([np.zeros(10), np.linspace(0, 10, 10), np.ones(10)*10, np.linspace(10, 0, 10)])
    elif type == 'sine':
        x = np.linspace(0, 20, N)
        y = 5 * np.sin(x)
    return x, y

def run_kalman(x, y, model, q, r, dt=1.0):
    
    A, C, Q = get_matrices(model, dt, q)
    R = r * np.eye(2, dtype=np.float32)
    
    state_dim = A.shape[0]
    state = np.zeros((state_dim, 1), dtype=np.float32)
    state[0, 0] = x[0]
    state[1, 0] = y[0]
    
    
    L = 10.0
    covariance = np.eye(state_dim, dtype=np.float32)
    covariance[0,0] = L
    covariance[1,1] = L
    covariance[2:, 2:] = np.eye(state_dim-2, dtype=np.float32) * (L*2)
    
    filtered_x, filtered_y = [x[0]], [y[0]]
    
    for j in range(1, len(x)):
        observation = np.array([[x[j]], [y[j]]], dtype=np.float32)
        state, covariance, _, _ = kalman_step(A, C, Q, R, observation, state, covariance)
        filtered_x.append(state[0, 0])
        filtered_y.append(state[1, 0])
        
    return filtered_x, filtered_y


# x, y = generate_trajectory('sine')
# model = 'NCV'
# fx, fy = run_kalman(x, y, model, q=1.0, r=5.0)
# plt.plot(x, y, 'r-', label='Measured')
# plt.plot(fx, fy, 'b-', label='Filtered')
# plt.title(f'model = {model}, q=1.0, r=5.0')
# plt.legend()
# plt.show()

qs = [1,5]
rs = [1,5,100]
x, y = generate_trajectory('sine')
plt.figure(figsize=(10, 5))
plot_idx = 1
for i, model in enumerate(['RW', 'NCV', 'NCA']):
    for r in rs:
        for q in qs:
            fx, fy = run_kalman(x, y, model, q=q, r=r)
            plt.subplot(3, 4, plot_idx)
            plt.plot(x, y, 'r-', alpha=0.6)
            plt.plot(x, y, 'o', markerfacecolor='none', markeredgecolor='red', alpha=0.7, markersize=3, markeredgewidth=1.2)
            plt.plot(fx, fy, 'b-', alpha=0.6)
            plt.plot(fx, fy, 'o', markerfacecolor='none', markeredgecolor='blue', alpha=0.7, markersize=3, markeredgewidth=1.2)
            plt.title(f'{model} (q={q}, r={r})')
            plot_idx += 1
            if r!= 1:
                break
plt.tight_layout(pad=3.0, w_pad=2.0, h_pad=2.0)
plt.show()



# Za dobit modele:
# import sympy as sp
# from sympy.interactive.printing import init_printing
# init_printing(use_unicode = False, wrap_line = False)
# T = sp.symbols('T')
# F = sp.Matrix([[0, 1, 0, 0 ], [0, 0,0,0], [0, 0, 0, 1], [0, 0, 0, 0]])
# F_rw = sp.Matrix([[0, 0], [0, 0]])
# F_nca = sp.Matrix([[0, 0, 1, 0, 0, 0], 
#                    [0, 0, 0, 1, 0, 0], 
#                    [0, 0, 0, 0, 1, 0], 
#                    [0, 0, 0, 0, 0, 1], 
#                    [0, 0, 0, 0, 0, 0], 
#                    [0, 0, 0, 0, 0, 0]])
# L = sp.Matrix([[0, 0], [1, 0], [0, 0], [0, 1]])
# L_rw = sp.Matrix([[1, 0], [0, 1]])
# A_rw = sp.exp(F_rw*T)
# print("RW A:")
# sp.pprint(A_rw)
# tau = sp.symbols('tau')
# phi_tau_rw = sp.exp(F_rw*tau)
# Q_integrand_rw = phi_tau_rw * L_rw * L_rw.T * phi_tau_rw.T
# Q_rw = sp.integrate(Q_integrand_rw, (tau, 0, T))
# print("RW Q:")
# sp.pprint(Q_rw)

# A = sp.exp(F*T)
# print("A:")
# sp.pprint(A)

# phi_tau = sp.exp(F*tau)
# Q_integrand = phi_tau * L * L.T * phi_tau.T
# Q = sp.integrate(Q_integrand, (tau, 0, T))
# print("Q:")
# sp.pprint(Q)
