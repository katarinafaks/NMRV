import numpy as np
import math
import matplotlib.pyplot as plt
from ex4_utils import kalman_step

def get_kinematic_matrices(model, dt, q):
    """Returns A (Phi), C (H), and Q for a given motion model."""
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
    A, C, Q = get_kinematic_matrices(model, dt, q)
    R = r * np.eye(2, dtype=np.float32)
    
    state_dim = A.shape[0]
    state = np.zeros((state_dim, 1), dtype=np.float32)
    state[0, 0] = x[0]
    state[1, 0] = y[0]
    
    covariance = np.eye(state_dim, dtype=np.float32)
    
    filtered_x, filtered_y = [x[0]], [y[0]]
    
    for j in range(1, len(x)):
        obs = np.array([[x[j]], [y[j]]], dtype=np.float32)
        state, covariance, _, _ = kalman_step(A, C, Q, R, obs, state, covariance)
        filtered_x.append(state[0, 0])
        filtered_y.append(state[1, 0])
        
    return filtered_x, filtered_y

# Example usage to plot (run this in your local environment)
x, y = generate_trajectory('spiral')
fx, fy = run_kalman(x, y, 'NCV', q=1.0, r=5.0)
plt.plot(x, y, 'r-', label='Measured')
plt.plot(fx, fy, 'b-', label='Filtered')
plt.legend()
plt.show()