# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # control system test design notebook
# This notebook is an example of how to install a control system using python.
# I think, idk what I'm doing

# %%
import control as ct
import numpy as np
import matplotlib.pyplot as plt

# %%
# Example: double integrator (position + velocity)
A = np.array([[0, 1],
              [0, 0]])
B = np.array([[0],
              [1]])
C = np.eye(2)       # observe both states
D = np.zeros((2,1))

sys = ct.ss(A, B, C, D)
print(sys)

# %%
Wc = ct.ctrb(A, B)
print("Controllable:", np.linalg.matrix_rank(Wc) == A.shape[0])

# %%
# Start with identity (Bryson's rule: 1/max_acceptable_value^2)
Q = np.diag([10, 1])   # penalize position more than velocity
R = np.array([[1]])    # single input cost
K, S, E = ct.lqr(sys, Q, R)

print("Gain K:", K)
print("Closed-loop poles:", E)

# %%
# Closed-loop A matrix
A_cl = A - B @ K

sys_cl = ct.ss(A_cl, B, C, D)
print("Closed-loop poles:", np.linalg.eigvals(A_cl))

# %%
ctrl, clsys = ct.create_statefbk_iosystem(sys, K)

# %%
t = np.linspace(0, 10, 500)
x0 = np.array([1.0, 0.0])   # initial condition (away from equilibrium)

# Simulate initial condition response
response = ct.initial_response(sys_cl, t, x0)

plt.figure()
plt.plot(response.time, response.outputs.T)
plt.xlabel("Time (s)")
plt.ylabel("States")
plt.title("LQR Closed-Loop Response")
plt.legend(["x1 (position)", "x2 (velocity)"])
plt.grid(True)
plt.show()

# %%
# Check closed-loop stability
poles = ct.poles(sys_cl)
print("All poles stable:", all(p.real < 0 for p in poles))

# Reconstruct control effort over time
t_out, y_out, x_out = ct.initial_response(sys_cl, t, x0, return_x=True)
u_out = -(K @ x_out)   # control effort at each timestep
plt.plot(t_out, u_out.T); plt.title("Control Effort u(t)")
