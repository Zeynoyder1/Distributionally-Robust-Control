import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

class DiscreteDynamicalSystem:
    def __init__(self, A, B, Bw, Cy, Cz, Dyu, Dzu, Dzw, Dyw):
        self.A = A
        self.B = B
        self.Bw = Bw
        self.Cy = Cy
        self.Cz = Cz
        self.Dyu = Dyu
        self.Dzu = Dzu
        self.Dzw = Dzw
        self.Dyw = Dyw

    def check_dimensions_system(self):
        try:
            if self.A.shape[0] != self.A.shape[1]:
                raise ValueError("Matrix A is not square")

            dx = self.A.shape[0]
            dv = self.B.shape[1]
            dw = self.Bw.shape[1]
            dy = self.Cy.shape[0]
            dz = self.Cz.shape[0]

            if self.B.shape[0] != dx:
                raise ValueError(f"Matrix B row dimension ({self.B.shape[0]}) does not match dx ({dx})")
            if self.Bw.shape[0] != dx:
                raise ValueError(f"Matrix Bw row dimension ({self.Bw.shape[0]}) does not match dx ({dx})")
            if self.Cy.shape[1] != dx:
                raise ValueError(f"Matrix Cy column dimension ({self.Cy.shape[1]}) does not match dx ({dx})")
            if self.Cz.shape[1] != dx:
                raise ValueError(f"Matrix Cz column dimension ({self.Cz.shape[1]}) does not match dx ({dx})")
            if self.Dyu.shape != (dy, dv):
                raise ValueError(f"Matrix Dyu dimensions {self.Dyu.shape} do not match (dy, dv) = ({dy}, {dv})")
            if self.Dyw.shape != (dy, dw):
                raise ValueError(f"Matrix Dyw dimensions {self.Dyw.shape} do not match (dy, dw) = ({dy}, {dw})")
            if self.Dzu.shape != (dz, dv):
                raise ValueError(f"Matrix Dzu dimensions {self.Dzu.shape} do not match (dz, dv) = ({dz}, {dv})")
            if self.Dzw.shape != (dz, dw):
                raise ValueError(f"Matrix Dzw dimensions {self.Dzw.shape} do not match (dz, dw) = ({dz}, {dw})")

        except ValueError as e:
            print(f"Dimension check failed: {e}")
            raise

    def simulate_closed_loop_system(self, x0, T, F, G, H, J, s0, disturbance_inputs):
        self.check_dimensions_system()

        states = [x0]
        s_values = [s0]
        outputs_y = []
        outputs_z = []
        u_values = []

        for t in range(T):
            x_t = states[-1]
            s_t = s_values[-1]
            d_t = disturbance_inputs[t]

            y_t = self.Cy @ x_t + self.Dyw @ d_t
            outputs_y.append(y_t)

            u_t = H @ s_t + J @ y_t
            u_values.append(u_t)

            z_t = self.Cz @ x_t + self.Dzu @ u_t + self.Dzw @ d_t
            outputs_z.append(z_t)

            x_next = self.A @ x_t + self.B @ u_t + self.Bw @ d_t
            states.append(x_next)

            s_next = F @ s_t + G @ y_t
            s_values.append(s_next)

        return states, outputs_y, outputs_z, s_values, u_values


def hinf_riccati_iteration(A, B, Q, R, gamma, P0, N):
    P = P0
    I = np.identity(A.shape[0])  
    errors = []

    for _ in range(N):
        P_next = Q + A.T @ P @ np.linalg.inv(I +(B @ np.linalg.inv(R) @ B.T - (I / gamma**2)) @ P) @ A
        error = np.linalg.norm(P_next - P, 2)
        errors.append(error)
        P = P_next

    spectral_norm_P = np.linalg.norm(P, 2)
    if spectral_norm_P <= gamma**2:
        K = np.linalg.inv(R) @ B.T @ P @ np.linalg.inv(I + (B @ np.linalg.inv(R) @ B.T - (I / gamma**2)) @ P) @ A
        return P, K, errors
    else:
        raise ValueError(f"Solved P does not satisfy the H-infinity condition: norm(P) = {spectral_norm_P}, expected <= {gamma**2}")


# Example setup
A = np.array([[0.99, 0.03, -0.02, -0.32],
              [0.01, 0.47, 4.7, 0],
              [0.02, -0.06, 0.4, 0],
              [0.01, -0.04, 0.72, 0.99]])

B = np.array([[0.01, 0.99],
              [-3.44, 1.66],
              [-0.83, 0.44],
              [-0.47, 0.25]])

Bw = np.array([[1, 0],
               [0, 1],
               [0, 0],
               [0, 0]])

Cy = np.identity(4)
Cz = np.array([[1, 0, 0, 0]])
Dyu = np.zeros((4, 2))
Dzu = np.array([[0, 1]])
Dyw = np.zeros((4, 2))
Dzw = np.zeros((1, 2))

x0 = np.array([[10], [10], [0], [0]])
Q = np.identity(4)
R = np.identity(2)
P0 = Q
gamma = 5.0
N = 15 

PS = la.solve_discrete_are(A, B, Q, R)
KS = np.linalg.inv(R + B.T @ PS @ B) @ (B.T @ PS @ A)
# Run H-infinity Riccati iteration
P, K, errors = hinf_riccati_iteration(A, B, Q, R, gamma, P0, N)

T = 50
dw = Bw.shape[1]
disturbance_inputs = [np.zeros((dw, 1)) for _ in range(T)]

# Controllers setup
F = np.zeros((1, 1))
G = np.zeros((1, 4))
H = np.zeros((2, 1))
J_no_controller = np.zeros((2, 4))
J_stochastic = -KS  # Stochastic controller with small random gains
J_robust = -K

s0 = np.zeros((1, 1))

system = DiscreteDynamicalSystem(A, B, Bw, Cy, Cz, Dyu, Dzu, Dzw, Dyw)

# Simulate the system under three conditions
states_no_controller, _, _, _, _ = system.simulate_closed_loop_system(
    x0, T, F, G, H, J_no_controller, s0, disturbance_inputs)
states_stochastic, _, _, _, _ = system.simulate_closed_loop_system(
    x0, T, F, G, H, J_stochastic, s0, disturbance_inputs)
states_robust, _, _, _, _ = system.simulate_closed_loop_system(
    x0, T, F, G, H, J_robust, s0, disturbance_inputs)

# Plotting the results
time_steps = np.arange(T + 1)

plt.figure(figsize=(18, 6))
plt.suptitle("Change in State Over Time of Discrete-time Linear Dynamical System (No Distrubances)")

# Subplot 1: Without controller
plt.subplot(1, 3, 1)
for i in range(len(states_no_controller[0])):
    plt.plot(time_steps, [states_no_controller[j][i] for j in range(len(states_no_controller))], label=f'State x{i+1}')
plt.title("No Controller")
plt.xlabel("Time Step (T)")
plt.ylabel("State Value")
plt.legend()

# Subplot 2: With Stochastic Controller
plt.subplot(1, 3, 2)
for i in range(len(states_stochastic[0])):
    plt.plot(time_steps, [states_stochastic[j][i] for j in range(len(states_stochastic))], label=f'State x{i+1}')
plt.title("Stochastic Controller")
plt.xlabel("Time Step (T)")
plt.ylabel("State Value")
plt.legend()

# Subplot 3: With Robust Controller
plt.subplot(1, 3, 3)
for i in range(len(states_robust[0])):
    plt.plot(time_steps, [states_robust[j][i] for j in range(len(states_robust))], label=f'State x{i+1}')
plt.title("Robust Controller")
plt.xlabel("Time Step (T)")
plt.ylabel("State Value")
plt.legend()

plt.tight_layout()
plt.savefig("States Plot with No Disturbances")
plt.show()
