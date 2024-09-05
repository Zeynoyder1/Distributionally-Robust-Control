import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from numpy import linalg as LA

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
            # 1. Check if A is square
            if self.A.shape[0] != self.A.shape[1]:
                raise ValueError("Matrix A is not square")

            dx = self.A.shape[0]  # Number of rows/columns in A (state dimension)
            dv = self.B.shape[1]  # Number of columns in B (control input dimension)
            dw = self.Bw.shape[1]  # Number of columns in Bw (disturbance input dimension)
            dy = self.Cy.shape[0]  # Number of rows in Cy (output y dimension)
            dz = self.Cz.shape[0]  # Number of rows in Cz (output z dimension)

            # 2. Check dimensions of B and Bw
            if self.B.shape[0] != dx:
                raise ValueError(f"Matrix B row dimension ({self.B.shape[0]}) does not match dx ({dx})")
            if self.Bw.shape[0] != dx:
                raise ValueError(f"Matrix Bw row dimension ({self.Bw.shape[0]}) does not match dx ({dx})")

            # 3. Check dimensions of Cy and Cz
            if self.Cy.shape[1] != dx:
                raise ValueError(f"Matrix Cy column dimension ({self.Cy.shape[1]}) does not match dx ({dx})")
            if self.Cz.shape[1] != dx:
                raise ValueError(f"Matrix Cz column dimension ({self.Cz.shape[1]}) does not match dx ({dx})")

            # 4. Check dimensions of Dyu, Dyw, Dzu, Dzw
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

    def check_dimensions_controller(self, F, G, H, J, number_states, number_outputs):
        try:
            # Check dimensions of F and G
            if F.shape[0] != F.shape[1]:
                raise ValueError("Matrix F is not square")
            if F.shape[0] != number_states:
                raise ValueError(f"Matrix F dimensions {F.shape} do not match the number of states ({number_states})")
            if G.shape[0] != number_states:
                raise ValueError(f"Matrix G row dimension ({G.shape[0]}) does not match the number of states ({number_states})")
            if G.shape[1] != number_states:
                raise ValueError(f"Matrix G column dimension ({G.shape[1]}) does not match the number of outputs ({number_outputs})")

            # Check dimensions of H and J
            if H.shape[0] != number_states:
                raise ValueError(f"Matrix H row dimension ({H.shape[0]}) does not match the number of outputs ({number_states})")
            if H.shape[1] != number_states:
                raise ValueError(f"Matrix H column dimension ({H.shape[1]}) does not match the number of states ({number_states})")
            if J.shape[0] != number_states:
                raise ValueError(f"Matrix J row dimension ({J.shape[0]}) does not match the number of outputs ({number_states})")
            if J.shape[1] != number_states:
                raise ValueError(f"Matrix J column dimension ({J.shape[1]}) does not match the number of outputs ({number_states})")

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

        # Loop over the time horizon (t = 0 to t = T-1)
        for t in range(T):
            x_t = states[-1]  # most recent vector from states list
            s_t = s_values[-1]  # most recent vector from s_values list

            d_t = disturbance_inputs[t]  # disturbance input for current time step t

            # Compute output y including disturbance
            y_t = self.Cy @ x_t + self.Dyw @ d_t  # observe/measure
            outputs_y.append(y_t)

            # Compute the controller
            u_t = H @ s_t + J @ y_t
            u_values.append(u_t)

            # Compute output z including disturbance
            z_t = self.Cz @ x_t + self.Dzu @ u_t + self.Dzw @ d_t  # compute the performance output
            outputs_z.append(z_t)

            # Compute next state vector including disturbance
            x_next = self.A @ x_t + self.B @ u_t + self.Bw @ d_t  # update the state
            states.append(x_next)

            # Update the controller's state
            s_next = F @ s_t + G @ y_t
            s_values.append(s_next)

        return states, outputs_y, outputs_z, s_values, u_values
    
  
def hinf_riccati_iteration(A, B, Q, R, gamma, P0, N):

    P = P0
    I = np.identity(A.shape[0])  
    errors = []

    for _ in range(N):
    
        P_next = Q + A.T @ P @ np.linalg.inv(I +(B @ np.linalg.inv(R) @ B.T - (I / gamma**2)) @ P) @ A

        # error between P_next and P
        error = np.linalg.norm(P_next - P, 2)  # Frobenius norm as a measure of the difference
        errors.append(error)

        P = P_next


    spectral_norm_P = np.linalg.norm(P, 2)
    if spectral_norm_P <= gamma**2:
        
        K = np.linalg.inv(R) @ B.T @ P @ np.linalg.inv(I + (B @ np.linalg.inv(R) @ B.T - (I / gamma**2)) @ P) @ A
        return P, K, errors
    else:
        raise ValueError(f"Solved P does not satisfy the H-infinity condition: norm(P) = {spectral_norm_P}, expected <= {gamma**2}")


def compute_cost(states, u_values, Q, R):
    costs = []
    total_cost = 0
    
    for x, u in zip(states, u_values):
        cost = x.T @ Q @ x + u.T @ R @ u
        costs.append(cost.item())  # Append the scalar value of the cost
        total_cost += cost.item()
        
    return costs, total_cost


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

Dyu = np.zeros((4, 2))
Dzu = np.array([
    [0, 1]])
Cz = np.array([
    [1, 0, 0, 0]])
Cy = np.identity(4)
Dyw = np.zeros((4, 2))
Dzw = np.zeros((1, 2))

x0 = np.array([[10], [10], [0], [0]])

Q = np.identity(4)
R = np.identity(2)
#P0 = np.identity(A.shape[0])
P0 = Q
gamma = 5.0
N = 15 


# # Run H-infinity Riccati iteration
P, K, errors = hinf_riccati_iteration(A, B, Q, R, gamma, P0, N)
# eigenvalues, _ = LA.eig(A - B @ K)
# print(eigenvalues)

# # Plotting the error norms
# plt.figure(figsize=(8, 5))
# plt.plot(range(len(errors)), errors, marker='o', linestyle='-', color='b')
# plt.title('Error Norm Between Successive P Matrices')
# plt.xlabel('Iteration')
# plt.ylabel('Norm of P_{n+1} - P_n')
# plt.show()


T = 50


dw = Bw.shape[1]  # dimension of disturbance vector

# disturbance_inputs = [np.random.randn(dw, 1) for _ in range(T)]
disturbance_inputs =[np.zeros((dw, 1)) for _ in range(T)]

####### controller #######
F = np.zeros((1, 1))
G = np.zeros((1, 4))
H = np.zeros((2, 1))
J_controller = -K
J_no_controller = np.zeros((2, 4))
s0 = np.zeros((1, 1))

system = DiscreteDynamicalSystem(A, B, Bw, Cy, Cz, Dyu, Dzu, Dzw, Dyw)

# Simulate the system with disturbance
states_controller, outputs_y, outputs_z, s_values, u_values = system.simulate_closed_loop_system(
    x0, T, F, G, H, J_controller, s0, disturbance_inputs)
states_no_controller, outputs_y, outputs_z, s_values, u_values = system.simulate_closed_loop_system(
    x0, T, F, G, H, J_no_controller, s0, disturbance_inputs)

# compute cost
costs_controller, total_cost_controller = compute_cost(states_controller, u_values, Q, R)
costs_no_controller, total_cost_no_controller = compute_cost(states_no_controller, u_values, Q, R)

# Plotting the data
time_steps = np.arange(T + 1)
time_steps_outputs = np.arange(T)

plt.figure(figsize=(14, 6))
plt.suptitle("Discrete-time Linear Dynamical System with Disturbances")

# Plot the state variables over time
plt.subplot(1, 2, 1)
for i in range(len(states_controller[0])):
    state_values_with = [states_controller[j][i] for j in range(len(states_controller))]
    plt.plot(time_steps, state_values_with, label=f'State x{i+1}')
plt.title("Change in State Over Time With Robust Controller")
plt.xlabel("Time Step")
plt.ylabel("State Value")
plt.legend()

plt.subplot(1, 2, 2)
for i in range(len(states_no_controller[0])):
    state_values_without = [states_no_controller[j][i] for j in range(len(states_no_controller))]
    plt.plot(time_steps, state_values_without, label=f'State x{i+1}')
plt.title("Change in State Over Time With Robust Controller")
plt.xlabel("Time Step")
plt.ylabel("Output Value")
plt.legend()

plt.savefig("discrete_lds_with_disturbances.png")
plt.show()


