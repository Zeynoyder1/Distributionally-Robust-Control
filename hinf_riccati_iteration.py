import numpy as np

def hinf_riccati_iteration(A, B, Q, R, gamma, P0, N):

    P = P0
    I = np.identity(A.shape[0])  

    for _ in range(N):
    
        P_next = Q + A.T @ P @ np.linalg.inv(I +(B @ np.linalg.inv(R) @ B.T - (I / gamma**2)) @ P) @ A
        P = P_next


    spectral_norm_P = np.linalg.norm(P, 2)
    if spectral_norm_P <= gamma**2:
        
        K = np.linalg.inv(R) @ B.T @ P @ np.linalg.inv(I + (B @ np.linalg.inv(R) @ B.T - (I *gamma**2 @ P) )) @ A
        return P, K
    else:
        raise ValueError(f"Solved P does not satisfy the H-infinity condition: norm(P) = {spectral_norm_P}, expected <= {gamma**2}")




A = np.array([[0.99, 0.03, -0.02, -0.32],
              [0.01, 0.47, 4.7, 0],
              [0.02, -0.06, 0.4, 0],
              [0.01, -0.04, 0.72, 0.99]])

B = np.array([[0.01, 0.99],
              [-3.44, 1.66],
              [-0.83, 0.44],
              [-0.47, 0.25]])

P0 = np.identity(A.shape[0])
Q = np.identity(4)
R = np.identity(2)
gamma = 5.0
N = 100 


P, K = hinf_riccati_iteration(A, B, Q, R, gamma, P0, N)

print("P Matrix:\n", P)
print("K Matrix:\n", K)