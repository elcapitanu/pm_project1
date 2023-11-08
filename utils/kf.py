import numpy as np
from numpy.linalg import inv

def predict(X, A, B, U, P, Q):
    X = A @ X + B @ U
    P = A @ P @ A.T + Q
    return X, P

def update(X, P, Z, H, R):
    Y = Z - X
    # S = H @ P @ H.T + R
    # K = P @ H.T @ inv(S)
    K = np.array([[0.01, 0, 0],
                  [0, 0.01, 0],
                  [0, 0, 0.01]])
    X = X + K @ Y
    # P = P - K @ S @ K.T
    return X, P