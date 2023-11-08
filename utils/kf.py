import numpy as np
from numpy.linalg import inv

def predict(X, A, B, U, P, Q):
    # X = A @ X + B @ U
    X = np.dot(A, X) + np.dot(B, U)
    P = A @ P @ A.T + Q
    return X, P

def update(X, P, Y, H, R):
    S = H @ P @ H.T + R
    K = P @ H.T @ inv(S)
    X = X + K @ Y
    P = P - K @ S @ K.T
    return X, P