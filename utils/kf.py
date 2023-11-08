import numpy as np
from numpy.linalg import inv

from utils import data

def predict(X, A, B, U, P, Q, dt):
    X = A @ X + B @ U
    X[2] = data.normalize_rad(X[2])

    # F_x = np.array([[1, 0, U[0]/U[1] * (np.cos(X[2]) - np.cos(X[2]))],
    #                 [0, 1, U[0]/U[1] * (np.sin(X[2]) - np.sin(X[2]))],
    #                 [0, 0, 1]])

    # F_u = np.array([[(np.sin(X[2]) - np.sin(X[2]))/U[1], U[0]/(U[1]**2) * (np.sin(X[2]) - dt*U[1]*np.cos(X[2]) - np.sin(X[2]))],
    #                 [(np.cos(X[2]) - np.cos(X[2]))/U[1], U[0]/(U[1]**2) * (np.cos(X[2]) + dt*U[1]*np.sin(X[2]) - np.cos(X[2]))],
    #                 [0, dt]])

    # P = F_x @ P @ F_x.T + F_u @ Q @ F_u.T
    P = A @ P @ A.T + Q
    
    return X, P

def update(X, Z, P, R, landmark):
    mod = np.sqrt(np.power(X[0] - landmark[0],2)+np.power(X[1] - landmark[1],2))
    ber = data.normalize_rad(np.arctan2(X[1] - landmark[1], X[0] - landmark[0]) - X[2])

    Z_pred = np.array([mod, ber])
    Y = Z - Z_pred
    H = np.array([[ (X[0] - landmark[0])/mod,    (X[1] - landmark[1])/mod,     0],
                  [-(X[1] - landmark[1])/mod**2, (X[0] - landmark[0])/mod**2, -1]])
    
    S = H @ P @ H.T + R
    K = P @ H.T @ inv(S)
    K[2, 0] = 0
    K[2, 1] = 0
    X = X + K @ Y
    X[2] = data.normalize_rad(X[2])

    P = P - K @ S @ K.T

    return X, P