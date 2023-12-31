import numpy as np
from numpy.linalg import inv

from utils import data

def predict(X, U, P, Q, dt):
    theta = X[2]
    
    X[2] = data.normalize_rad(X[2] + U[1] * dt)
    X[0] = X[0] + U[0]/U[1] * ( np.sin(X[2]) - np.sin(theta))
    X[1] = X[1] + U[0]/U[1] * (-np.cos(X[2]) + np.cos(theta))

    F_x = np.array([[1, 0, U[0]/U[1] * (np.cos(X[2]) - np.cos(theta))],
                    [0, 1, U[0]/U[1] * (np.sin(X[2]) - np.sin(theta))],
                    [0, 0, 1]])

    F_u = np.array([[(np.sin(X[2]) - np.sin(theta))/U[1], U[0]/(U[1]**2) * (np.sin(X[2]) - dt*U[1]*np.cos(X[2]) - np.sin(theta))],
                    [(np.cos(X[2]) - np.cos(theta))/U[1], U[0]/(U[1]**2) * (np.cos(X[2]) + dt*U[1]*np.sin(X[2]) - np.cos(theta))],
                    [0, dt]])

    P = F_x @ P @ F_x.T + F_u @ Q @ F_u.T
    
    return X, P

def update(X, Z, P, R, landmark):
    mod = np.sqrt((X[0] - landmark[0])**2+(X[1] - landmark[1])**2)
    ber = data.normalize_rad(np.arctan2(landmark[1]-X[1], landmark[0]-X[0]) - X[2])

    Z_pred = np.array([mod, ber])
    Y = Z - Z_pred
    Y[1] = data.normalize_rad(Y[1])

    H = np.array([[ (X[0] - landmark[0])/mod,    (X[1] - landmark[1])/mod,     0],
                  [-(X[1] - landmark[1])/mod**2, (X[0] - landmark[0])/mod**2, -1]])
    
    S = H @ P @ H.T + R
    K = P @ H.T @ inv(S)

    X = X + K @ Y
    X[2] = data.normalize_rad(X[2])

    P = P - K @ S @ K.T

    return X, P