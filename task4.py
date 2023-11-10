from numpy.linalg import inv
import numpy as np
import matplotlib.pyplot as plt

from utils import video
from utils import data

#video
vid = True
output_file = './videos/task4.avi'
frame_rate = 120
frame_width = 1920
frame_height = 1080
out = video.init(output_file, frame_rate, frame_width, frame_height)
#data
file_path = "./data/data4.txt"
df = data.extract(file_path)

x_real = df[:, 1]
y_real = df[:, 2]
theta_real = df[:, 3]
dt = df[1,0]

sigma_v = 0.5
sigma_w = 0.05

sigma_r = 0.5
sigma_psi = 0.1

x_pred = []
y_pred = []
theta_pred = []

x1_pred = []
y1_pred = []

x2_pred = []
y2_pred = []

state = np.array([[x_real[0]],
                  [y_real[0]],
                  [theta_real[0]]])

P = np.eye(3) * 999

Q = np.array([[sigma_v**2, 0],
              [0,          sigma_w**2]])

R = np.array([[sigma_r**2, 0],
              [0,          sigma_psi**2]])


# save beacons [x, y, n] n times viewed
known_beacons = []
saved_beacons = []
#!!!!!
def get_beacon(state, r, psi):

    x, y, theta = state[:3].reshape(3)
    bx = x + r*np.cos(theta + psi)
    by = y + r*np.sin(theta + psi)

    if len(state) <= 3:
        known_beacons.append([bx, by, 1])
        return True, [bx, by, 0]

    x, y, theta = state[:3].reshape(3)
    bx = x + r*np.cos(theta + psi)
    by = y + r*np.sin(theta + psi)

    for i, beacon in enumerate(known_beacons):
        dst = np.sqrt((beacon[0] - bx)** 2 + (beacon[1] - by)** 2)
        if dst <= 12 :
            n = beacon[2]
            beacon[0] = (beacon[0] * n + bx)/(n+1)
            beacon[1] = (beacon[1] * n + by)/(n+1)
            beacon[2] = n+1
            known_beacons[i] = beacon
            return False, [beacon[0], beacon[1], i]

    known_beacons.append([bx, by, 1])
    id = len(known_beacons) - 1
    print(id)
    return True, [bx, by, id]

def predict(X, U, P, Q, dt):
    x, y, theta = X[:3].reshape(3)
    v, omega = U.reshape(2)

    theta_pred = data.normalize_rad(theta + omega * dt)
    x_pred = x + v/omega * ( np.sin(theta_pred) - np.sin(theta))
    y_pred = y + v/omega * (-np.cos(theta_pred) + np.cos(theta))

    F_x = np.array([[1, 0, v/omega * (np.cos(theta_pred) - np.cos(theta))],
                    [0, 1, v/omega * (np.sin(theta_pred) - np.sin(theta))],
                    [0, 0, 1]])
    
    F_u = np.array([[(np.sin(theta_pred) - np.sin(theta))/omega, v/(omega**2) * (np.sin(theta_pred) - dt*omega*np.cos(theta_pred) - np.sin(theta))],
                    [(np.cos(theta) - np.cos(theta_pred))/omega, v/(omega**2) * (np.cos(theta_pred) + dt*omega*np.sin(theta_pred) - np.cos(theta))],
                    [0, dt]])

    Fx = np.eye(len(X), len(X))
    for line in range(F_x.shape[0]):
        for column in range(F_x.shape[0]):
            Fx[line][column] = F_x[line][column]

    Fu = np.zeros((len(X), 2))
    for line in range(F_u.shape[0]):
        for column in range(F_u.shape[1]):
            Fu[line][column] = F_u[line][column]

    tmp = np.eye(len(X), len(X)) * 10
    for line in range(P.shape[0]):
        for column in range(P.shape[1]):
            tmp[line][column] = P[line][column]
    P = tmp

    P = Fx @ P @ Fx.T + Fu @ Q @ Fu.T

    X_pred = X
    X_pred[0,0] = x_pred
    X_pred[1,0] = y_pred
    X_pred[2,0] = theta_pred
    return X_pred, P


def step(state, r, psi, beacon):

    x, y, theta = state[:3].reshape(3)
    z = np.array([[r, psi]])

    norm = np.sqrt((x-beacon[0])**2 + (y-beacon[1])**2)

    H_1 = np.array([[(x-beacon[0])/norm,      (y-beacon[1])/norm,     0],
                  [-(y-beacon[1])/(norm**2),(x-beacon[0])/(norm**2),-1]])


    H = np.zeros((2, len(state)))

    for line in range(H_1.shape[0]):
      for column in range(H_1.shape[1]):
        H[line][column] = H_1[line][column]
    H[0, 3+2*beacon[2]] = 1
    H[1, 3+2*beacon[2]+1] = 1


    h = np.array([[norm,
                   np.arctan2(beacon[1]-y, beacon[0]-x)-theta]])

    delta = (z-h).reshape(2, 1)

    return delta, R, H

for i in range(len(df)):
    t = df[i,0]
    v = df[i,4]
    w = df[i,5]
    r = df[i,6]
    psi = df[i,7]

    U = np.array([[v], [w]])

    if (r != 0) and (psi <= np.pi/4): 
        new, beacon = get_beacon(state, r, psi)
        if new :
            state = np.vstack((state, [[r], [psi]]))

        state, P = predict(state, U, P, Q, dt)
        delta, R, H = step(state, r, psi, beacon)

        K = P @ H.T @ inv(H @ P @ H.T + R)
        P = P - K @ (H @ P @ H.T + R) @ K.T

        state = state + K @ delta

    # no beacon
    else:
        state, P = predict(state, U, P, Q, dt)

    x_pred.append(state[0, 0])
    y_pred.append(state[1, 0])
    theta_pred.append(state[2, 0])

    if vid:
        out = video.update_2(out, t, frame_width, frame_height, state, np.array([df[i,1],df[i,2],df[i,3]]), True, True, known_beacons)

if vid:
    video.export(out)

print(known_beacons)

data.show_comparasion(df[:, 0], x_real, x_pred)
data.show_comparasion(df[:, 0], y_real, y_pred)
data.show_comparasion(df[:, 0], theta_real, theta_pred)

plt.scatter(x_real, y_real, marker='.', color='red', label='real') # type: ignore
plt.scatter(x_pred, y_pred, marker='.', color='blue', label='pred') # type: ignore
plt.legend()
plt.show()