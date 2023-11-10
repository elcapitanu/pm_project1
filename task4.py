from doctest import FAIL_FAST
import numpy as np
import matplotlib.pyplot as plt

from utils import video
from utils import data

#video
vid = False
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

state = np.array([[0], [0], [0]])

P = np.eye(3)

Q = np.array([[sigma_v**2, 0],
              [0,          sigma_w**2]])

R = np.array([[sigma_r**2, 0],
              [0,          sigma_psi**2]])

from numpy.linalg import inv


# save beacons [x, y, n] n times viewed
known_beacons = []
def new_beacon(state, r, psi):
    
    if len(state) <= 3:
        return True

    x, y, theta = state[:3].reshape(3)
    bx = x + r*np.cos(theta + psi)
    by = y + r*np.sin(theta + psi)

    for i, beacon in enumerate(known_beacons):
        dst = np.sqrt((beacon[0] - bx)** 2 + (beacon[1] - by)** 2)
        if dst <= 10 :
            n = beacon[2]
            beacon[0] = (beacon[0] * n + bx)/(n+1)
            beacon[1] = (beacon[1] * n + by)/(n+1)
            beacon[2] = n+1
            known_beacons[i] = beacon
            return False

    known_beacons.append([bx, by, 1])
    return True

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

    P = F_x @ P @ F_x.T + F_u @ Q @ F_u.T

    X_pred = X
    X_pred[0,0] = x_pred
    X_pred[1,0] = y_pred
    X_pred[2,0] = theta_pred
    return X_pred, P


for i in range(len(df)):
    t = df[i,0]
    v = df[i,4]
    w = df[i,5]
    r = df[i,6]
    psi = df[i,7]

    if new_beacon(state, r, psi) :
        state = np.vstack((state, [[r], [psi]]))

    U = np.array([[v], [w]])

    state, P = predict(state, U, P, Q, dt)

    x_pred.append(state[0, 0])
    y_pred.append(state[1, 0])
    theta_pred.append(state[2, 0])

    if vid:
        out = video.update(out, t, frame_width, frame_height, state, np.array([df[i,1],df[i,2],df[i,3]]), True, False)

if vid:
    video.export(out)

exit()

data.show_comparasion(df[:, 0], x_real, x_pred)
data.show_comparasion(df[:, 0], y_real, y_pred)
data.show_comparasion(df[:, 0], theta_real, theta_pred)

plt.scatter(x_real, y_real, marker='.', color='red', label='real') # type: ignore
plt.scatter(x_pred, y_pred, marker='.', color='blue', label='pred') # type: ignore
plt.legend()
plt.show()