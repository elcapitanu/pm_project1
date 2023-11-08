import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

from utils import video
from utils import data
from utils import kf

#video
vid = False
output_file = './videos/task1.avi'
frame_rate = 120
frame_width = 1920
frame_height = 1080
out = video.init(output_file, frame_rate, frame_width, frame_height)
#data
file_path = "./data/data1.txt"
df = data.extract(file_path)

x_real = df[:, 1]
y_real = df[:, 2]
theta_real = df[:, 3]
dt = df[1,0]

x0 = x_real[0]
y0 = y_real[0]
theta0 = theta_real[0]

sigma_v = 0.5
sigma_w = 0.05

sigma_r = 0.5
sigma_psi = 0.1
l1 = [0,0]
l2 = [10,0]

x_pred = []
y_pred = []
theta_pred = []

x1_pred = []
y1_pred = []

x2_pred = []
y2_pred = []


state = np.array([x0, y0, theta0])

A = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])

P = np.array([[sigma_v**2,  0,          0],
              [0,           sigma_v**2, 0],
              [0,           0,          sigma_w**2]])

Q = np.array([[sigma_v**2,  0,          0],
              [0,           sigma_v**2, 0],
              [0,           0,          sigma_w**2]])

H = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 0]])

R = np.array([[sigma_r**2,  0,              0],
              [0,           sigma_psi**2,   0],
              [0,           0,              0]])

for i in range(len(df)):
    t = df[i,0]
    v = df[i,4]
    w = df[i,5]
    r1 = df[i,6]
    psi1 = df[i,7]
    r2 = df[i,8]
    psi2 = df[i,9]

    # kalman filter predict
    # state[0] = state[0] + v * np.cos(state[2]) * dt
    # state[1] = state[1] + v * np.sin(state[2]) * dt
    # state[2] = data.normalize_rad(state[2] + w * dt)
    
    B = np.array([[dt * np.cos(state[2]), 0],
                  [dt * np.sin(state[2]), 0],
                  [0, dt]])
    
    U = np.array([v, w])

    state, P = kf.predict(state, A, B, U, P, Q)

    print(P)

    state[2] = data.normalize_rad(state[2])

    x_pred.append(state[0])
    y_pred.append(state[1])
    theta_pred.append(state[2])

    # kalman filter update

    # state, P = kf.update(state, P, Y, H, R)

    phi1 = data.normalize_rad(psi1+state[2]-np.pi)
    phi2 = data.normalize_rad(psi2+state[2]-np.pi)

    x1_pred.append(l1[0] + r1 * np.cos(phi1))
    y1_pred.append(l1[1] + r1 * np.sin(phi1))

    x2_pred.append(l2[0] + r2 * np.cos(phi2))
    y2_pred.append(l2[1] + r2 * np.sin(phi2))

    if vid:
        out = video.update(out, frame_width, frame_height, df[i,1], df[i,2], df[i,3])

if vid:
    video.export(out)

data.show_comparasion(df[:, 0], x_real, x_pred)
data.show_comparasion(df[:, 0], y_real, y_pred)
data.show_comparasion(df[:, 0], theta_real, theta_pred)

plt.scatter(x_real, y_real, marker='.', color='red', label='real')
plt.scatter(x_pred, y_pred, marker='.', color='blue', label='pred')
plt.scatter(x1_pred, y1_pred, marker='.', color='cyan', label='with beacon 1')
plt.scatter(x2_pred, y2_pred, marker='.', color='green', label='with beacon 2')
plt.legend()
plt.show()