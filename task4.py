import numpy as np
import matplotlib.pyplot as plt

from utils import video
from utils import data
from utils import kf

def mark_beacon():
    return

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

P = np.eye(3)

# Q = np.array([[sigma_v**2, 0],
#               [0,          sigma_w**2]])

Q = np.array([[sigma_v**2, 0,          0],
              [0,          sigma_v**2, 0],
              [0,          0,          sigma_w**2]])

R = np.array([[sigma_r**2, 0],
              [0,          sigma_psi**2]])

for i in range(len(df)):
    t = df[i,0]
    v = df[i,4]
    w = df[i,5]
    r = df[i,6]
    psi = df[i,7]

    # kalman filter predict    
    B = np.array([[dt * np.cos(state[2]), 0],
                  [dt * np.sin(state[2]), 0],
                  [0,                     dt]])
    
    U = np.array([v, w])

    state, P = kf.predict(state, A, B, U, P, Q, dt)

    x_pred.append(state[0])
    y_pred.append(state[1])
    theta_pred.append(state[2])

    mark_beacon()

    if vid:
        out = video.update(out, t, frame_width, frame_height, state, np.array([df[i,1],df[i,2],df[i,3]]), True, False)

if vid:
    video.export(out)

data.show_comparasion(df[:, 0], x_real, x_pred)
data.show_comparasion(df[:, 0], y_real, y_pred)
data.show_comparasion(df[:, 0], theta_real, theta_pred)

plt.scatter(x_real, y_real, marker='.', color='red', label='real')
plt.scatter(x_pred, y_pred, marker='.', color='blue', label='pred')
plt.legend()
plt.show()