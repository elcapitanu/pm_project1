import numpy as np
import matplotlib.pyplot as plt

from utils import video
from utils import data
from utils import kf

l1 = [0,0]
l2 = [10,0]

def choose_landmark(state):
    psi1 = data.normalize_rad(np.arctan2(l1[1]-state[1],l1[0]-state[0])-state[2])
    psi2 = data.normalize_rad(np.arctan2(l2[1]-state[1],l2[0]-state[0])-state[2])

    diff_psi1 = abs(psi - psi1)
    diff_psi2 = abs(psi - psi2)

    dist_l1 = np.sqrt((state[0]-l1[0])**2+(state[1]-l1[1])**2)
    dist_l2 = np.sqrt((state[0]-l2[0])**2+(state[1]-l2[1])**2)

    if dist_l1 < dist_l2:
      if(abs(diff_psi1) <= abs(diff_psi2)):
        return l1
      else:
        return l2
    else:
      if(abs(diff_psi2) <= abs(diff_psi1)):
        return l2
      else:
        return l1
      
    return 0

#video
vid = True
output_file = './videos/task3.avi'
frame_rate = 120
frame_width = 1920
frame_height = 1080
out = video.init(output_file, frame_rate, frame_width, frame_height)
#data
file_path = "./data/data3.txt"
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

x_pred = []
y_pred = []
theta_pred = []

state = np.array([x0, y0, theta0])

A = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]])

P = np.array([[1, 1, 1],
              [1, 1, 1],
              [1, 1, 1]])

Q = np.array([[sigma_v**2, 0],
              [0,          sigma_w**2]])

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

    # kalman filter update
    if (r != 0 and abs(psi) <= np.pi):
        Z = np.array([r, psi])
        landmark = choose_landmark(state)
        if (landmark):
            state, P = kf.update(state, Z, P, R, landmark)

    x_pred.append(state[0])
    y_pred.append(state[1])
    theta_pred.append(state[2])

    if vid:
        out = video.update(out, t, frame_width, frame_height, state, np.array([df[i,1],df[i,2],df[i,3]]), True)

if vid:
    video.export(out)

data.show_comparasion(df[:, 0], x_real, x_pred)
data.show_comparasion(df[:, 0], y_real, y_pred)
data.show_comparasion(df[:, 0], theta_real, theta_pred)

plt.scatter(x_real, y_real, marker='.', color='red', label='real')
plt.scatter(x_pred, y_pred, marker='.', color='blue', label='pred')
plt.legend()
plt.show()