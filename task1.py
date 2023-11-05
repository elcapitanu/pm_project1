import numpy as np
import re
import matplotlib.pyplot as plt

def normalize_rad(r):
    if r > np.pi:
        r -= 2 * np.pi
    elif r < -np.pi:
        r += 2 * np.pi
    return r

file_path = "./data/data1.txt"

df = []

with open(file_path, 'r') as file:
    for row in file:
        col = re.split(r'\s+', row.strip())
        df.append(col)

data = np.array(df).astype(float)

x_real = data[:, 1]
y_real = data[:, 2]
theta_real = data[:, 3]
dt = data[1,0]

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

for i in range(len(data)):
    t = data[i,0]
    v = data[i,4]
    w = data[i,5]
    r1 = data[i,6]
    psi1 = data[i,7]
    r2 = data[i,8]
    psi2 = data[i,9]

    theta_new = normalize_rad(state[2] + w * dt)

    vx = v * np.cos(theta_new)
    vy = v * np.sin(theta_new)

    x_new = state[0] + vx * dt
    y_new = state[1] + vy * dt

    phi1 = normalize_rad(psi1+theta_new-np.pi/2)
    phi2 = normalize_rad(psi2+theta_new-np.pi/2)

    x1_pred.append(l1[0] + r1 * np.sin(phi1))
    y1_pred.append(l1[1] - r1 * np.cos(phi1))

    x2_pred.append(l2[0] + r2 * np.sin(phi2))
    y2_pred.append(l2[1] - r2 * np.cos(phi2))

    state = np.array([x_new, y_new, theta_new])
    
    x_pred.append(state[0])
    y_pred.append(state[1])
    theta_pred.append(state[2])    
    

plt.scatter(data[:,0], x_pred, marker='o', color='red', label='x_pred')
plt.scatter(data[:,0], data[:, 1], marker='o', color='blue', label='x_real')
plt.show()

plt.scatter(data[:,0], y_pred, marker='o', color='red', label='y_pred')
plt.scatter(data[:,0], data[:, 2], marker='o', color='blue', label='y_real')
plt.show()

plt.scatter(data[:,0], theta_pred, marker='o', color='red', label='theta_pred')
plt.scatter(data[:,0], data[:, 3], marker='o', color='blue', label='theta_real')
plt.show()

plt.scatter(x_pred, y_pred, marker='.', color='red', label='pred')
plt.scatter(data[:,1], data[:, 2], marker='.', color='blue', label='real')
plt.scatter(x1_pred, y1_pred, marker='.', color='cyan', label='pred1')
plt.scatter(x2_pred, y2_pred, marker='.', color='green', label='pred2')
plt.legend()
plt.show()