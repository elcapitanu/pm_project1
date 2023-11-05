from math import asin
import numpy as np
import matplotlib.pyplot as plt
from pyrsistent import b
from sklearn.metrics import balanced_accuracy_score
from zmq import PROBE_ROUTER

data = np.loadtxt("data/data3.txt")

x_real      = data[:,1]
y_real      = data[:,2]
theta_real  = data[:,3]

v_measure       = data[:,4]
w_measure       = data[:,5]
r_measure      	= data[:,6]
bear_measure   	= data[:,7]

n_samples = len(x_real)

# Define constants
dt = data[1][0] - data[0][0]    # Time step

# Define beacon positions
beacon1 = np.array([[0], [0]])
beacon2 = np.array([[10], [0]])

# Initial robot pose
# [x, y, theta, beacon1_x, beacon1_y, beacon2_x, beacon2_y]
x = np.array([[x_real[0]],
              [y_real[0]], 
              [theta_real[0]]])  

P = np.array([[1, 1, 1],
              [1, 1, 1],
              [1, 1, 1]])

# Measurement noise
sigma_v         = 0.5
sigma_omega     = 0.05
sigma_distance  = 0.5
sigma_bearing   = 0.1

Q = np.diag([sigma_v**2, sigma_omega**2])


def norm_radians(ang):
    ang = ang % (2*np.pi)
    if ang > np.pi :
        ang -= 2*np.pi
    return ang

# state prediction
def prediction(state, v, w, dt):
    x = state[0] + v/w * (np.sin(state[2] + dt*w) - np.sin(state[2]))
    y = state[1] + v/w * (np.cos(state[2]) - np.cos(state[2] + dt*w))
    theta = norm_radians(state[2] + w*dt)
    return np.array([x, y, theta])

iter = 0

def getBeacon(state, b1, b2, distance, bearing):

    dst_1 = np.linalg.norm(state[:2]-b1)
    dst_2 = np.linalg.norm(state[:2]-b2)

    bearing_1 = np.arctan2(b1[1]-state[1], b1[0]-state[0]) - state[2]
    bearing_1 = norm_radians(bearing_1)

    bearing_2 = np.arctan2(b2[1]-state[1], b2[0]-state[0]) - state[2]
    bearing_2 = norm_radians(bearing_2)

    # err_bear_1 = abs(bearing - bearing_1)
    # err_bear_2 = abs(bearing - bearing_2)

    # err_dst_1 = abs(distance - dst_1)
    # err_dst_2 = abs(distance - dst_2)

    print(f"e1 = {bearing_1} and e2 = {bearing_2}")
    print(f"d1 = {dst_1}  and d2 = {dst_2}")

    if dst_1 < dst_2 :
        # check if beacon 1 is in sight
        if abs(bearing_1) <= np.pi/4 :
            print(f"[{iter}] is b1")
            return b1
        # it must be reading from beacon  2 then
        else :
            print(f"[{iter}] is b2")
            return b2

    if dst_2 < dst_1 :
        # check if beacon 2 is in sight
        if abs(bearing_2) <= np.pi/4 :
            print(f"[{iter}] is b2")
            return b2
        # it must be reading from beacon  1 then
        else :
            print(f"[{iter}] is b1")
            return b1

    print(f"error ??")
    return b2

def step(dst, bear, x, y, theta, beacon):
    
    b_x, b_y = beacon.reshape(2)

    z = np.array([[dst, bear]])
    R = np.diag([sigma_distance**2, sigma_bearing**2])

    norm = np.sqrt((x-b_x)**2 + (y-b_x)**2)
    H = np.array([[(x-b_x)/norm,
                   (y-b_y)/norm,
                   0]])

    H = np.append(H, np.array([[(b_y-y)/(norm**2),
                                (b_x-x)/(norm**2),
                                -1]]), axis=0)

    h = np.array([[norm,
                   np.arctan2(b_y-y, b_x-x)-theta]])


    delta = (z-h).reshape(z.size, 1)
    return delta, R, H

def update(state, P, measurement, beacon_1, beacon_2):
    dt, v, omega, dst, bear = measurement

    # Prediction
    pred = prediction(state, v, omega, dt)
    # print(f"pred : {pred}")
    x, y, theta = state.reshape(3)
    x_pred, y_pred, theta_pred = pred.reshape(3)

    # Jacobians
    F_x = np.array([[1, 0, v/omega * (np.cos(theta_pred) - np.cos(theta))],
                    [0, 1, v/omega * (np.sin(theta_pred) - np.sin(theta))],
                    [0, 0, 1]])

    F_u = np.array([[(np.sin(theta_pred) - np.sin(theta))/omega, v/(omega**2) * (np.sin(theta_pred) - dt*omega*np.cos(theta_pred) - np.sin(theta))],
                    [(np.cos(theta) - np.cos(theta_pred))/omega, v/(omega**2) * (np.cos(theta_pred) + dt*omega*np.sin(theta_pred) - np.cos(theta))],
                    [0, dt]])

    P_pred = F_x @ P @ F_x.T + F_u @ Q @ F_u.T
    # print(F_x)
    # print(F_u)
    # print(P_pred)
    
    if (dst != 0) and (abs(bear) <= np.pi/4) :
        beacon_seen = getBeacon(state, beacon_1, beacon_2, dst, bear)
        state_delta, R, H = step(dst, bear, x_pred, y_pred, theta_pred, beacon_seen)

        K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
        P = P_pred - K @ (H @ P_pred @ H.T + R) @ K.T

        new_state = pred + K @ state_delta
        return new_state, P

    # no beacons in sight
    return pred, P_pred

estimated_states = np.zeros((np.size(data[:,0]),3,1))

for i in range(50):
    iter = i
    x, P = update(x, P, 
                  np.array([dt, v_measure[i], 
                           w_measure[i],
                           r_measure[i],
                           bear_measure[i]]),
                  beacon1,
                  beacon2)

    estimated_states[i] = x

plt.figure(figsize=(12, 8))
plt.subplot(321)
plt.plot(data[:, 0], data[:, 1], label='True x', linestyle='--')
plt.plot(data[:, 0], estimated_states[:,0], label='Estimated x', marker='.')
plt.legend(loc = 'lower left')

plt.subplot(323)
plt.plot(data[:, 0], data[:, 2], label='True y', linestyle='--')
plt.plot(data[:, 0], estimated_states[:,1], label='Estimated y', marker='.')
plt.legend(loc = 'lower left')

plt.subplot(325)
plt.plot(data[:, 0], data[:, 3], label='True θ', linestyle='--')
plt.plot(data[:, 0], estimated_states[:,2], label='Estimated θ', marker='.')
plt.legend(loc = 'lower left')

plt.subplot(122, aspect = 'equal')
plt.plot(data[:,1], data[:,2], label='True Mapping', marker='.')
plt.plot(estimated_states[:,0], estimated_states[:,1], label='Estimated Mapping', marker='.')
plt.plot(beacon1[0], beacon1[1], label='Beacon 1', marker='o')
plt.plot(beacon2[0], beacon2[1], label='Beacon 2', marker='o')
plt.legend(loc = 'upper left')

plt.tight_layout()
plt.show()