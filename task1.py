import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("data/data1.txt")

x_real      = data[:,1]
y_real      = data[:,2]
theta_real  = data[:,3]

v_measure       = data[:,4]
w_measure       = data[:,5]
r1_measure      = data[:,6]
bear1_measure   = data[:,7]
r2_measure      = data[:,8]
bear2_measure   = data[:,9]

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


def update(state, P, measurement, beacon, beacon2):
    dt, v, omega, dst_1, bear_1, dst_2, bear_2 = measurement

    # Prediction
    pred = prediction(state, v, omega, dt)
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

    z = np.array([[dst_1],
                  [bear_1],
                  [dst_2],
                  [bear_2]])

    R = np.diag([sigma_distance**2, sigma_bearing**2, sigma_distance**2, sigma_bearing**2])

    b_x, b_y = beacon.reshape(2)
    b2_x, b2_y = beacon2.reshape(2)
    #! Verify this equations
    H = np.array([[(x_pred - b_x)/np.linalg.norm(pred[:2]-beacon),          (y_pred - b_y)/np.linalg.norm(pred[:2]-beacon),         0],
                  [-(y_pred - b_y)/np.linalg.norm(pred[:2]-beacon)**2,      (x_pred - b_x)/np.linalg.norm(pred[:2]-beacon)**2,     -1],
                  [(x_pred - b2_x)/np.linalg.norm(pred[:2]-beacon2),        (y_pred - b2_y)/np.linalg.norm(pred[:2]-beacon2),       0],
                  [-(y_pred - b2_y)/np.linalg.norm(pred[:2]-beacon2)**2,    (x_pred - b2_x)/np.linalg.norm(pred[:2]-beacon2)**2,   -1]])
    
    K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
    P = P_pred - K @ (H @ P_pred @ H.T + R) @ K.T

    distance_1 = np.linalg.norm(pred[:2] - beacon)
    bearing_1 = np.arctan2(b_y-y_pred, b_x-x_pred) - theta_pred

    distance_2 = np.linalg.norm(pred[:2] - beacon2)
    bearing_2 = np.arctan2(b2_y-y_pred, b2_x-x_pred) - theta_pred
    h = np.array([[distance_1],
                  [bearing_1],
                  [distance_2],
                  [bearing_2]])

    x_delta = (z - h)
    x_delta[1] = norm_radians(x_delta[1])
    x_delta[3] = norm_radians(x_delta[3])
    new_state = pred + K @ x_delta
    return new_state, P

estimated_states = np.zeros((np.size(data[:,0]),3,1))

for i in range(1000):

    x, P = update(x, P, 
                  np.array([dt, v_measure[i], 
                           w_measure[i],
                           r1_measure[i],
                           bear1_measure[i],
                           r2_measure[i],
                           bear2_measure[i]]),
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