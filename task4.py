import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

from utils import video
from utils import data

def get_landmark(landmarks, all_landmarks, new_landmark):
    n_landmark = -1
    distances = []

    for landmark in landmarks:
        distance = np.sqrt((landmark[0] - new_landmark[0])** 2 + (landmark[1] - new_landmark[1])** 2)
        distances.append(distance)

    if (distances):
        min_dist = min(distances)
        if min_dist <= 12:
            n_landmark = distances.index(min_dist)

    # Known
    if n_landmark != -1:
        all_landmarks[n_landmark].append(new_landmark)
        x_sum , y_sum = 0, 0
        for landmark in all_landmarks[n_landmark]:
            x_sum += landmark[0]
            y_sum += landmark[1]

        landmarks[n_landmark][0] = x_sum / len(all_landmarks[n_landmark])
        landmarks[n_landmark][1] = y_sum / len(all_landmarks[n_landmark])

    # New
    else:
        landmarks.append(new_landmark)
        all_landmarks.append([new_landmark])

    return landmarks, all_landmarks, n_landmark

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

state =  np.array([[0], [0], [0]])

P = np.eye(3) * 1

sigma_v = 0.5
sigma_w = 0.05

Q = np.array([[sigma_v**2, 0],
              [0,          sigma_w**2]])

sigma_r = 0.5
sigma_psi = 0.1

R = np.array([[sigma_r**2, 0],
              [0,          sigma_psi**2]])

x_pred = []
y_pred = []
theta_pred = []

landmarks = []
all_landmarks = []

P_m = np.zeros((5,5))
F_x = np.zeros((3,3))
F_u = np.zeros((3,2))
state = np.zeros((3,1))

for i in range(len(df)):
    t = df[i,0]
    v = df[i,4]
    w = df[i,5]
    r = df[i,6]
    psi = df[i,7]

    x, y, theta = state[:3].reshape(3)

    state[:3] = np.array([[x + v/w * ( np.sin(theta + dt * w) - np.sin(theta))],
                          [y + v/w * (-np.cos(theta + dt * w) + np.cos(theta))],
                          [data.normalize_rad(theta + w *dt)]])

    x_m, y_m, theta_m = state[:3].reshape(3)


    F_x_aux = np.array([[1, 0, v/w * (np.cos(theta + dt * w) - np.cos(theta))],
                        [0, 1, v/w * (np.sin(theta + dt * w) - np.sin(theta))],
                        [0, 0, 1]])

    F_u_aux = np.array([[( np.sin(theta + dt * w) - np.sin(theta))/w,  v/(w**2) * (np.sin(theta + dt * w) - dt * w * np.cos(theta + dt * w)- np.sin(theta))],
                        [(-np.cos(theta + dt * w) + np.cos(theta))/w, v/(w**2) * (np.cos(theta + dt * w) + dt * w * np.sin(theta + dt * w)- np.cos(theta))],
                        [0, dt]])

    if (r != 0):
        landmark_x = x_m + r * np.cos(theta_m + psi)
        landmark_y = y_m + r * np.sin(theta_m + psi)

        landmarks, all_landmarks, n_landmark = get_landmark(landmarks, all_landmarks, [landmark_x, landmark_y])

        len_landmarks= len(landmarks)

        H_aux = np.array([[ (x_m-landmark_x)/np.sqrt((x_m-landmark_x)**2+(y_m-landmark_y)**2),  (y_m-landmark_y)/np.sqrt((x_m-landmark_x)**2+(y_m-landmark_y)**2), 0],
                        [-(y_m-landmark_y)/((landmark_x-x_m)**2+(y_m-landmark_y)**2),        -(landmark_x-x_m)/((landmark_x-x_m)**2+(y_m-landmark_y)**2),       -1]])

        H = np.zeros((2, (3 + 2*len(landmarks))))
        for line in range(H_aux.shape[0]):
            for column in range(H_aux.shape[1]):
                H[line][column] = H_aux[line][column]
        H[0, 3 + 2*n_landmark] = 1
        H[1, 3 + 2*n_landmark + 1] = 1

        # Known landmark
        if n_landmark != -1:
            landmark_x = landmarks[n_landmark][0]
            landmark_y = landmarks[n_landmark][1]

            z_k = np.array([[r], [psi]])

            P_m = F_x @ P @ F_x.T + F_u @ Q @ F_u.T

            S_k = H @ P_m @ H.T + R

            K_k = P_m @ H.T @ np.linalg.inv(S_k)

            P = P_m - K_k @ S_k @ K_k.T

            h_k = np.array([[np.sqrt((x_m-landmark_x)**2+(y_m-landmark_y)**2)],
                            [data.normalize_rad(np.arctan2(landmark_y-y_m,landmark_x-x_m)-theta_m)]])

            y = z_k - h_k

            y[1] = data.normalize_rad(y[1])

            state = state + (K_k @ y)

        # New landmark
        else:
            F_x = np.eye(3 + 2*len(landmarks), 3 + 2*len(landmarks))
            for line in range(F_x_aux.shape[0]):
                for column in range(F_x_aux.shape[1]):
                    F_x[line][column] = F_x_aux[line][column]

            F_u = np.zeros((3 + 2*len(landmarks), 2))
            for line in range(F_u_aux.shape[0]):
                for column in range(F_u_aux.shape[1]):
                    F_u[line][column] = F_u_aux[line][column]

            P_new = np.eye(P.shape[0] + 2, P.shape[1] + 2) * 1
            for line in range(P.shape[0]):
                for column in range(P.shape[1]):
                    P_new[line][column] = P[line][column]

            P = P_new
            
            P_m = F_x @ P @ F_x.T + F_u @ Q @ F_u.T

            z_k = np.array([[r], [psi]])

            S_k = H @ P_m @ H.T + R

            K_k = P_m @ H.T @ np.linalg.inv(S_k)

            P = P_m - K_k @ S_k @ K_k.T

            h_k = np.array([[np.sqrt((x_m-landmark_x)**2+(y_m-landmark_y)**2)],
                            [data.normalize_rad(np.arctan2(landmark_y-y_m,landmark_x-x_m)-theta_m)]])

            y = z_k - h_k

            y[1] = data.normalize_rad(y[1])

            state = np.vstack((state, [[r], [psi]]))

            state = state + (K_k @ y)

    # No landmark
    else:
        P = F_x @ P @ F_x.T + F_u @ Q @ F_u.T

    x_pred.append(state[0])
    y_pred.append(state[1])
    theta_pred.append(state[2])

    if vid:
        out = video.update(out, t, frame_width, frame_height, state, np.array([df[i,1],df[i,2],df[i,3]]), True, False)

if vid:
    video.export(out)

data.show_comparasion(df[:, 0], x_real, x_pred)
data.show_comparasion(df[:, 0], y_real, y_pred)
data.show_comparasion(df[:, 0], theta_real, theta_pred)

for l in landmarks:
    plt.scatter(l[0], l[1], marker='o', color='green')
plt.scatter(x_real, y_real, marker='.', color='blue', label='real')
plt.scatter(x_pred, y_pred, marker='.', color='red', label='pred')
plt.legend()
plt.show()