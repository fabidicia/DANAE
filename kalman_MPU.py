# Kalman Filter Implementation for MPU-6050 6DOF IMU
#
# Author: Philip Salmony [pms67@cam.ac.uk]
# Riadaptation

from datasets import DatasetMPU9250
import numpy as np
from time import sleep, time
from math import sin, cos, tan, pi

imu = DatasetMPU9250()

sleep_time = 0.01

# Initialise matrices and variables
C = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
P = np.eye(4)
Q = np.eye(4)
R = np.eye(2)

state_estimate = np.array([[0], [0], [0], [0]])

phi_hat = 0.0
theta_hat = 0.0

phi_est = []
theta_est = []
# Calculate accelerometer offsets
N = 100
phi_offset = 0.0
theta_offset = 0.0

for i in range(N):
    [phi_acc, theta_acc] = imu.get_acc_angles(N)
    phi_offset += phi_acc
    theta_offset += theta_acc
    sleep(sleep_time)

phi_offset = float(phi_offset) / float(N)
theta_offset = float(theta_offset) / float(N)

print("Accelerometer offsets: " + str(phi_offset) + "," + str(theta_offset))
sleep(2)

# Measured sampling time
dt = 0.0
start_time = time()  # time derived from IMU or not?

print("Running...")
for i in range(N):

    # Sampling time
    dt = time() - start_time
    start_time = time()

    # Get accelerometer measurements and remove offsets
    [phi_acc, theta_acc] = imu.get_acc_angles(N)
    phi_acc -= phi_offset
    theta_acc -= theta_offset
    
    # Gey gyro measurements and calculate Euler angle derivatives
    [_, _, _, _, p, q, r] = imu.__getitem__(N)
    phi_dot = p + sin(phi_hat) * tan(theta_hat) * q + cos(phi_hat) * tan(theta_hat) * r
    theta_dot = cos(phi_hat) * q - sin(phi_hat) * r

    # Kalman filter
    A = np.array([[1, -dt, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, -dt],
                  [0, 0, 0, 1]])
    B = np.array([[dt, 0],
                 [0, 0],
                 [0, dt],
                 [0, 0]])

    gyro_input = np.array([[phi_dot], [theta_dot]])
    state_estimate = A.dot(state_estimate) + B.dot(gyro_input)
    P = A.dot(P.dot(np.transpose(A))) + Q

    measurement = np.array([[phi_acc], [theta_acc]])
    y_tilde = measurement - C.dot(state_estimate)
    S = R + C.dot(P.dot(np.transpose(C)))
    K = P.dot(np.transpose(C).dot(np.linalg.inv(S)))
    state_estimate = state_estimate + K.dot(y_tilde)
    P = (np.eye(4) - K.dot(C)).dot(P)

    phi_hat = state_estimate[0]
    theta_hat = state_estimate[2]
    phi_est.append(phi_hat)
    theta_est.append(theta_hat)
    # Display results
    print("Phi: " + str([phi_hat * 180.0 / pi, 1]) + " Theta: " + str([theta_hat * 180.0 / pi, 1]))

    sleep(sleep_time)
