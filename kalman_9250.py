# Kalman Filter Implementation for MPU-6050 6DOF IMU
#
# Author: Philip Salmony [pms67@cam.ac.uk]
# Riadaptation

from datasets import Dataset9250
import numpy as np
from time import time
from math import sin, cos, tan, pi
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path 
import matplotlib.pyplot as plt
import io
from utils import plot_tensorboard
import warnings
from random import randint

warnings.filterwarnings('ignore',category=FutureWarning)

imu = Dataset9250()

seed = randint(0,1000)
print("experiment seed: "+str(seed))
exper_path = "./runs/KF_9250_"+str(seed)+"/"
Path(exper_path).mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(exper_path)

# Initialise matrices and variables
    # Kalman filter
dt = 0.01
A = np.array([[1, -dt, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, -dt],
              [0, 0, 0, 1]])
B = np.array([[dt, 0],
              [0, 0],
              [0, dt],
              [0, 0]])

C = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
P = np.eye(4)
Q = np.eye(4) * 200
R = np.eye(2) * .1

state_estimate = np.array([[0], [0], [0], [0]]) #roll and pitch

phi_hat = 0.0
theta_hat = 0.0

phi_est = []
theta_est = []

phi_orient = []
theta_orient = []

# Calculate accelerometer offsets
N = 1000
phi_offset = 0.0
theta_offset = 0.0

for i in range(N):
    [phi_acc, theta_acc] = imu.get_acc_angles(i)
    phi_offset += phi_acc
    theta_offset += theta_acc

phi_offset = float(phi_offset) / float(N)
theta_offset = float(theta_offset) / float(N)

print("Accelerometer offsets: " + str(phi_offset) + "," + str(theta_offset))


print("Running...")
for i in range(N):
    # Get accelerometer measurements and remove offsets
    [phi_acc, theta_acc] = imu.get_acc_angles(i)
    # phi_acc -= phi_offset
    # theta_acc -= theta_offset

    # Gey gyro measurements and calculate Euler angle derivatives
    [p, q, r, _, _, _, _, _, _] = imu.__getitem__(i)
    p = p - .349
    q = q - .349
    r = r - .349
    phi_dot = p + sin(phi_hat) * tan(theta_hat) * q + cos(phi_hat) * tan(theta_hat) * r
    theta_dot = cos(phi_hat) * q - sin(phi_hat) * r

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

    roll, pitch, _ = imu.get_orient(i)
    phi_orient.append(roll)
    theta_orient.append(pitch)

    # Display results
    # print("Phi: " + str(np.round(phi_hat * 180.0 / pi, 1)) + " Theta: " + str(np.round(theta_hat * 180.0 / pi, 1)))
    # print("Phi: " + str(np.round(roll * 180.0 / pi, 1)) + " Theta: " + str(np.round(pitch * 180.0 / pi, 1)))

    # writer.add_scalar('Phi Angle_Degrees', {'kf_phi': np.round(phi_hat * 180.0 / pi, 1),
    #                                        'orient_phi': np.round(pitch * 180.0 / pi, 1)}, i)
    # writer.add_scalar('Theta Angle_Degrees', {'kf_theta': np.round(theta_hat * 180.0 / pi, 1), 
    #                                          'orient_theta': np.round(pitch * 180.0 / pi, 1)}, i)

#    writer.add_scalar('kf_phi_degrees', np.round(phi_hat * 180.0 / pi, 1), i+1)
#    writer.add_scalar('kf_theta_degrees', np.round(theta_hat * 180.0 / pi, 1), i+1)
#    writer.add_scalar('orient_phi_degrees', np.round(roll * 180.0 / pi, 1), i+1)
#    writer.add_scalar('orient_theta_degrees', np.round(pitch * 180.0 / pi, 1), i+1)

    writer.add_scalar('kf_phi_degrees', phi_hat, i+1)
    writer.add_scalar('kf_theta_degrees', theta_hat, i+1)
    writer.add_scalar('orient_phi_degrees', roll, i+1)
    writer.add_scalar('orient_theta_degrees', pitch, i+1)

times_list = [i for i in range(0, N)]
plot_tensorboard(writer, [phi_est, phi_orient], ['b', 'r'], ['phi_kf', 'phi_orient'])
# plot_tensorboard(writer, [phi_orient], ['r'], ['orient_phi'])
plot_tensorboard(writer, [theta_est, theta_orient], ['b', 'r'], ['theta_kf', 'theta_orient'])
# plot_tensorboard(writer, [theta_orient], ['r'], ['orient_theta'])
writer.close()
