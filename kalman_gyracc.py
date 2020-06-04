# Kalman Filter Implementation for MPU-6050 6DOF IMU
#
# Author: Philip Salmony [pms67@cam.ac.uk]
# Riadaptation

from datasets import *
import numpy as np
from time import time
from math import sin, cos, tan, pi
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import matplotlib.pyplot as plt
import io
import argparse
from utils import plot_tensorboard
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from random import randint
import pickle

parser = argparse.ArgumentParser("script to show i-value of IMU data")
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--path', type=str, required=True)

args = parser.parse_args()

if args.dataset == "oxford":
    imu = OXFDataset(path=args.path)
elif args.dataset == "matlab":
    imu = datasetMatlabIMU()
elif args.dataset == "phils":   # not usable since it doesnt have orientation
    imu = DatasetPhils()
elif args.dataset == "novedue":
    imu = Dataset9250()

seed = randint(0, 1000)
print("experiment seed: "+str(seed))
exper_path = "./runs/KF_" + args.dataset + "_" +str(seed)+"/"
Path(exper_path).mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(exper_path)

# Initialise matrices and variables
# Kalman filter
dt = 0.1
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
Q = np.eye(4) 
R = np.eye(2) 

state_estimate = np.array([[0], [0], [0], [0]]) #roll, roll bias, pitch, pitch bias

phi_hat = 0.0
theta_hat = 0.0

phi_kf = []
theta_kf = []

phi_gt = []
theta_gt = []
phi_acc_list = []
theta_acc_list = []
p_list = []
q_list = []
r_list = []
phi_dot_list = []
theta_dot_list = []

# Calculate accelerometer offsets
N = 1000
print("Running...")

for i in range(N):
    # Get accelerometer measurements and remove offsets
    [phi_acc, theta_acc] = imu.get_acc_angles(i)

    # Gey gyro measurements and calculate Euler angle derivatives
    [p, q, r, _, _, _, _, _, _] = imu.__getitem__(i)
    # import pdb; pdb.set_trace()
    p = p   # - .349
    q = q   # - .349
    r = r   # - .349
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
    phi_kf.append(phi_hat)
    theta_kf.append(theta_hat)

    roll, pitch, _ = imu.get_orient(i)
    phi_gt.append(roll)
    theta_gt.append(pitch)
    ### other list appends:
    p_list.append(p)
    q_list.append(q)
    r_list.append(r)
    phi_dot_list.append(phi_dot)
    theta_dot_list.append(theta_dot)
    phi_acc_list.append(phi_acc)
    theta_acc_list.append(theta_acc)

    writer.add_scalar('kf_phi', phi_hat, i+1)
    writer.add_scalar('kf_theta', theta_hat, i+1)
    writer.add_scalar('orient_phi', roll, i+1)
    writer.add_scalar('orient_theta', pitch, i+1)


np_phi_kf = np.asarray(phi_kf)
np_phi_gt = np.asarray(phi_gt)
np_theta_kf = np.asarray(theta_kf)
np_theta_gt = np.asarray(theta_gt)
np_p = np.asarray(p_list)
np_q = np.asarray(q_list)
np_r = np.asarray(r_list)
np_phi_dot = np.asarray(phi_dot_list)
np_theta_dot = np.asarray(theta_dot_list)

dictionary = {
"phi_kf"     : np.asarray(phi_kf),
"phi_gt"     : np.asarray(phi_gt),
"theta_kf"   : np.asarray(theta_kf),
"theta_gt"   : np.asarray(theta_gt),
"p"          : np.asarray(p_list),
"q"          : np.asarray(q_list),
"r"          : np.asarray(r_list),
"phi_dot"    : np.asarray(phi_dot_list),
"theta_dot"  : np.asarray(theta_dot_list),
"phi_acc"    : np.asarray(phi_acc_list),
"theta_acc"  : np.asarray(theta_acc_list)
}

#np.save("./preds/" + "theta_gt_" + args.path.split("/")[-3]+"_"+ args.path.split("/")[-1][0:4] + ".npy", np_theta_gt)
#np.save("./preds/" + "theta_kf_" + args.path.split("/")[-3]+"_"+ args.path.split("/")[-1][0:4] + ".npy", np_theta_kf)
with open("./preds/" + "dict_" + args.path.split("/")[-3]+"_"+ args.path.split("/")[-1][0:4] + ".pkl", 'wb') as f: pickle.dump(dictionary, f)

rel_errors = [abs(i-j)/i*100 for i,j in zip(phi_gt,phi_kf) if abs(i)!=0 ]
rel_errors = np.array([num for num in rel_errors if num == num]) #sporco barbatrucco per scoprire se un numero è NaN!!
print("TRUE_rel_error phi: %.5f" %  rel_errors.mean())
mse = ((np_phi_gt - np_phi_kf)**2).mean(axis=None)
print("mse phi: " + str(mse))

rel_errors = [abs(i-j)/i*100 for i,j in zip(theta_gt,theta_kf) if abs(i)!=0 ]
rel_errors = np.array([num for num in rel_errors if num == num])
print("TRUE_rel_error theta: %.5f" %  rel_errors.mean())
mse = ((np_theta_gt - np_theta_kf)**2).mean(axis=None)
print("mse theta: " + str(mse))

times_list = [i for i in range(0, N)]
plot_tensorboard(writer, [phi_kf, phi_gt], ['b', 'r'], ['phi_kf', 'phi_gt'])
# plot_tensorboard(writer, [phi_gt], ['r'], ['orient_phi'])
plot_tensorboard(writer, [theta_kf, theta_gt], ['b', 'r'], ['theta_kf', 'theta_gt'])
# plot_tensorboard(writer, [theta_gt], ['r'], ['orient_theta'])
writer.close()
