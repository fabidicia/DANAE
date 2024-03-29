##########################################################################################

# Kalman Filter Implementation for MPU-6050 6DOF IMU
# Author: Philip Salmony [pms67@cam.ac.uk]
# Riadaptation

##########################################################################################



################################# Various import #######################################
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from datasets import *
import numpy as np
import pickle
import io
import argparse
from math import sin, cos, tan, pi, atan2, sqrt 
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import matplotlib.pyplot as plt
from random import randint
from sklearn.metrics import mean_squared_error
from utils import plot_tensorboard
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

############################# Code call definition ######################################

parser = argparse.ArgumentParser("script to show i-value of IMU data")
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--path', type=str, default="None")
parser.add_argument('--max_iter', type=str, default="None") #just for plotting
parser.add_argument('--Q', type=float, default=1)   # 0.45
parser.add_argument('--P', type=float, default=1)   # 0.1
args = parser.parse_args()

############################# Filter choice ###########################################

############################# Dataset choice ###########################################
if args.dataset == "oxford":
    args.path = "./data/Oxio_Dataset/slow walking/data1/syn/imu3.csv" if args.path == 'None' else args.path
    imu = OXFDataset(path=args.path)
elif args.dataset == "caves":
    args.path="./data/caves/full_dataset/imu_adis.txt" if args.path == 'None' else args.path
    imu = caves(args.path,noise=True)

############################# Some settings ###########################################

seed = randint(0, 1000)
print("experiment seed: "+str(seed))
exper_path = "./runs/KF_" + args.dataset + "_" +str(seed)+"/"
Path(exper_path).mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(exper_path)

############################### INIZIALIZATION of the variables ####################################

dt = 0.1
A = np.array([[1, -dt, 0, 0, 0, 0],
    	      [0, 1, 0, 0, 0, 0],
              [0, 0, 1, -dt, 0, 0],
              [0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 1, -dt],
              [0, 0, 0, 0, 1, 0]])
B = np.array([[dt, 0, 0],
              [0, 0, 0],
              [0, dt, 0],
              [0, 0, 0],
              [0, 0, dt],
              [0, 0, 0]])

C = np.array([[1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0]])
P = np.eye(6)*args.P
Q = np.eye(6)*args.Q
R = np.eye(3)

state_estimate = np.array([[0], [0], [0], [0], [0], [0]])   # roll, roll bias, pitch, pitch bias, yaw, yaw bias

phi_hat, theta_hat, psi_hat = imu.get_ang_groundt(0) ## INITIALIZE TO TRUE GT VALUES!

phi_kf = []
theta_kf = []
psi_kf = []

phi_gt = []
theta_gt = []
psi_gt = []

phi_acc_list = []
theta_acc_list = []
psi_acc_list = []

p_list = []
q_list = []
r_list = []
ax_list = []
ay_list = []
az_list = []
mx_list = []
my_list = []
mz_list = []

phi_dot_list = []
theta_dot_list = []
psi_dot_list = []

################################# FILTER LOOP #########################################

print("Running...")
args.max_iter = imu.len if args.max_iter == 'None' else int(args.max_iter)

for i in range(args.max_iter):
    # Get gyro and mag measurements
    [p, q, r, ax, ay, az, mx, my, mz] = imu.__getitem__(i)
    # normalize mag readings
    m_norm = sqrt((mx*mx)+(my*my)+(mz*mz))
    mx = (mx/m_norm)
    my = (my/m_norm)
    mz = (mz/m_norm) 
    # Get accelerometer measurements
    [phi_acc, theta_acc, psi_acc] = imu.get_acc_angles(i)
    # Calculate psi on the basis of mag data and phi and theta derived from acc (STILL CALLED ACC FOR EASY READING)
    psi_acc = atan2((-my*cos(phi_hat) + mz*sin(phi_hat)), (mx*cos(theta_hat) + my*sin(theta_hat)*sin(phi_hat) + mz*sin(theta_hat)*cos(phi_hat)))
    # calculate Euler angle derivatives from gyro measurements
    phi_dot = (p + sin(phi_hat) * tan(theta_hat) * q + cos(phi_hat) * tan(theta_hat) * r)
    theta_dot = cos(phi_hat) * q - sin(phi_hat) * r
    psi_dot = (sin(phi_hat) / cos(theta_hat)*q + cos(phi_hat) / cos(theta_hat) * r)
    
    # initialize kf using gyro as external input
    gyro_input = np.array([[phi_dot], [theta_dot], [psi_dot]])
    state_estimate = A.dot(state_estimate) + B.dot(gyro_input)
    P = A.dot(P.dot(np.transpose(A))) + Q
    # get orientation and use it as measurements

    measurement = np.array([[phi_acc], [theta_acc], [psi_acc]])
    y_tilde = measurement - C.dot(state_estimate)
    S = R + C.dot(P.dot(np.transpose(C)))
    K = P.dot(np.transpose(C).dot(np.linalg.inv(S)))
    state_estimate = state_estimate + K.dot(y_tilde)
    P = (np.eye(6) - K.dot(C)).dot(P)


    phi_hat = state_estimate[0][0]
    theta_hat = state_estimate[2][0]
    psi_hat = state_estimate[4][0]

    roll, pitch, yaw = imu.get_ang_groundt(i)

    if args.dataset == "caves":
        psi_hat = 0.0
        yaw = 0.0
        psi_dot = 0.0
    ############################ LIST CREATION ########################################

    phi_kf.append(phi_hat)
    theta_kf.append(theta_hat)
    psi_kf.append(psi_hat)  
    phi_gt.append(roll)
    theta_gt.append(pitch)
    psi_gt.append(yaw)
    # other list appends:
    p_list.append(p)
    q_list.append(q)
    r_list.append(r)
    ax_list.append(ax)
    ay_list.append(ay)
    az_list.append(az)
    mx_list.append(mx)
    my_list.append(my)
    mz_list.append(mz)

    phi_dot_list.append(phi_dot)
    theta_dot_list.append(theta_dot)
    psi_dot_list.append(psi_dot)
    phi_acc_list.append(phi_acc)
    theta_acc_list.append(theta_acc)
    psi_acc_list.append(psi_acc)

################################## ARRAYS CREATION FOR PLOTTING #############################

np_phi_kf = np.asarray(phi_kf)
np_phi_gt = np.asarray(phi_gt)
np_theta_kf = np.asarray(theta_kf)
np_theta_gt = np.asarray(theta_gt)
np_psi_kf = np.asarray(psi_kf)
np_psi_gt = np.asarray(psi_gt)
np_p = np.asarray(p_list)
np_q = np.asarray(q_list)
np_r = np.asarray(r_list)
np_phi_dot = np.asarray(phi_dot_list)
np_theta_dot = np.asarray(theta_dot_list)
np_psi_dot = np.asarray(psi_dot_list)

####################################### STATISTICS ########################################

print("mean deviation phi (gt-kf): %.4f" % np.mean(np.abs((np_phi_gt - np_phi_kf))))
print("mean deviation theta (gt-kf): %.4f" % np.mean(np.abs((np_theta_gt - np_theta_kf))))
print("mean deviation psi (gt-kf): %.4f" % np.mean(np.abs((np_psi_gt - np_psi_kf))))

print("max deviation phi (gt-kf): %.4f" % np.max(np.abs((np_phi_gt - np_phi_kf))))
print("max deviation theta (gt-kf): %.4f" % np.max(np.abs((np_theta_gt - np_theta_kf))))
print("max deviation psi (gt-kf): %.4f" % np.max(np.abs((np_psi_gt - np_psi_kf))))

print("RMS error phi: %.4f" % sqrt(mean_squared_error(np_phi_gt, np_phi_kf)))
print("RMS error theta: %.4f" % sqrt(mean_squared_error(np_theta_gt, np_theta_kf)))
print("RMS error psi: %.4f" % sqrt(mean_squared_error(np_psi_gt, np_psi_kf)))

####################################### DICTIONARY ########################################

dictionary = {
    "phi_kf": np.asarray(phi_kf),
    "phi_gt": np.asarray(phi_gt),
    "theta_kf": np.asarray(theta_kf),
    "theta_gt": np.asarray(theta_gt),
    "psi_kf": np.asarray(psi_kf),
    "psi_gt": np.asarray(psi_gt),
    "p": np.asarray(p_list),
    "q": np.asarray(q_list),
    "r": np.asarray(r_list),
    "ax": np.asarray(ax_list),
    "ay": np.asarray(ay_list),
    "az": np.asarray(az_list),
    "mx": np.asarray(mx_list),
    "my": np.asarray(my_list),
    "mz": np.asarray(mz_list),
    "phi_dot": np.asarray(phi_dot_list),
    "theta_dot": np.asarray(theta_dot_list),
    "psi_dot": np.asarray(psi_dot_list),
    "phi_acc": np.asarray(phi_acc_list),
    "theta_acc": np.asarray(theta_acc_list),
    "psi_acc": np.asarray(psi_acc_list),
}

Path("./preds/").mkdir(parents=True, exist_ok=True)
with open("./preds/" + "dict_" + args.path.split("/")[-3]+"_"+ args.path.split("/")[-1][0:4] + ".pkl", 'wb') as f: pickle.dump(dictionary, f)

####################################### PLOTTING ########################################

times_list = [i for i in range(0, 500)]

plot_tensorboard(writer, [phi_kf, phi_gt], ['b', 'r'], ['phi_kf', 'phi_gt'])
plot_tensorboard(writer, [theta_kf, theta_gt], ['b', 'r'], ['theta_kf', 'theta_gt'])
plot_tensorboard(writer, [psi_kf, psi_gt], ['b', 'r'], ['psi_kf', 'psi_gt'])

writer.close()
