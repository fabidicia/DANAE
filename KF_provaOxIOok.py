##
from numpy import *
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.tensorboard import SummaryWriter
import csv
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import chain
from datasets import IMUdata
from networks import *
from tqdm import tqdm
import math
import scipy.io
import math
from datasets import IMUdata
from pykalman import KalmanFilter
import statistics

parser = argparse.ArgumentParser("script to show i-value of IMU data")
parser.add_argument('--folder', type=str, default="/mnt/c/Users/fabia/OneDrive/Desktop/Deep_learning/Oxio_Dataset/handheld/data2/syn/")
parser.add_argument('--phi', type=float, required=True)
args = parser.parse_args()

if args.phi is None:
  raise Exception("You must define the spectral density by which multiply the Q matrix")

# OXFORD Dataset
if args.folder == "fabiana":
    args.folder = "/mnt/c/Users/fabia/OneDrive/Desktop/Deep_learning/Oxio_Dataset/handheld/data2/syn/"
elif args.folder == "paolo":
    args.folder = "/home/paolo/datasets/Oxford_Inertial_Odometry_Dataset/handheld/data2/syn/"
elif args.folder == "fabianassh":
    args.folder = "/home/fabianadiciaccio/Datasets/Oxio_Dataset/handheld/data2/syn/"
elif args.folder == "prova":
    args.folder = "/mnt/c/Users/fabia/OneDrive/Desktop"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MyDataset = IMUdata(args.folder)
dt = 0.1

# matrice di stato/transizione 3x3
A = np.identity(3) * 20  ##domanda: anche qui si puo' usare sia lista di liste che np.array ??
A = A.tolist()
# matrice di controllo 6x3
B = np.identity(3)  ##domanda: anche qui si puo' usare sia lista di liste che np.array ??
B = B.tolist()

# matrice di osservazioni 6x3
C = [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1],
     [dt, 0, 0],
     [0, dt, 0],
     [0, 0, dt]]

# matrice di covarianza dello stato 6x6 (nxn con n numero variabili)
P = np.identity(3) * dt
P = P.tolist()
# matrice di covarianza del rumore di processo
Q = np.identity(3) * args.phi * (0.002031, 0.0126643, 0.005517)
Q = Q.tolist()
# matrice di covarianza del rumore di misure/sensori 6x6 (3acc e 3gyr)
R = np.identity(6) * (10^-1) * (2.78, 2.78, 0.002, 2.78, 2.78, 0.02)
R = R.tolist()
# inizializzazione stato iniziale
initial_state_mean = [0, 0, 0]
n_dim_state = 3
n_timesteps = len(MyDataset)

# inizializziamo l'output
phi_angles = []
theta_angles = []
psi_angles = []

phi_angles_gt = []
theta_angles_gt = []
psi_angles_gt = []
times = []

_, orient, _, _, _, gt_rot, _ = MyDataset.__getitem__(0) #avrei potuto chiamare direttamente orient initial_state_mean, perchè posso mettere qualsiasi
initial_state_mean = orient

# import pdb; pdb.set_trace()
def quaternion_to_euler(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return [yaw, pitch, roll]


kf = KalmanFilter(transition_matrices=A, observation_matrices=C,
                  transition_covariance=Q, observation_covariance=R,
                  initial_state_covariance=P, initial_state_mean=initial_state_mean)

kfstate = np.zeros((n_timesteps, n_dim_state)) #probabilmente si puo' usare sia np.array che una lista di liste, ma cosi' è più comodo
kfcov = np.zeros((n_timesteps, n_dim_state, n_dim_state))
kfstate[0] = initial_state_mean
kfcov[0] = P
orient_list = []
kfstate_list = []
gt_euler = []
phi_diff_gtkf = []
theta_diff_gtkf = []
psi_diff_gtkf = []

for t in range(1, len(MyDataset)): # il caso t=0 l'ho sviluppato prima del for
    times, orient, acc_v, gyr, mag, gt_rot, _ = MyDataset.__getitem__(t)
    x, y, z, w = gt_rot
    euler_ang = quaternion_to_euler(x, y, z, w)
    measurements = [acc_v[0],acc_v[1],acc_v[2], gyr[0], gyr[1], gyr[2]]
    kfstate[t], kfcov[t] = (kf.filter_update(kfstate[t-1], kfcov[t-1], measurements))

    phi_diff = euler_ang[0] - kfstate[0]
    theta_diff = euler_ang[1] - kfstate[1]
    psi_diff = euler_ang[2] - kfstate[2]
    # giusto per salvarmi i dati
    orient_list.append(orient)
    kfstate_list.append(kfstate[t])
    gt_euler.append(euler_ang)
    phi_diff_gtkf.append(phi_diff)
    theta_diff_gtkf.append(theta_diff)
    psi_diff_gtkf.append(psi_diff)

# kf = KalmanFilter.em(measurements, n_iter=5)
# (smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurements)

phi_list = [elem[2] for elem in orient_list]
kf_phi_list = [elem[0] for elem in kfstate_list]
gt_phi_list = [elem[0] for elem in gt_euler]
phi_diff_gtkf_list = [elem for elem in phi_diff_gtkf]

print("Standard Deviation of phi_list is % s " % (statistics.stdev(phi_list)))
print("Standard Deviation of kf_phi_list is % s " % (statistics.stdev(kf_phi_list)))
print("Standard Deviation of gt_phi_list is % s " % (statistics.stdev(gt_phi_list)))
print("Mean difference between GT phi and KF phi is % s " % (mean(phi_diff_gtkf_list)))

theta_list = [elem[1] for elem in orient_list]
kf_theta_list = [elem[1] for elem in kfstate_list]
gt_theta_list = [elem[1] for elem in gt_euler]
theta_diff_gtkf_list = [elem for elem in theta_diff_gtkf]
print("Standard Deviation of theta is % s " % (statistics.stdev(theta_list)))
print("Standard Deviation of kf_theta is % s " % (statistics.stdev(kf_theta_list)))
print("Standard Deviation of gt_theta is % s " % (statistics.stdev(gt_theta_list)))
print("Mean difference between GT theta and KF theta is % s " % (mean(theta_diff_gtkf_list)))

psi_list = [elem[0] for elem in orient_list]
kf_psi_list = [elem[2] for elem in kfstate_list]
gt_psi_list = [elem[2] for elem in gt_euler]
psi_diff_gtkf_list = [elem for elem in psi_diff_gtkf]
print("Standard Deviation of psi is % s " % (statistics.stdev(psi_list)))
print("Standard Deviation of kf_psi is % s " % (statistics.stdev(kf_psi_list)))
print("Standard Deviation of gt_psi is % s " % (statistics.stdev(gt_psi_list)))
print("Mean difference between GT psi and KF psi is % s " % (mean(psi_diff_gtkf_list)))

times_list = [i for i in range(1, n_timesteps)]

plt.figure(1)
plt.plot(times_list, phi_list, 'b',
         times_list, kf_phi_list, 'r',
         times_list, gt_phi_list, 'g')
plt.show()

plt.figure(2)
plt.plot(times_list, theta_list, 'b',
         times_list, kf_theta_list, 'r',
         times_list, gt_theta_list, 'g')
plt.show()

plt.figure(3)
plt.plot(times_list, psi_list, 'b',
         times_list, kf_psi_list, 'r',
         times_list, gt_psi_list, 'g')
plt.show()
