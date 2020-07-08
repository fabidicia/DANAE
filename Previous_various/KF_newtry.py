# THIS CODE IS BASED ON THESE FINDINGS: http://philsal.co.uk/projects/imu-attitude-estimation

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
from datasets import IMUdata, datasetMatlabIMU
from pykalman import KalmanFilter
import statistics
from math import sin, cos, tan, pi
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter
import io, PIL

parser = argparse.ArgumentParser("script to show i-value of IMU data")
parser.add_argument('--folder', type=str, default="/mnt/c/Users/fabia/OneDrive/Desktop/Deep_learning/Oxio_Dataset/handheld/data2/syn/")
parser.add_argument('--dset_type',default="oxford", help="oxford or matlab")

args = parser.parse_args()


# OXFORD Dataset
if args.folder == "fabiana":
    args.folder = "/mnt/c/Users/fabia/OneDrive/Desktop/Deep_learning/Oxio_Dataset/handheld/data2/syn/"
elif args.folder == "paolo":
    args.folder = "/home/paolo/datasets/Oxford_Inertial_Odometry_Dataset/handheld/data2/syn/"
elif args.folder == "fabianassh":
    args.folder = "/home/fabianadiciaccio/Datasets/Oxio_Dataset/handheld/data2/syn/"
elif args.folder == "prova":
    args.folder = "/mnt/c/Users/fabia/OneDrive/Desktop"

if args.dset_type == "matlab":
   args.folder = "./dataMatrix.mat"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter("./runs/KF/")
MyDataset = IMUdata(args.folder) if args.dset_type == "oxford" else datasetMatlabIMU(args.folder)

dt = 0.1
## il mio vettore di stato X è composto da 4 elementi: phi, bias phi, tetha, bias tetha
## il mio vettore u è dato dalle velocità angolari phi e tetha misurate dal giroscopio
# il mio vettore di misurazione è dato dalle  phi e tetha stimati dall' accelerometro

# matrice di stato/transizione 4x4
A = [[1, -dt, 0,   0],
     [0,   1, 0,   0],
     [0,   0, 1, -dt],
     [0,   0, 0,   1]]   


# matrice C di osservazioni 2x4 IN ALTRI PAPERS SI CHIAMA H
C = [
     [1, 0, 0, 0],
     [0, 0, 1, 0]]


# initial state covariance matrix  4x4 (nxn con n numero variabili)
P = np.identity(4) * dt
P = P.tolist()
# matrice di covarianza del rumore di processo
Q = np.identity(4) 
Q = Q.tolist()
# matrice di covarianza del rumore di misure/sensori 2x2 
R = np.identity(2)  
R = R.tolist()
# creazione stato iniziale
initial_state_mean = [0, 0, 0, 0]
_, orient, _, gyr, _, gt_rot, _ = MyDataset.__getitem__(0) 
initial_state_mean[0] = orient[0] ##DA CONTROLLARE
initial_state_mean[2] = orient[1]
initial_offsets = np.array([dt*gyr[0],0, dt*gyr[1],0]) #the offset b is equal to B*u
n_dim_state = 4
n_timesteps = len(MyDataset)

# inizializziamo l'output
phi_angles = []
theta_angles = []

phi_angles_gt = []
theta_angles_gt = []
times = []


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
    return [roll, pitch, yaw]


kf = KalmanFilter(transition_matrices=A, observation_matrices=C,transition_offsets=initial_offsets,
                  transition_covariance=Q,
                  initial_state_covariance=P, initial_state_mean=initial_state_mean,n_dim_obs=2,n_dim_state=n_dim_state)

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
phi_diff_gtkf_rel = []
theta_diff_gtkf_rel = []
psi_diff_gtkf_rel = []

for t in range(1, len(MyDataset)): # il caso t=0 l'ho sviluppato prima del for
    times, orient, acc_v, gyr, mag, gt_rot, _ = MyDataset.__getitem__(t)
    x, y, z, w = gt_rot
    euler_ang = quaternion_to_euler(x, y, z, w)
    ## trying transformation to intertial coordinates:
    phi_hat = kfstate[t-1,0]
    theta_hat = kfstate[t-1,2]
    p,q,r = gyr
    phi_dot = p + sin(phi_hat) * tan(theta_hat) * q + cos(phi_hat) * tan(theta_hat) * r
    theta_dot = cos(phi_hat) * q - sin(phi_hat) * r
    ###
    measurements = [acc_v[0],acc_v[1]]
    offsets = np.array([dt*phi_dot,0,dt*theta_dot,0])
    kfstate[t], kfcov[t] = kf.filter_update(kfstate[t-1], kfcov[t-1], measurements,transition_offset=offsets)

    phi_diff = euler_ang[0] - kfstate[t,0]
    theta_diff = euler_ang[1] - kfstate[t,2]

    phi_diff_rel = phi_diff / euler_ang[0]
    theta_diff_rel = theta_diff / euler_ang[1]

    # giusto per salvarmi i dati
    orient_list.append(orient)
    kfstate_list.append(kfstate[t])
    gt_euler.append(euler_ang)
    phi_diff_gtkf.append(phi_diff)
    theta_diff_gtkf.append(theta_diff)

    phi_diff_gtkf_rel.append(phi_diff_rel)
    theta_diff_gtkf_rel.append(theta_diff_rel)
    ## tensorboard logging
    writer.add_scalar('orient_phi', orient[0], t)
    writer.add_scalar('KF_phi', kfstate[t,0], t)
    writer.add_scalar('gt_phi', euler_ang[0], t)
# kf = KalmanFilter.em(measurements, n_iter=5)
# (smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurements)

phi_list = [elem[0] for elem in orient_list]
kf_phi_list = [elem[0] for elem in kfstate_list]
gt_phi_list = [elem[0] for elem in gt_euler]
phi_diff_gtkf_list = [elem for elem in phi_diff_gtkf]
phi_diff_gtkf_list_rel = [elem for elem in phi_diff_gtkf_rel]

print("Mean difference between GT phi and KF phi is % s " % (mean(phi_diff_gtkf_list)))
print("Mean rel error between GT phi and KF phi is % s " % (mean(phi_diff_gtkf_list_rel)))

theta_list = [elem[1] for elem in orient_list]
kf_theta_list = [elem[1] for elem in kfstate_list]
gt_theta_list = [elem[1] for elem in gt_euler]
theta_diff_gtkf_list = [elem for elem in theta_diff_gtkf]
theta_diff_gtkf_list_rel = [elem for elem in theta_diff_gtkf_rel]
print("Mean difference between GT theta and KF theta is % s " % (mean(theta_diff_gtkf_list)))
print("Mean rel error between GT theta and KF theta is % s " % (mean(theta_diff_gtkf_list_rel)))


times_list = [i for i in range(1, n_timesteps)]

plt.figure(1)
plt.plot(times_list, phi_list, 'b',
         times_list, kf_phi_list, 'r',
         times_list, gt_phi_list, 'g')
plt.show()


## code to plot the image in tensorboard
plt.title("test")
buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
image = PIL.Image.open(buf)
image = ToTensor()(image)
writer.add_image('Image', image, 0)
writer.close()
