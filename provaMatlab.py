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
from datasets import datasetMatlabIMU
from pykalman import KalmanFilter

parser = argparse.ArgumentParser("script to show i-value of IMU data")
parser.add_argument('--folder', type=str, default="/mnt/c/Users/fabia/OneDrive/Desktop/Deep_learning/dataMatrix.mat")
parser.add_argument('--phi', type=float)
args = parser.parse_args()

if args.phi == "none":
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
MyDataset = datasetMatlabIMU()

# measurements = np.asarray([(399,293),(403,299),(409,308),(416,315),(418,318),(420,323),(429,326),(423,328),(429,334),(431,337),(433,342),(434,352),(434,349),(433,350),(431,350),(430,349),(428,347),(427,345),(425,341),(429,338),(431,328),(410,313),(406,306),(402,299),(397,291),(391,294),(376,270),(372,272),(351,248),(336,244),(327,236),(307,220)])

dt = 0.1
# matrice di stato/transizione 3x3
A = [[dt, dt, -dt], ## domanda: anche qui si puo' usare sia lista di liste che np.array ??
    [dt, dt, -dt],
    [dt, dt, -dt]]
# matrice di controllo 6x3
B = [[dt, 0, 0],
      [0, dt, 0],
      [0, 0, dt],
      [dt, 0, 0],
      [0, dt, 0],
      [0, 0, dt]]
# matrice di osservazione 6x3
C = [[dt, 0, 0],
      [0, dt, 0],
      [0, 0, dt],
      [dt, 0, 0],
      [0, dt, 0],
      [0, 0, dt]]
# matrice di covarianza dello stato 6x6 (nxn con n numero variabili)
P = np.identity(3).tolist()
# matrice di covarianza del rumore di processo
Q = np.identity(3) * args.phi
Q = Q.tolist()
# matrice di covarianza del rumore di misure/sensori 6x6 (3acc e 3gyr)
R = [[.1, 0, 0, 0, 0, 0],
      [0, .1, 0, 0, 0, 0], 
      [0, 0, .1, 0, 0, 0],
      [0, 0, 0, .1, 0, 0],
      [0, 0, 0, 0, .1, 0],
      [0, 0, 0, 0, 0, .1] ]

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

_, orient, _, _, _, _, _ = MyDataset.__getitem__(0) #avrei potuto chiamare direttamente orient initial_state_mean, perchè posso mettere qualsiasi
initial_state_mean = orient
kf = KalmanFilter(transition_matrices=A, observation_matrices=C,
                  transition_covariance=Q, observation_covariance=R,
                  initial_state_covariance=P, initial_state_mean=initial_state_mean)

kfstate = np.zeros((n_timesteps, n_dim_state)) #probabilmente si puo' usare sia np.array che una lista di liste, ma cosi' è più comodo
kfcov = np.zeros((n_timesteps, n_dim_state, n_dim_state))
kfstate[0] = initial_state_mean
kfcov[0] = P
orient_list = []
kfstate_list = []
for t in range(1, len(MyDataset)): # il caso t=0 l'ho sviluppato prima del for
    times, orient, acc_v, gyr, mag, _, _ = MyDataset.__getitem__(t)
    measurements = [acc_v[0],acc_v[1],acc_v[2], gyr[0], gyr[1], gyr[2]]
    kfstate[t], kfcov[t] = (kf.filter_update(kfstate[t-1], kfcov[t-1], measurements))

    # giusto per salvarmi i dati
    orient_list.append(orient)
    kfstate_list.append(kfstate[t])

# kf = KalmanFilter.em(measurements, n_iter=5)
# (smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurements)

plt.figure(1)
phi_list = [elem[0] for elem in orient_list]
kf_phi_list = [elem[0] for elem in kfstate_list]

theta_list = [elem[1] for elem in orient_list]
kf_theta_list = [elem[1] for elem in kfstate_list]

psi_list = [elem[2] for elem in orient_list]
kf_psi_list = [elem[2] for elem in kfstate_list]

times_list = [i for i in range(1, n_timesteps)]
plt.plot(times_list, phi_list, 'b',
         times_list, kf_phi_list, 'r')
plt.show()

times_list = [i for i in range(1, n_timesteps)]
plt.plot(times_list, theta_list, 'b',
         times_list, kf_theta_list, 'r')
plt.show()

times_list = [i for i in range(1, n_timesteps)]
plt.plot(times_list, psi_list, 'b',
         times_list, kf_psi_list, 'r')
plt.show()