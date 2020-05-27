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

class KFclass:
    def __init__(phi,orient,n_timesteps=10000):
    self.phi = phi
    self.dt = 0.1
    self.n_timesteps = n_timesteps

    # matrice di stato/transizione 3x3
    self.A = np.identity(3) * 20  ##domanda: anche qui si puo' usare sia lista di liste che np.array ??
    self.A = self.A.tolist()

    # matrice di controllo 3x3
    self.B = np.identity(3)  ##domanda: anche qui si puo' usare sia lista di liste che np.array ??
    self.B = self.B.tolist()

    # matrice di osservazioni 6x3
    self.C = [[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1],
              [dt, 0, 0],
              [0, dt, 0],
              [0, 0, dt]]

    # matrice di covarianza dello stato 6x6 (nxn con n numero variabili)
    self.P = np.identity(3) * self.dt
    self.P = self.P.tolist()

    # matrice di covarianza del rumore di processo
    self.Q = np.identity(3) * self.phi * (0.002031, 0.0126643, 0.005517)
    self.Q = self.Q.tolist()

    # matrice di covarianza del rumore di misure/sensori 6x6 (3acc e 3gyr)
    self.R = np.identity(6) * (10^-1) * (2.78, 2.78, 0.002, 2.78, 2.78, 0.02)
    self.R = self.R.tolist()

    # inizializzazione stato iniziale
    self.initial_state_mean = [0, 0, 0]
    self.n_dim_state = 3

    # inizializziamo l'output
    self.phi_angles = []
    self.theta_angles = []
    self.psi_angles = []

    self.phi_angles_gt = []
    self.theta_angles_gt = []
    self.psi_angles_gt = []
    self.times = []

    #_, orient, _, _, _, gt_rot, _ = MyDataset.__getitem__(0)
    self.initial_state_mean = orient

    self.kf = KalmanFilter(transition_matrices=self.A, observation_matrices=self.C,
                  transition_covariance=self.Q, observation_covariance=self.R,
                  initial_state_covariance=self.P, initial_state_mean=self.initial_state_mean)

    self.kfstate = np.zeros((self.n_timesteps, self.n_dim_state)) #probabilmente si puo' usare sia np.array che una lista di liste, ma cosi' è più comodo
    kfcov = np.zeros((self.n_timesteps, self.n_dim_state, self.n_dim_state))
    self.kfstate[0] = self.initial_state_mean
    self.kfcov[0] = self.P
    self.orient_list = []
    self.kfstate_list = []
    self.gt_euler = []
    self.phi_diff_gtkf = []
    self.theta_diff_gtkf = []
    self.psi_diff_gtkf = []

    self.t = 0 # e' un counter rispetto a quante iterazioni abbiamo fatto finora

    def quaternion_to_euler(self,x, y, z, w):
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


    def step(self,times,orient, acc_v, gyr, mag):
        self.t += 1 
        measurements = [acc_v[0],acc_v[1],acc_v[2], gyr[0], gyr[1], gyr[2]]
        self.kfstate[self.t], self.kfcov[self.t] = (self.kf.filter_update(self.kfstate[self.t-1], self.kfcov[self.t-1], measurements))
        return self.kfstate[self.
#        x, y, z, w = gt_rot
#        euler_ang = self.quaternion_to_euler(x, y, z, w)
#        phi_diff = euler_ang[0] - self.kfstate[0]
#        theta_diff = euler_ang[1] - self.kfstate[1]
#        psi_diff = euler_ang[2] - self.kfstate[2]
        # giusto per salvarmi i dati
#        self.orient_list.append(orient)
#        self.kfstate_list.append(self.kfstate[self.t])
#        self.gt_euler.append(euler_ang)
#        self.phi_diff_gtkf.append(phi_diff)
#        self.theta_diff_gtkf.append(theta_diff)
#        self.psi_diff_gtkf.append(psi_diff)


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
