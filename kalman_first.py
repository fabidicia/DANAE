import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.tensorboard import SummaryWriter
import csv
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import chain
#from Class_IMUdata import IMUdata
from datasets import IMUdata
from networks import *
from tqdm import tqdm
import math
import scipy.io
import math
from datasets import datasetMatlabIMU

parser = argparse.ArgumentParser("script to show i-value of IMU data")
parser.add_argument('--folder', type=str, default="/mnt/c/Users/fabia/OneDrive/Desktop/Deep_learning/Oxio_Dataset/handheld/data2/syn/")
parser.add_argument('--past_gt', type=bool, default=False)
# parser.add_argument ('--n', type=int, required=True) # i_th value to display
args = parser.parse_args()

if args.folder == "fabiana":
    args.folder = "/mnt/c/Users/fabia/OneDrive/Desktop/Deep_learning/Oxio_Dataset/handheld/data2/syn/"
elif args.folder == "paolo":
    args.folder = "/home/paolo/datasets/Oxford_Inertial_Odometry_Dataset/handheld/data2/syn/"
elif args.folder == "fabianassh":
    args.folder = "/home/fabianadiciaccio/Datasets/Oxio_Dataset/handheld/data2/syn/"
elif args.folder == "prova":
    args.folder = "/mnt/c/Users/fabia/OneDrive/Desktop"
#else:
   #raise Exception("Are u paolo or fabiana? Write the answer to define the folder :)")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MyDataset = datasetMatlabIMU()
#MyDataset = datasetMatlabIMU(args.folder)
#yDataLoader = DataLoader(MyDataset)

## Matrices 
dt = 0.0010
A=np.matrix([[1,0,0,-dt,0,0],[0,1,0,0,-dt,0],[0,0,1,0,0,-dt],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
B = np.matrix([[dt,0,0],[0,dt,0],[0,0,dt],[0,0,0],[0,0,0],[0,0,0]])
C = np.matrix([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0]])
P = np.identity(6)
Q = np.identity(6) * 100
R = np.matrix([[.1, 0, 0], [0, .1, 0], [0, 0, .1]])
state_estimate = np.transpose([[0, 0, 0, 0, 0, 0]])
## Initialization

phi_angles = []
theta_angles = []
psi_angles = []

phi_angles_gt = []
theta_angles_gt = []
psi_angles_gt = []

times = []

roll_error = []
pitch_error = []
yaw_error = []
def rad_to_grad(orient):
    phi,tetha,psi = orient
    return phi* (180/math.pi), tetha* (180/math.pi), psi* (180/math.pi)

#Get Angles from raw measurements
for i in range( len(MyDataset)):
    time, orient, acc_v, gyr, mag, _, _ = MyDataset.__getitem__(i)
    acc_x, acc_y, acc_z = acc_v
    gyr_x, gyr_y, gyr_z =gyr
    mag_x, mag_y, mag_z =mag
    true_phi, true_theta, true_psi = rad_to_grad(orient)
    if i == 0:
        state_estimate[0] = true_phi
        state_estimate[1] = true_theta
        state_estimate[2] = true_psi

    phi_hat_acc = math.atan2((acc_y),math.sqrt(acc_x**2 + acc_z**2))
    theta_hat_acc = math.atan2((acc_x),math.sqrt(acc_y**2+acc_z**2))
    
    phi_hat = state_estimate.item((0,0))
    theta_hat = state_estimate.item((1,0))
    psi_hat = state_estimate.item((2,0))
    
    psi_hat_mag=math.atan2((-1*mag_y*math.cos(phi_hat)+mag_z*math.sin(phi_hat)), (mag_x*math.cos(theta_hat)+mag_y*math.sin(theta_hat)*math.sin(phi_hat)+mag_z*math.sin(theta_hat)*math.cos(phi_hat)))
    psi_hat_mag -= .0873
    
    phi_dot = gyr_x+math.sin(phi_hat)*math.tan(theta_hat)*gyr_y+math.cos(phi_hat)*math.tan(theta_hat)*gyr_z
    theta_dot = math.cos(phi_hat)*gyr_y - math.sin(phi_hat)*gyr_z
    psi_dot = math.sin(phi_hat)/math.cos(theta_hat)*gyr_y + math.cos(phi_hat)/math.cos(theta_hat)*gyr_z
    delta_angle = np.matrix([[phi_dot],[theta_dot],[psi_dot]])
    
    #predict attitude
    state_estimate = A * state_estimate + B * delta_angle
    P = A*P*np.transpose(A) + Q
    
    #update
    Z = np.matrix([[phi_hat_acc],[theta_hat_acc],[psi_hat_mag]])
    r = Z - C*state_estimate
    S = R + C*P*np.transpose(C)
    K = P*np.transpose(C)*(np.linalg.inv(S)) 
    state_estimate = state_estimate + K*r
    P = (np.identity(6) - (K*C)) * P
    state_estimate_degrees = state_estimate * (180/math.pi)
    phi_angles.append(state_estimate_degrees.item((0,0)))
    theta_angles.append(-1*state_estimate_degrees.item((1,0)))
    psi_angles.append(state_estimate_degrees.item((2,0)))
    roll_error.append(state_estimate_degrees.item((0,0)) - true_psi)
    pitch_error.append(-1 * state_estimate_degrees.item((1,0)) - true_theta)
    yaw_error.append(state_estimate_degrees.item((2,0)) - true_phi)
    #misc append calls
    times.append(i)
    phi_angles_gt.append(true_phi)
    theta_angles_gt.append(true_theta)
    psi_angles_gt.append(true_psi)

plt.figure(1)
phi = np.asarray(phi_angles)
phi_gt = np.asarray(phi_angles_gt)
print(phi_gt.var())
print(phi.var())

#import pdb; pdb.set_trace()
plt.plot(times, phi_angles, phi_angles_gt)
#plt.plot(times, phi_angles, label = 'Filtered Phi')
plt.legend(loc = 'upper right')
plt.title('Phi gt and estimated values')
plt.xlabel('Time (readings)')
plt.ylabel('Angle (degrees)')
plt.savefig('Phi gt and estimated values')
plt.show()



plt.figure(1)
plt.plot(times, roll_error, label = 'Phi error')
#plt.plot(times, phi_angles, label = 'Filtered Phi')
plt.legend(loc = 'upper right')
plt.title('True phi vs Filtered')
plt.xlabel('Time (readings)')
plt.ylabel('Angle (degrees)')
plt.savefig('rollError')
#plt.show()

plt.figure(2)
plt.plot(times, pitch_error, label = 'Theta Error')
#plt.plot(times, theta_angles, label = 'Filtered Theta')
plt.legend(loc = 'upper right')
plt.title('True theta vs Filtered')
plt.xlabel('Time (readings)')
plt.ylabel('Angle (degrees)')
plt.savefig('pitchError')
#plt.show()

plt.figure(3)
plt.plot(times, yaw_error, label = 'Psi Error')
#plt.plot(times, psi_angles, label = 'Filtered Psi')
plt.legend(loc = 'upper right')
plt.title('True psi vs Filtered')
plt.xlabel('Time (readings)')
plt.ylabel('Angle (degrees)')
plt.savefig('yawError')
#plt.show()

