# Kalman Filter Implementation for MPU-6050 6DOF IMU
#
# Author: Philip Salmony [pms67@cam.ac.uk]
# Riadaptation

from datasets import *
from datasets import butter_lowpass, butter_lowpass_filter
import numpy as np
import pickle
import io
import argparse
import math
from math import sin, cos, tan, pi, atan2, sqrt 
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import matplotlib.pyplot as plt
from random import randint
from sklearn.metrics import mean_squared_error
from utils import plot_tensorboard
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

parser = argparse.ArgumentParser("script to show i-value of IMU data")
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--path', type=str, required=True)
parser.add_argument('--Q', type=float, default=1)   # 0.45
parser.add_argument('--P', type=float, default=1)   # 0.1

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

phi_kf = []
theta_kf = []
psi_kf = []

phi_gt = []
theta_gt = []
psi_gt = []

gyroRoll = 0
gyroPitch = 0
gyroYaw = 0

roll = 0
pitch = 0
yaw = 0

dt = 0.1
tau = 0.98 # fattore moltiplicativo del comp filter (98% affidabilità gyr, 2% acc)

q = [1,0,0,0]
beta = 1
N = 1000

for i in range(N):
    # Get the processed values from IMU
    gx, gy, gz, ax, ay, az, mx, my, mz = imu.__getitem__(i)

    # Acceleration vector angle
    accRoll = math.atan2(ay, math.sqrt(ax ** 2.0 + az ** 2.0))
    accPitch = math.atan2(-ax, math.sqrt(ay ** 2.0 + az ** 2.0))

    # Gyro integration angle
    gyroRoll += gx * dt
    gyroPitch -= gy * dt
    gyroYaw += gz * dt

    # Comp filter
    rollRads = (tau)*(roll + gx*dt) + (1-tau)*(accRoll)
    pitchRads = (tau)*(pitch - gy*dt) + (1-tau)*(accPitch)

    # Reassign mag X and Y values in accordance to MPU-9250 
    # codice originale scambia x ed y: prima modifica
    Mx = my
    My = mx
    Mz = mz

    # Normalize the values
    norm = math.sqrt(Mx * Mx + My * My + Mz * Mz)
    Mx1 = Mx / norm
    My1 = My / norm
    Mz1 = Mz / norm

    # Apply tilt compensation
    Mx2 = Mx1*math.cos(pitchRads) + Mz1*math.sin(pitchRads)
    My2 = Mx1*math.sin(rollRads)*math.sin(pitchRads) + My1*math.cos(rollRads) - Mz1*math.sin(rollRads)*math.cos(pitchRads)
    Mz2 = -Mx1*math.cos(rollRads)*math.sin(pitchRads) + My1*math.sin(rollRads) + Mz1*math.cos(rollRads)*math.cos(pitchRads)

    # Heading calculation
    if ((Mx2 > 0) and (My2 >= 0)):
        yaw = math.degrees(math.atan(My2/Mx2))
    elif (Mx2 < 0):
        yaw = 180 + math.degrees(math.atan(My2/Mx2))
    elif ((Mx2 > 0) and (My2 <= 0)):
        yaw = 360 + math.degrees(math.atan(My2/Mx2))
    elif ((Mx2 == 0) and (My2 < 0)):
        yaw = 90
    elif ((Mx2 == 0) and (My2 > 0)):
        yaw = 270
    else:
        print('Error')
    yawRads = math.radians(yaw) *0.0078
    phi_kf.append(rollRads)
    theta_kf.append(pitchRads)
    psi_kf.append(yawRads)  # è il nostro complementary

    roll, pitch, yaw = imu.get_ang_groundt(i)
    phi_gt.append(roll)
    theta_gt.append(pitch)
    psi_gt.append(yaw)

# filter requirements
order = 1
fs = 10.0   # sample rate, Hz
cutoff = 0.5  # desired cutoff frequency of the filter, Hz
# filter coefficients
b, a = butter_lowpass(cutoff, fs, order)
# filtering data

np_phi_kf = np.asarray(phi_kf)
np_phi_gt = np.asarray(phi_gt)
np_theta_kf = np.asarray(theta_kf)
np_theta_gt = np.asarray(theta_gt)
np_psi_kf = np.asarray(psi_kf)
np_psi_gt = np.asarray(psi_gt)

# filtering data
phi_kf_fil = butter_lowpass_filter(np_phi_kf, cutoff, fs, order)
theta_kf_fil = butter_lowpass_filter(np_theta_kf, cutoff, fs, order)
psi_kf_fil = butter_lowpass_filter(np_psi_kf, cutoff, fs, order)


print("mean deviation phi (gt-kf): %.4f" % np.mean(np.abs((np_phi_gt - np_phi_kf)*180/pi)))
print("mean deviation theta (gt-kf): %.4f" % np.mean(np.abs((np_theta_gt - np_theta_kf)*180/pi)))
print("mean deviation psi (gt-kf): %.4f" % np.mean(np.abs((np_psi_gt - np_psi_kf)*180/pi)))

print("max deviation phi (gt-kf): %.4f" % np.max(np.abs((np_phi_gt - np_phi_kf)*180/pi)))
print("max deviation theta (gt-kf): %.4f" % np.max(np.abs((np_theta_gt - np_theta_kf)*180/pi)))
print("max deviation psi (gt-kf): %.4f" % np.max(np.abs((np_psi_gt - np_psi_kf)*180/pi)))

print("RMS error phi: %.4f" % sqrt(mean_squared_error(np_phi_gt, np_phi_kf)))
print("RMS error theta: %.4f" % sqrt(mean_squared_error(np_theta_gt, np_theta_kf)))
print("RMS error psi: %.4f" % sqrt(mean_squared_error(np_psi_gt, np_psi_kf)))

# filtered data
print("mean deviation phi (gt-kf_fil): %.4f" % np.mean(np.abs((np_phi_gt - phi_kf_fil)*180/pi)))
print("mean deviation theta (gt-kf_fil): %.4f" % np.mean(np.abs((np_theta_gt - theta_kf_fil)*180/pi)))
print("mean deviation psi (gt-kf_fil): %.4f" % np.mean(np.abs((np_psi_gt - psi_kf_fil)*180/pi)))

print("max deviation phi (gt-kf_fil): %.4f" % np.max(np.abs((np_phi_gt - phi_kf_fil)*180/pi)))
print("max deviation theta (gt-kf_fil): %.4f" % np.max(np.abs((np_theta_gt - theta_kf_fil)*180/pi)))
print("max deviation psi (gt-kf_fil): %.4f" % np.max(np.abs((np_psi_gt - psi_kf_fil)*180/pi)))

print("RMS error phi_fil: %.4f" % sqrt(mean_squared_error(np_phi_gt, phi_kf_fil)))
print("RMS error theta_fil: %.4f" % sqrt(mean_squared_error(np_theta_gt, theta_kf_fil)))
print("RMS error psi_fil: %.4f" % sqrt(mean_squared_error(np_psi_gt, psi_kf_fil)))

dictionary = {
    "phi_kf": np.asarray(phi_kf),
    "phi_gt": np.asarray(phi_gt),
    "theta_kf": np.asarray(theta_kf),
    "theta_gt": np.asarray(theta_gt),
    "psi_kf": np.asarray(psi_kf),
    "psi_gt": np.asarray(psi_gt),
    "phi_kf_fil": np.asarray(phi_kf_fil),
    "theta_kf_fil": np.asarray(theta_kf_fil),
    "psi_kf_fil": np.asarray(psi_kf_fil)
}

Path("./preds/").mkdir(parents=True, exist_ok=True)
with open("./preds/" + "dict_" + args.path.split("/")[-3]+"_"+ args.path.split("/")[-1][0:4] + ".pkl", 'wb') as f: pickle.dump(dictionary, f)
times_list = [i for i in range(0, N)]
plot_tensorboard(writer, [phi_kf, phi_gt, phi_kf_fil], ['b', 'r', 'g'], ['phi_kf', 'phi_gt', 'phi_kf_fil'])
# plot_tensorboard(writer, [phi_gt], ['r'], ['orient_phi'])
plot_tensorboard(writer, [theta_kf, theta_gt, theta_kf_fil], ['b', 'r', 'g'], ['theta_kf', 'theta_gt', 'theta_kf_fil'])
# plot_tensorboard(writer, [theta_gt], ['r'], ['orient_theta'])
plot_tensorboard(writer, [psi_kf, theta_gt, psi_kf_fil], ['b', 'r', 'g'], ['psi_kf', 'psi_gt', 'psi_kf_fil'])
writer.close()