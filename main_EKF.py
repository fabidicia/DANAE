##########################################################################################

# This version is the original LKF used to implement DANAE in which the upgrade to an EKF
# using QUATERNIONS is implemented (or, better, I AM TRYING TO) inspired to:
# https://www.thepoorengineer.com/en/ekf-impl/#EKFimpl

##########################################################################################

# 

################################# Various import #######################################

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import quaternion_EKF as ekf
from datasets import *
import numpy as np
import pickle
import io
import argparse
import pandas as pd
from math import sin, cos, tan, pi, atan2, sqrt 
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import matplotlib.pyplot as plt
from random import randint
from sklearn.metrics import mean_squared_error
from utils import plot_tensorboard,quaternion_to_euler
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from scipy.ndimage.filters import uniform_filter1d
from scipy.signal import butter, lfilter
from scipy.signal import freqs

############################# Code call definition ######################################

parser = argparse.ArgumentParser("script to show i-value of IMU data")
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--path', type=str, default="None")
parser.add_argument('--max_iter', type=str, default="None")
parser.add_argument('--gtpath', type=str)# solo per Aqua dataset

############################# Dataset choice ###########################################

args = parser.parse_args()
if args.dataset == "oxford":
    #faccio una modifica
    args.path = "./data/Oxio_Dataset/slow walking/data1/syn/imu3.csv" if args.path == 'None' else args.path
    imu = OXFDataset(path=args.path) ##IN THIS CASE args.path IS REQUIRED
elif args.dataset == "aqua":
    args.path="./data/Aqualoc/imu_sequence_5.csv" if args.path == 'None' else args.path
    imu = Aqua(args.path)
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

#raw meas.
p_list = []
q_list = []
r_list = []
ax_list = []
ay_list = []
az_list = []
mx_list = []
my_list = []
mz_list = []

#final outputs to be plotted
phi_kf = []
theta_kf = []
psi_kf = []

phi_gt = []
theta_gt = []
psi_gt = []

##intermediate values exclusive to ekf
phi_interm_Gyro = []
phi_interm_AccMag = []
theta_interm_Gyro = []
theta_interm_AccMag = []
psi_interm_Gyro = []
psi_interm_AccMag = []

############################### INIZIALIZATION of the filter ####################################

#in this algorithm, the state estimate (xHat) is provided by the elaboration of quaternion derived from
#the gyro elaborations as state (Ax) combined with the RAW gyro acquisitions as external input (Bu)
#the update provided by the measurements (yHatBar) is obtained by the integration of mag and acc instead
#both the variables (xHat and YHatBar) have to be expressed in quaternion form

ekf_sys = ekf.System()
dt = 0.1

#phi_hat, theta_hat, psi_hat = imu.get_ang_groundt(0) ## INITIALIZE TO TRUE GT VALUES!


################################# FILTER LOOP #########################################

print("Running...")
args.max_iter = imu.len if args.max_iter == 'None' else int(args.max_iter)


for i in range(args.max_iter):
    # Get raw measurements
    [p, q, r, ax, ay, az, mx, my, mz] = imu.__getitem__(i)

    w = (p, q, r)
    a = (ax, ay, az)
    m = (mx, my, mz)
    
    # filter call
    ekf_sys.predict(w, dt)
    psi_hat, theta_hat, phi_hat = ekf_sys.update(a, m)

    ## io qui posso salvarmi le stime intermedie di phi, theta, roll contenute in ekf_sys.xHatBar e yHatBar.
    phi_interm, theta_interm, psi_interm = quaternion_to_euler(ekf_sys.xHatBar[0],ekf_sys.xHatBar[1],ekf_sys.xHatBar[2],ekf_sys.xHatBar[3])
    phi_interm2, theta_interm2, psi_interm2 = quaternion_to_euler(ekf_sys.yHatBar[0],ekf_sys.yHatBar[1],ekf_sys.yHatBar[2],ekf_sys.yHatBar[3])

    if args.dataset == "oxford":
        phi_hat = -phi_hat
        theta_hat = -theta_hat
        phi_interm = -phi_interm
        phi_interm2 = -phi_interm2
        theta_interm = -theta_interm
        theta_interm2 = -theta_interm2    
    #Ground truth
    roll, pitch, yaw = imu.get_ang_groundt(i)

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

    phi_interm_Gyro.append(phi_interm)
    phi_interm_AccMag.append(phi_interm2)
    theta_interm_Gyro.append(theta_interm)
    theta_interm_AccMag.append(theta_interm2)
    psi_interm_Gyro.append(psi_interm)
    psi_interm_AccMag.append(psi_interm2)
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
np_ax = np.asarray(ax_list)
np_ay = np.asarray(ay_list)
np_az = np.asarray(az_list)
np_mx = np.asarray(mx_list)
np_my = np.asarray(my_list)
np_mz = np.asarray(mz_list)

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

############################# Butter Lowpass Filter ###########################################
#https://stackoverflow.com/questions/25191620/creating-lowpass-filter-in-scipy-understanding-methods-and-units
# filter requirements
def butter_lowpass(cutOff, fs, order=5):
    nyq = 0.5 * fs
    normalCutoff = cutOff / nyq
    b, a = butter(order, normalCutoff, btype='low', analog = True)
    return b, a

def butter_lowpass_filter(data, cutOff, fs, order=4):
    b, a = butter_lowpass(cutOff, fs, order=order)
    y = lfilter(b, a, data)
    return y

cutOff = 5 #cutoff frequency in rad/s
fs = 100 #sampling frequency in rad/s
order = 20 #order of filter

phi_kf_fil = butter_lowpass_filter(np_phi_kf, cutOff, fs, order)
theta_kf_fil = butter_lowpass_filter(np_theta_kf, cutOff, fs, order)
psi_kf_fil = butter_lowpass_filter(np_psi_kf, cutOff, fs, order)

# filtered data statistics
print("mean deviation phi (gt-kf_fil): %.4f" % np.mean(np.abs((np_phi_gt - phi_kf_fil))))
print("mean deviation theta (gt-kf_fil): %.4f" % np.mean(np.abs((np_theta_gt - theta_kf_fil))))
print("mean deviation psi (gt-kf_fil): %.4f" % np.mean(np.abs((np_psi_gt - psi_kf_fil))))

print("max deviation phi (gt-kf_fil): %.4f" % np.max(np.abs((np_phi_gt - phi_kf_fil))))
print("max deviation theta (gt-kf_fil): %.4f" % np.max(np.abs((np_theta_gt - theta_kf_fil))))
print("max deviation psi (gt-kf_fil): %.4f" % np.max(np.abs((np_psi_gt - psi_kf_fil))))

print("RMS error phi_fil: %.4f" % sqrt(mean_squared_error(np_phi_gt, phi_kf_fil)))
print("RMS error theta_fil: %.4f" % sqrt(mean_squared_error(np_theta_gt, theta_kf_fil)))
print("RMS error psi_fil: %.4f" % sqrt(mean_squared_error(np_psi_gt, psi_kf_fil)))

########################## Cumulative moving average ##############
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.uniform_filter1d.html
N = 100000
phi_kf_uniform = uniform_filter1d(np_phi_kf, N, mode='nearest')
theta_kf_uniform = uniform_filter1d(np_theta_kf, N, mode='nearest')
psi_kf_uniform = uniform_filter1d(np_psi_kf, N, mode='nearest')

print("mean deviation phi (gt-kf_uniform): %.4f" % np.mean(np.abs((np_phi_gt - phi_kf_uniform))))
print("mean deviation theta (gt-kf_uniform): %.4f" % np.mean(np.abs((np_theta_gt - theta_kf_uniform))))
print("mean deviation psi (gt-kf_uniform): %.4f" % np.mean(np.abs((np_psi_gt - psi_kf_uniform))))

print("max deviation phi (gt-kf_cumulative): %.4f" % np.max(np.abs((np_phi_gt - phi_kf_uniform))))
print("max deviation theta (gt-kf_cumulative): %.4f" % np.max(np.abs((np_theta_gt - theta_kf_uniform))))
print("max deviation psi (gt-kf_cumulative): %.4f" % np.max(np.abs((np_psi_gt - psi_kf_uniform))))

print("RMS error phi_fil: %.4f" % sqrt(mean_squared_error(np_phi_gt, phi_kf_uniform)))
print("RMS error theta_fil: %.4f" % sqrt(mean_squared_error(np_theta_gt, theta_kf_uniform)))
print("RMS error psi_fil: %.4f" % sqrt(mean_squared_error(np_psi_gt, psi_kf_uniform)))

####################################### DICTIONARY ########################################

dictionary = {}
for var in ["phi_kf", "phi_gt", "theta_kf", "theta_gt","psi_kf","psi_gt","p_list","q_list","r_list","ax_list","ay_list","az_list","mx_list","my_list","mz_list","phi_interm_Gyro","phi_interm_AccMag", "theta_interm_Gyro", "theta_interm_AccMag", "psi_interm_Gyro","psi_interm_AccMag","phi_kf_fil","theta_kf_fil","psi_kf_fil","phi_kf_uniform","theta_kf_uniform","psi_kf_uniform"]:
    dictionary[var] = np.asarray(eval(var))
Path("./preds/").mkdir(parents=True, exist_ok=True)
with open("./preds/" + "dict_" + args.path.split("/")[-3]+"_"+ args.path.split("/")[-1][0:4] + ".pkl", 'wb') as f: pickle.dump(dictionary, f)


####################################### PLOTTING ########################################

times_list = [i for i in range(500, args.max_iter)]

plot_tensorboard(writer, [phi_kf, phi_gt], ['b', 'r'], ['phi_kf', 'phi_gt'])
plot_tensorboard(writer, [theta_kf, theta_gt], ['b', 'r'], ['theta_kf', 'theta_gt'])
plot_tensorboard(writer, [psi_kf, psi_gt], ['b', 'r'], ['psi_kf', 'psi_gt'])

writer.close()
