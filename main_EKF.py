##########################################################################################

# This version is the original LKF used to implement DANAE in which the upgrade to an EKF
# using QUATERNIONS is implemented (or, better, I AM TRYING TO) inspired to:
# https://www.thepoorengineer.com/en/ekf-impl/#EKFimpl

##########################################################################################

# 

################################# Various import #######################################

import quaternion_EKF as ekf
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
parser.add_argument('--max_iter', type=str, default="None")
parser.add_argument('--gtpath', type=str)# solo per Aqua dataset
parser.add_argument('--Q', type=float, default=1)   # 0.45
parser.add_argument('--P', type=float, default=1)   # 0.1

############################# Dataset choice ###########################################

args = parser.parse_args()
if args.dataset == "oxford":
    #faccio una modifica
    args.path = "./data/Oxio_Dataset/handheld/data3/syn/imu3.csv" if args.path == 'None' else args.path
    imu = OXFDataset(path=args.path) ##IN THIS CASE args.path IS REQUIRED
elif args.dataset == "aqua":
    args.path="./data/Aqualoc/imu_sequence_5.csv" if args.path == 'None' else args.path
    imu = Aqua(args.path)
elif args.dataset == "caves":
    args.path="./data/caves/full_dataset/imu_adis.txt" if args.path == 'None' else args.path
    imu = caves(args.path,noise=True)
elif args.dataset == "matlab":
    imu = datasetMatlabIMU()
elif args.dataset == "phils":   # not usable since it doesnt have orientation
    imu = DatasetPhils()
elif args.dataset == "novedue":
    imu = Dataset9250()

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

#intermediate euler angles estimation

    #obtained from gyro
phi_dot_list = []
theta_dot_list = []
psi_dot_list = []

    #obtained from accelerometer and magnetometer
phi_acc_list = []
theta_acc_list = []
psi_acc_list = []

#final outputs to be plotted
phi_kf = []
theta_kf = []
psi_kf = []

phi_gt = []
theta_gt = []
psi_gt = []


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
    phi_hat = -phi_hat
    theta_hat = -theta_hat
    
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

print("mean deviation phi (gt-kf): %.4f" % np.mean(np.abs((np_phi_gt - np_phi_kf)*pi/180)))
print("mean deviation theta (gt-kf): %.4f" % np.mean(np.abs((np_theta_gt - np_theta_kf)*pi/180)))
print("mean deviation psi (gt-kf): %.4f" % np.mean(np.abs((np_psi_gt - np_psi_kf)*pi/180)))

print("max deviation phi (gt-kf): %.4f" % np.max(np.abs((np_phi_gt - np_phi_kf)*pi/180)))
print("max deviation theta (gt-kf): %.4f" % np.max(np.abs((np_theta_gt - np_theta_kf)*pi/180)))
print("max deviation psi (gt-kf): %.4f" % np.max(np.abs((np_psi_gt - np_psi_kf)*pi/180)))

print("RMS error phi: %.4f" % sqrt(mean_squared_error(np_phi_gt, np_phi_kf)*pi/180))
print("RMS error theta: %.4f" % sqrt(mean_squared_error(np_theta_gt, np_theta_kf)*pi/180))
print("RMS error psi: %.4f" % sqrt(mean_squared_error(np_psi_gt, np_psi_kf)*pi/180))


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
}

Path("./preds/").mkdir(parents=True, exist_ok=True)
with open("./preds/" + "dict_" + args.path.split("/")[-3]+"_"+ args.path.split("/")[-1][0:4] + ".pkl", 'wb') as f: pickle.dump(dictionary, f)


####################################### PLOTTING ########################################

times_list = [i for i in range(500, args.max_iter)]

plot_tensorboard(writer, [phi_kf, phi_gt], ['b', 'r'], ['phi_kf', 'phi_gt'])
plot_tensorboard(writer, [theta_kf, theta_gt], ['b', 'r'], ['theta_kf', 'theta_gt'])
plot_tensorboard(writer, [psi_kf, psi_gt], ['b', 'r'], ['psi_kf', 'psi_gt'])

writer.close()