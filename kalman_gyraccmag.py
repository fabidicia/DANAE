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
parser.add_argument('--gtpath', type=str)# solo per Aqua dataset
parser.add_argument('--Q', type=float, default=1)   # 0.45
parser.add_argument('--P', type=float, default=1)   # 0.1

args = parser.parse_args()

if args.dataset == "oxford":
    imu = OXFDataset(path=args.path)
elif args.dataset == "aqua":
    imu = Aqua()
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

phi_hat = 0.0
theta_hat = 0.0
psi_hat = 0.0

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

phi_dot_list = []
theta_dot_list = []
psi_dot_list = []

N = 169
print("Running...")

for i in range(N):
    # Get gyro and mag measurements
    [p, q, r, _, _, _, mx, my, mz] = imu.__getitem__(i)
    # normalize mag readings
    m_norm = sqrt((mx*mx)+(my*my)+(mz*mz))
    mx = (mx/m_norm)
    my = (my/m_norm)
    mz = (mz/m_norm)
    # import pdb; pdb.set_trace()
    # Get accelerometer measurements
    [phi_acc, theta_acc, psi_acc] = imu.get_acc_angles(i)
    # Calculate psi on the basis of mag data and phi and theta derived from acc (STILL CALLED ACC FOR EASY READING)
    psi_acc = atan2((-my*cos(phi_hat) + mz*sin(phi_hat)), (mx*cos(theta_hat) + my*sin(theta_hat)*sin(phi_hat) + mz*sin(theta_hat)*cos(phi_hat)))
    psi_acc *= .0073    # con questo coefficiente la stima è più veritiera

    # calculate Euler angle derivatives from gyro measurements
    phi_dot = (p + sin(phi_hat) * tan(theta_hat) * q + cos(phi_hat) * tan(theta_hat) * r)
    theta_dot = cos(phi_hat) * q - sin(phi_hat) * r
    psi_dot = (sin(phi_hat) / cos(theta_hat)*q + cos(phi_hat) / cos(theta_hat) * r)
    # psi_dot = psi_hat + (sin(phi_hat) / cos(theta_hat)*q + cos(phi_hat) / cos(theta_hat) * r)
    # NECESSARIO AGGIUNGERE PHI_HAT, THETA_HAT E PSI_HAT? nei risultati non ci sono evidenti cambiamenti

    # initialize kf using gyro as external input
    gyro_input = np.array([[phi_dot], [theta_dot], [psi_dot]])
    state_estimate = A.dot(state_estimate) + B.dot(gyro_input)
    P = A.dot(P.dot(np.transpose(A))) + Q
    # get orientation and use it as measurements
    # roll_or, pitch_or, yaw_or = imu.get_orient(i)
    # measurement = np.array([[yaw_or], [pitch_or], [-roll_or]])
    measurement = np.array([[phi_acc], [theta_acc], [psi_acc]])     # mettendo psi_acc = 0 si ottiene un risultato più veritiero
    y_tilde = measurement - C.dot(state_estimate)
    S = R + C.dot(P.dot(np.transpose(C)))
    K = P.dot(np.transpose(C).dot(np.linalg.inv(S)))
    state_estimate = state_estimate + K.dot(y_tilde)
    P = (np.eye(6) - K.dot(C)).dot(P)

    phi_hat = state_estimate[0][0]
    theta_hat = state_estimate[2][0]
    psi_hat = state_estimate[4][0]
    phi_kf.append(phi_hat)
    theta_kf.append(theta_hat)
    psi_kf.append(-psi_hat)  # qui c'era un moltiplicativo (* .0273)

    roll, pitch, yaw = imu.get_ang_groundt(i)
    phi_gt.append(roll)
    theta_gt.append(pitch)
    psi_gt.append(yaw)
    # other list appends:
    p_list.append(p)
    q_list.append(q)
    r_list.append(r)
    phi_dot_list.append(phi_dot)
    theta_dot_list.append(theta_dot)
    psi_dot_list.append(theta_dot)
    phi_acc_list.append(phi_acc)
    theta_acc_list.append(theta_acc)
    psi_acc_list.append(psi_acc)

#    writer.add_scalar('kf_phi', phi_hat, i+1)
#    writer.add_scalar('kf_theta', theta_hat, i+1)
#    writer.add_scalar('kf_psi', psi_hat, i+1)
#    writer.add_scalar('orient_phi', roll, i+1)
#    writer.add_scalar('orient_theta', pitch, i+1)
#    writer.add_scalar('orient_psi', yaw, i+1)

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

# filter requirements
order = 2
fs = 10.0   # sample rate, Hz
cutoff = 0.667  # desired cutoff frequency of the filter, Hz
# filter coefficients
b, a = butter_lowpass(cutoff, fs, order)
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
    "p": np.asarray(p_list),
    "q": np.asarray(q_list),
    "r": np.asarray(r_list),
    "phi_dot": np.asarray(phi_dot_list),
    "theta_dot": np.asarray(theta_dot_list),
    "psi_dot": np.asarray(psi_dot_list),
    "phi_acc": np.asarray(phi_acc_list),
    "theta_acc": np.asarray(theta_acc_list),
    "psi_acc": np.asarray(psi_acc_list),
    "phi_kf_fil": np.asarray(phi_kf_fil),
    "theta_kf_fil": np.asarray(theta_kf_fil),
    "psi_kf_fil": np.asarray(psi_kf_fil)
}

#np.save("./preds/" + "theta_gt_" + args.path.split("/")[-3]+"_"+ args.path.split("/")[-1][0:4] + ".npy", np_theta_gt)
#np.save("./preds/" + "theta_kf_" + args.path.split("/")[-3]+"_"+ args.path.split("/")[-1][0:4] + ".npy", np_theta_kf)
Path("./preds/").mkdir(parents=True, exist_ok=True)
with open("./preds/" + "dict_" + args.path.split("/")[-3]+"_"+ args.path.split("/")[-1][0:4] + ".pkl", 'wb') as f: pickle.dump(dictionary, f)

#rel_errors = [abs(i-j)/i*100 for i,j in zip(phi_gt,phi_kf) if abs(i)!=0 ]
#rel_errors = np.array([num for num in rel_errors if num == num]) #sporco barbatrucco per scoprire se un numero è NaN!!
#print("TRUE_rel_error phi: %.5f" %  rel_errors.mean())
#mse = ((np_phi_gt - np_phi_kf)**2).mean(axis=None)
#print("mse phi: " + str(mse))

#rel_errors = [abs(i-j)/i*100 for i,j in zip(theta_gt,theta_kf) if abs(i)!=0 ]
#rel_errors = np.array([num for num in rel_errors if num == num])
#print("TRUE_rel_error theta: %.5f" %  rel_errors.mean())
#mse = ((np_theta_gt - np_theta_kf)**2).mean(axis=None)
#print("mse theta: " + str(mse))

times_list = [i for i in range(0, N)]
plot_tensorboard(writer, [phi_kf, phi_gt, phi_kf_fil], ['b', 'r', 'g'], ['phi_kf', 'phi_gt', 'phi_kf_fil'])
# plot_tensorboard(writer, [phi_gt], ['r'], ['orient_phi'])
plot_tensorboard(writer, [theta_kf, theta_gt, theta_kf_fil], ['b', 'r', 'g'], ['theta_kf', 'theta_gt', 'theta_kf_fil'])
# plot_tensorboard(writer, [theta_gt], ['r'], ['orient_theta'])
plot_tensorboard(writer, [psi_kf, theta_gt, psi_kf_fil], ['b', 'r', 'g'], ['psi_kf', 'psi_gt', 'psi_kf_fil'])
writer.close()
