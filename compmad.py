# this is a riadaptation of the code found in this page:
# https://github.com/MarkSherstan/MPU-6050-9250-I2C-CompFilter/tree/master/9DOF

from datasets import *
from datasets import butter_lowpass, butter_lowpass_filter
import numpy as np
import pickle
import io
import argparse
import math
from math import sin, cos, tan, pi, atan2, sqrt
import quat2eul_functions 
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

phi_mad = []
theta_mad = []
psi_mad = []

gyroRoll = 0
gyroPitch = 0
gyroYaw = 0

roll, pitch, yaw = imu.get_ang_groundt(0) ##INITIALIZE WITH TRUE GT VALUES!
rollRads, pitchRads, yawRads = imu.get_ang_groundt(0) ##INITIALIZE WITH TRUE GT VALUES!

dt = 0.1
tau = 0.5 # fattore moltiplicativo del comp filter (98% affidabilità gyr, 2% acc)

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
    rollRads = (tau)*(rollRads + gx*dt) + (1-tau)*(accRoll)
    pitchRads = (tau)*(pitchRads - gy*dt) + (1-tau)*(accPitch)

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
    yawRads = math.radians(yaw) - 1.75
    phi_kf.append(rollRads)
    theta_kf.append(pitchRads)
    psi_kf.append(yawRads)  # è il nostro complementary

    roll, pitch, yaw = imu.get_ang_groundt(i)
    phi_gt.append(roll)
    theta_gt.append(pitch)
    psi_gt.append(yaw)

    ## madgwickFilter
    # Quaternion values
    q1,q2,q3,q4 = imu.get_quat_groundt(i)

    # Auxiliary variables
    q1x2 = 2 * q1
    q2x2 = 2 * q2
    q3x2 = 2 * q3
    q4x2 = 2 * q4
    q1q3x2 = 2 * q1 * q3
    q3q4x2 = 2 * q3 * q4
    q1q1 = q1 * q1
    q1q2 = q1 * q2
    q1q3 = q1 * q3
    q1q4 = q1 * q4
    q2q2 = q2 * q2
    q2q3 = q2 * q3
    q2q4 = q2 * q4
    q3q3 = q3 * q3
    q3q4 = q3 * q4
    q4q4 = q4 * q4

    # Normalize accelerometer measurement
    norm = math.sqrt(ax * ax + ay * ay + az * az)
    # if norm is 0: return
    ax /= norm
    ay /= norm
    az /= norm

    # Normalize magnetometer measurement
    norm = math.sqrt(mx * mx + my * my + mz * mz)
    # if norm is 0: return
    mx /= norm
    my /= norm
    mz /= norm

    # Reference direction of Earth's magnetic field
    hx = mx * q1q1 - (2*q1*my) * q4 + (2*q1*mz) * q3 + mx * q2q2 + q2x2 * my * q3 + q2x2 * mz * q4 - mx * q3q3 - mx * q4q4
    hy = (2*q1*mx) * q4 + my * q1q1 - (2*q1*mz) * q2 + (2*q2*mx) * q3 - my * q2q2 + my * q3q3 + q3x2 * mz * q4 - my * q4q4
    bx_2 = math.sqrt(hx * hx + hy * hy)
    bz_2 = -(2*q1*mx) * q3 + (2*q1*my) * q2 + mz * q1q1 + (2*q2*mx) * q4 - mz * q2q2 + q3x2 * my * q4 - mz * q3q3 + mz * q4q4
    bx_4 = 2 * bx_2
    bz_4 = 2 * bz_2

    # Gradient descent algorithm corrective step
    s1 = -q3x2 * (2 * q2q4 - q1q3x2 - ax) + q2x2 * (2 * q1q2 + q3q4x2 - ay) - bz_2 * q3 * (bx_2 * (0.5 - q3q3 - q4q4) + bz_2 * (q2q4 - q1q3) - mx) + (-bx_2 * q4 + bz_2 * q2) * (bx_2 * (q2q3 - q1q4) + bz_2 * (q1q2 + q3q4) - my) + bx_2 * q3 * (bx_2 * (q1q3 + q2q4) + bz_2 * (0.5 - q2q2 - q3q3) - mz)
    s2 = q4x2 * (2 * q2q4 - q1q3x2 - ax) + q1x2 * (2 * q1q2 + q3q4x2 - ay) - 4 * q2 * (1 - 2 * q2q2 - 2 * q3q3 - az) + bz_2 * q4 * (bx_2 * (0.5 - q3q3 - q4q4) + bz_2 * (q2q4 - q1q3) - mx) + (bx_2 * q3 + bz_2 * q1) * (bx_2 * (q2q3 - q1q4) + bz_2 * (q1q2 + q3q4) - my) + (bx_2 * q4 - bz_4 * q2) * (bx_2 * (q1q3 + q2q4) + bz_2 * (0.5 - q2q2 - q3q3) - mz)
    s3 = -q1x2 * (2 * q2q4 - q1q3x2 - ax) + q4x2 * (2 * q1q2 + q3q4x2 - ay) - 4 * q3 * (1 - 2 * q2q2 - 2 * q3q3 - az) + (-bx_4 * q3 - bz_2 * q1) * (bx_2 * (0.5 - q3q3 - q4q4) + bz_2 * (q2q4 - q1q3) - mx) + (bx_2 * q2 + bz_2 * q4) * (bx_2 * (q2q3 - q1q4) + bz_2 * (q1q2 + q3q4) - my) + (bx_2 * q1 - bz_4 * q3) * (bx_2 * (q1q3 + q2q4) + bz_2 * (0.5 - q2q2 - q3q3) - mz)
    s4 = q2x2 * (2 * q2q4 - q1q3x2 - ax) + q3x2 * (2 * q1q2 + q3q4x2 - ay) + (-bx_4 * q4 + bz_2 * q2) * (bx_2 * (0.5 - q3q3 - q4q4) + bz_2 * (q2q4 - q1q3) - mx) + (-bx_2 * q1 + bz_2 * q3) * (bx_2 * (q2q3 - q1q4) + bz_2 * (q1q2 + q3q4) - my) + bx_2 * q2 * (bx_2 * (q1q3 + q2q4) + bz_2 * (0.5 - q2q2 - q3q3) - mz)

    # Normalize step magnitude
    norm = math.sqrt(s1 * s1 + s2 * s2 + s3 * s3 + s4 * s4)
    s1 /= norm
    s2 /= norm
    s3 /= norm
    s4 /= norm

    # Compute rate of change of quaternion
    qDot1 = 0.5 * (-q2 * gx - q3 * gy - q4 * gz) - beta * s1
    qDot2 = 0.5 * (q1 * gx + q3 * gz - q4 * gy) - beta * s2
    qDot3 = 0.5 * (q1 * gy - q2 * gz + q4 * gx) - beta * s3
    qDot4 = 0.5 * (q1 * gz + q2 * gy - q3 * gx) - beta * s4

    # Integrate to yield quaternion
    q1 += qDot1 * dt
    q2 += qDot2 * dt
    q3 += qDot3 * dt
    q4 += qDot4 * dt

    # Normalize quaternion
    norm = math.sqrt(q1 * q1 + q2 * q2 + q3 * q3 + q4 * q4)
    q[0] = q1 / norm
    q[1] = q2 / norm
    q[2] = q3 / norm
    q[3] = q4 / norm

    # Get the data from the matrix
    a12 = 2 * (q[1] * q[2] + q[0] * q[3])
    a22 = q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3]
    a31 = 2 * (q[0] * q[1] + q[2] * q[3])
    a32 = 2 * (q[1] * q[3] - q[0] * q[2])
    a33 = q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3]

    # Perform the conversion to euler
    roll = (math.degrees(math.atan2(a31, a33))) * pi/180
    pitch = -math.degrees(math.asin(a32)) * pi/180
    yaw = math.degrees(math.atan2(a12, a22)) * pi/180 * 0.0078

    # Declination 14 deg 7 minutes at Edmonton May 31, 2019. Bound yaw between [0 360]
#    yaw += 14.1
#   if (yaw < 0):
#      yaw += 360

    # Print data
    # print('R: {:<8.1f} P: {:<8.1f} Y: {:<8.1f}'.format(roll,pitch,yaw))
    phi_mad.append(roll)
    theta_mad.append(pitch)
    psi_mad.append(yaw)


# filter requirements
order = 1
fs = 10.0   # sample rate, Hz
cutoff = 0.5  # desired cutoff frequency of the filter, Hz
# filter coefficients
b, a = butter_lowpass(cutoff, fs, order)
# filtering data

np_phi_kf = np.asarray(phi_kf)
np_phi_gt = np.asarray(phi_gt)
np_phi_mad = np.asarray(phi_mad)
np_theta_kf = np.asarray(theta_kf)
np_theta_gt = np.asarray(theta_gt)
np_theta_mad = np.asarray(theta_mad)
np_psi_kf = np.asarray(psi_kf)
np_psi_gt = np.asarray(psi_gt)
np_psi_mad = np.asarray(psi_mad)

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

#madgwick
print("mean deviation phi (gt-mad): %.4f" % np.mean(np.abs((np_phi_gt - np_phi_mad)*180/pi)))
print("mean deviation theta (gt-mad): %.4f" % np.mean(np.abs((np_theta_gt - np_theta_mad)*180/pi)))
print("mean deviation psi (gt-mad): %.4f" % np.mean(np.abs((np_psi_gt - np_psi_mad)*180/pi)))

print("max deviation phi (gt-mad): %.4f" % np.max(np.abs((np_phi_gt - np_phi_mad)*180/pi)))
print("max deviation theta (gt-mad): %.4f" % np.max(np.abs((np_theta_gt - np_theta_mad)*180/pi)))
print("max deviation psi (gt-mad): %.4f" % np.max(np.abs((np_psi_gt - np_psi_mad)*180/pi)))

print("RMS error phi: %.4f" % sqrt(mean_squared_error(np_phi_gt, np_phi_mad)))
print("RMS error theta: %.4f" % sqrt(mean_squared_error(np_theta_gt, np_theta_mad)))
print("RMS error psi: %.4f" % sqrt(mean_squared_error(np_psi_gt, np_psi_mad)))


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
    "psi_kf_fil": np.asarray(psi_kf_fil),
    "phi_mad": np.asarray(phi_mad),
    "theta_mad": np.asarray(theta_mad),
    "psi_mad": np.asarray(psi_mad)
}

Path("./preds/").mkdir(parents=True, exist_ok=True)
with open("./preds/" + "dict_" + args.path.split("/")[-3]+"_"+ args.path.split("/")[-1][0:4] + ".pkl", 'wb') as f: pickle.dump(dictionary, f)
times_list = [i for i in range(0, N)]
plot_tensorboard(writer, [phi_kf, phi_gt, phi_mad], ['b', 'r', 'g'], ['phi_kf', 'phi_gt', 'phi_mad'])
# plot_tensorboard(writer, [phi_gt], ['r'], ['orient_phi'])
plot_tensorboard(writer, [theta_kf, theta_gt, theta_mad], ['b', 'r', 'g'], ['theta_kf', 'theta_gt', 'theta_mad'])
# plot_tensorboard(writer, [theta_gt], ['r'], ['orient_theta'])
plot_tensorboard(writer, [psi_kf, psi_gt, psi_mad], ['b', 'r', 'g'], ['psi_kf', 'psi_gt', 'psi_mad'])
writer.close()
