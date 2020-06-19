from scipy.linalg import block_diag
from ukfm import ATTITUDE as MODEL
import ukfm
import numpy as np
import matplotlib
ukfm.utils.set_matplotlib_config()
from torch.utils.tensorboard import SummaryWriter
from utils import plot_tensorboard
from datasets import *
from scipy.spatial.transform import Rotation 
from random import randint
from pathlib import Path
from math import sqrt
from sklearn.metrics import mean_squared_error
from utils import plot_tensorboard
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


path="./data/caves/full_dataset/imu_adis.txt"
imu = caves(path)

seed = randint(0, 1000)
print("experiment seed: "+str(seed))
exper_path = "./runs/UKFM_" + "caves" + "_" +str(seed)+"/"
Path(exper_path).mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(exper_path)
phi_gt = []
theta_gt = []
psi_gt = []
phi_kf = []
theta_kf = []
psi_kf = []

# sequence time (s)
T = 100
# IMU frequency (Hz)
imu_freq = 100
# create the model
model = MODEL(T, imu_freq)

# IMU noise standard deviation (noise is isotropic)
imu_std = np.array([5/180*np.pi,  # gyro (rad/s)
                    0.4,          # accelerometer (m/s^2)
                    0.2])         # magnetometer
# simulate true trajectory and noisy inputs
states, omegas = model.simu_f(imu_std)
ys = model.simu_h(states, imu_std)

# propagation noise covariance matrix
Q = imu_std[0]**2*np.eye(3)
# measurement noise covariance matrix
R = block_diag(imu_std[1]**2*np.eye(3), imu_std[2]**2*np.eye(3))
# initial uncertainty matrix
P0 = np.zeros((3, 3))  # The state is perfectly initialized
# sigma point parameters
alpha = np.array([1e-3, 1e-3, 1e-3])
state0 = model.STATE(Rot=states[0].Rot)
phi, theta, psi = imu.get_ang_groundt(0)
r = Rotation.from_euler('zyx',[phi, theta, psi]).as_matrix()
state0 = model.STATE(Rot=r)
ukf = ukfm.UKF(state0=state0,
               P0=P0,
               f=model.f,
               h=model.h,
               Q=Q,
               R=R,
               phi=model.phi,
               phi_inv=model.phi_inv,
               alpha=alpha)
# set variables for recording estimates along the full trajectory
ukf_states = [state0]
ukf_Ps = np.zeros((model.N, 3, 3))
ukf_Ps[0] = P0

for i in range(1, model.N):
    [gx, gy, gz, ax, ay, az, mx, my, mz] = imu.__getitem__(i)
    omegas[i-1].gyro = [gx, gy, gz]
    omegas[i-1].states = None
    # propagation
    ukf.propagation(omegas[i-1], 0.1)
    # update
    ukf.update([ax,ay,az,mx,my,mz])
    # save estimates
    ukf_states.append(ukf.state)
    ukf_Ps[i] = ukf.P


    roll, pitch, yaw = imu.get_ang_groundt(i-1)
    phi_gt.append(roll)
    theta_gt.append(pitch)
    psi_gt.append(yaw)

    r = Rotation.from_matrix(ukf.state.Rot).as_euler('zyx')
    phi_kf.append(r[0])
    theta_kf.append(r[1])
    psi_kf.append(r[2])

np_phi_kf = np.asarray(phi_kf)
np_phi_gt = np.asarray(phi_gt)
np_theta_kf = np.asarray(theta_kf)
np_theta_gt = np.asarray(theta_gt)
np_psi_kf = np.asarray(psi_kf)
np_psi_gt = np.asarray(psi_gt)

print("RMS error phi: %.4f" % sqrt(mean_squared_error(np_phi_gt, np_phi_kf)))
print("RMS error theta: %.4f" % sqrt(mean_squared_error(np_theta_gt, np_theta_kf)))
print("RMS error psi: %.4f" % sqrt(mean_squared_error(np_psi_gt, np_psi_kf)))

