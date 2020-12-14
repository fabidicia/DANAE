#################################################################################################

# All the different datasets together with some useful functions are defined here

#################################################################################################


################################# Various import #######################################
import numpy as np
from torch.utils.data import Dataset
import torch
import csv
from math import sin, cos, atan, pi
from scipy.signal import butter, lfilter, filtfilt, freqz
import scipy.io
import math
from time import sleep
import glob
import pickle
from scipy.interpolate import UnivariateSpline


############################# Functions definition ######################################
def euler_to_quaternion(roll, pitch, yaw):

    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return [qx, qy, qz, qw]


def norm_angle(angle):
    if angle <= 0:
        angle = angle + 2*3.141
    if angle >= 2*3.141:
        angle = angle - 2*3.141
    return angle


def interp_resize(arr,new_length): #https://stackoverflow.com/questions/32724546/numpy-interpolation-to-increase-a-vector-size
    if len(arr.shape) == 1:
        arr = arr[...,None] ##add new fake column
    L = []
    for a in arr.T:
        old_indices = np.arange(0,len(a))
        new_indices = np.linspace(0,len(a)-1,new_length)
        spl = UnivariateSpline(old_indices,a,k=1,s=0)
        new_array = spl(new_indices)
        L.append(new_array)
    result = np.asarray(L).T
    return result

#######################################################################################
# Aqualoc
#######################################################################################
class Aqua(Dataset):
    def __init__(self, path="./data/Aqualoc/imu_sequence_5.csv"):
        self.path = path
        with open(self.path) as imudata:
            imu_iter = csv.reader(imudata)
            imulist = [line for line in imu_iter]
            imulist.pop(0)  # rimuovo il primo elemento della lista visto che non contiene numeri!
        with open(self.path.replace("imu", "mag")) as magdata:
            mag_iter = csv.reader(magdata)
            maglist = [line for line in mag_iter]
            maglist.pop(0)  # rimuovo il primo elemento della lista visto che non contiene numeri!
            self.imu_mat = np.array(imulist)
            self.mag_mat = np.array(maglist)

        self.gtpath = "./data/Aqualoc/archaeo_gt/colmap_traj_sequence_5.txt"
        gt_iter = [x.split(' ') for x in open(self.gtpath).readlines()]
        gtlist = [line for line in gt_iter]
        gtlist.pop(0)  # rimuovo il primo elemento della lista visto che non contiene numeri!
        # ho convertito la lista di liste in una matrice
        self.gt_mat = np.array(gtlist)
        self.gt_mat = interp_resize(self.gt_mat,self.imu_mat.shape[0]) #resizing the original gt_mat to the correct imu_mat size! Missing points are obtained by interpolation
    # ho convertito la lista di liste in una matrice
        self.len = self.imu_mat.shape[0]

    def __len__(self):
        return self.len

    def gettime(self, i):
        time = self.imu_mat[i, 0]
        return time

    def __getitem__(self, i):
        Gx = float(self.imu_mat[i, 1])
        Gy = float(self.imu_mat[i, 2])
        Gz = float(self.imu_mat[i, 3])
        Ax = float(self.imu_mat[i, 4])
        Ay = float(self.imu_mat[i, 5])
        Az = float(self.imu_mat[i, 6])

        Mx = float(self.mag_mat[i, 1])
        My = float(self.mag_mat[i, 2])
        Mz = float(self.mag_mat[i, 3]) 
        return Gx, Gy, Gz, Ax, Ay, Az, Mx, My, Mz

    def get_acc_angles(self, i):
        [_, _, _, ax, ay, az, _, _, _] = self.__getitem__(i)
        phi = math.atan2(ay, math.sqrt(ax ** 2.0 + az ** 2.0))
        theta = math.atan2(-ax, math.sqrt(ay ** 2.0 + az ** 2.0))
        psi = math.atan2(math.sqrt(ax ** 2.0 + ay ** 2.0), az)
        return [phi, theta, psi]

    def quaternion_to_euler(self, x, y, z, w):
        x, y, z, w = float(x), float(y), float(z), float(w)
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

    def get_pl_groundt(self, n):   # METODO
        transl_x = self.gt_mat[n, 1]
        transl_y = self.gt_mat[n, 2]
        transl_z = self.gt_mat[n, 3]
        return transl_x, transl_y, transl_z

    def get_ang_groundt(self, n):   # METODO  
        # pose in quaternion
        x = self.gt_mat[n, 4]
        y = self.gt_mat[n, 5]
        z = self.gt_mat[n, 6]
        w = self.gt_mat[n, 7]
#        pitch, roll, yaw = self.quaternion_to_euler(x, y, z, w) #from Fabiana: change of coordinate w.r.t. usual roll-pitch-yaw!
        roll, pitch, yaw = self.quaternion_to_euler(x, y, z, w) 
        return roll, -pitch, yaw

    def get_quat_groundt(self, n):   # METODO  
        # pose in quaternion
        x = self.gt_mat[n, 4]
        y = self.gt_mat[n, 5]
        z = self.gt_mat[n, 6]
        w = self.gt_mat[n, 7]
        return float(x), float(y), float(z), float(w)


#######################################################################################
#Oxio Dataset
#######################################################################################
class OXFDataset(Dataset):
    def __init__(self, path="./data/Oxio_Dataset/handheld/data3/syn/imu3.csv"):
        self.path = path
        with open(self.path) as imudata:
            imu_iter = csv.reader(imudata)
            imulist = [line for line in imu_iter]
        with open(self.path.replace("imu", "vi")) as gtdata:
            gt_iter = csv.reader(gtdata)
            gtlist = [line for line in gt_iter]

            self.imu_mat = np.array(imulist)    # ho convertito la lista di liste in una matrice
            self.gt_mat = np.array(gtlist)  # ho convertito la lista di liste in una matrice
        self.len = self.imu_mat.shape[0]

    def __len__(self):
        return self.len

    def gettime(self, i):
        time = self.imu_mat[i, 0]
        return time

    def __getitem__(self, i):
        # gyro noise 4mdps/sqrt(Hz)
        Gx = float(self.imu_mat[i, 4])
        Gy = float(self.imu_mat[i, 5])
        Gz = float(self.imu_mat[i, 6])
        # acc noise  100µg/sqrt(Hz)
        gravx = float(self.imu_mat[i, 7])
        gravy = float(self.imu_mat[i, 8])
        gravz = float(self.imu_mat[i, 9])
        accx = float(self.imu_mat[i, 10])
        accy = float(self.imu_mat[i, 11])
        accz = float(self.imu_mat[i, 12])
        # Ax = accx + gravx
        # Ay = accy + gravy
        # Az = accz + gravz
        Ax = gravx - accx
        Ay = gravy - accy
        Az = gravz - accz

        # mag output resolution  0.3µT /LSB
        Mx = float(self.imu_mat[i, 13])  #+ 80.0
        My = float(self.imu_mat[i, 14])  #-10.0
        Mz = float(self.imu_mat[i, 15])  #+100.0
        return Gx, Gy, Gz, Ax, Ay, Az, Mx, My, Mz

    def get_acc_angles(self, i):
        [_, _, _, ax, ay, az, _, _, _] = self.__getitem__(i)
        phi = math.atan2(ay, math.sqrt(ax ** 2.0 + az ** 2.0))
        theta = math.atan2(-ax, math.sqrt(ay ** 2.0 + az ** 2.0))
        psi = math.atan2(math.sqrt(ax ** 2.0 + ay ** 2.0), az)
        return [phi, theta, psi]

#    def get_orient(self, i):   # METODO
#        roll = float(self.imu_mat[i, 1]) 
#        pitch = float(self.imu_mat[i, 2]) 
#        yaw = float(self.imu_mat[i, 3]) 
#        return roll, pitch, yaw

    def quaternion_to_euler(self, x, y, z, w):
        x, y, z, w = float(x), float(y), float(z), float(w)
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

    def get_pl_groundt(self, n):   # METODO
        transl_x = self.gt_mat[n, 2]
        transl_y = self.gt_mat[n, 3]
        transl_z = self.gt_mat[n, 4]
        return transl_x, transl_y, transl_z

    def get_ang_groundt(self, n):   # METODO  
        # pose in quaternion
        x = self.gt_mat[n, 5]
        y = self.gt_mat[n, 6]
        z = self.gt_mat[n, 7]
        w = self.gt_mat[n, 8]
        roll, pitch, yaw = self.quaternion_to_euler(x, y, z, w)
        return roll, pitch, norm_angle(yaw)


    def get_quat_groundt(self, n):   # METODO  
        # pose in quaternion
        x = self.gt_mat[n, 5]
        y = self.gt_mat[n, 6]
        z = self.gt_mat[n, 7]
        w = self.gt_mat[n, 8]
        return float(x), float(y), float(z), float(w)


#######################################################################################
#UCS Dataset
#######################################################################################

class caves(Dataset):
    def __init__(self, path="./data/caves/full_dataset/imu_adis.txt", noise=False):
        self.path = path
        with open(self.path) as imudata:
            imu_iter = csv.reader(imudata)
            imulist = [line for line in imu_iter]
        imulist.pop(0) #rimuovo la prima riga che non contiene letture
        self.imu_mat = np.array(imulist)    # ho convertito la lista di liste in una matrice

        self.len = self.imu_mat.shape[0]
        self.noise = noise
    def __len__(self):
        return self.len

    def gettime(self, i):
        time = self.imu_mat[i, 0]
        return time

    def __getitem__(self, i):
        # gyro noise 4mdps/sqrt(Hz)
        Gx = float(self.imu_mat[i, 17]) if not self.noise  else float(self.imu_mat[i, 17]) + np.random.normal(0,0.5)
        Gy = float(self.imu_mat[i, 18]) if not self.noise  else float(self.imu_mat[i, 18]) + np.random.normal(0,0.5)
        Gz = float(self.imu_mat[i, 19]) if not self.noise  else float(self.imu_mat[i, 19]) + np.random.normal(0,0.5)
        # gyro biases
        bx = float(self.imu_mat[i, 20])
        by = float(self.imu_mat[i, 21])
        bz = float(self.imu_mat[i, 22])
        # acc noise  100µg/sqrt(Hz)
        accx = float(self.imu_mat[i, 14])
        accy = float(self.imu_mat[i, 15])
        accz = float(self.imu_mat[i, 16])

        Ax =  accx if self.noise is False else accx+np.random.normal(0,0.5) 
        Ay =  accy if self.noise is False else accy+np.random.normal(0,0.5) 
        Az =  accz if self.noise is False else accz+np.random.normal(0,0.5) 

        # mag output resolution  0.3µT /LSB
        Mx = float(self.imu_mat[i, 11])
        My = float(self.imu_mat[i, 12])
        Mz = float(self.imu_mat[i, 13])
        return Gx, Gy, Gz, Ax, Ay, Az, Mx, My, Mz

    def get_acc_angles(self, i):
        [_, _, _, ax, ay, az, _, _, _] = self.__getitem__(i)
        phi = math.atan2(ay, math.sqrt(ax ** 2.0 + az ** 2.0))
        theta = math.atan2(-ax, math.sqrt(ay ** 2.0 + az ** 2.0))
        psi = math.atan2(math.sqrt(ax ** 2.0 + ay ** 2.0), az)
        return [phi, theta, psi]

    def get_orient(self, i):   # METODO
        roll = float(self.imu_mat[i, 3]) #* pi / 180.0
        pitch = float(self.imu_mat[i, 4]) # * pi / 180.0
        yaw = float(self.imu_mat[i, 5]) #* pi / 180.0
        return roll, pitch, yaw

    def quaternion_to_euler(self, x, y, z, w):
        x, y, z, w = float(x), float(y), float(z), float(w)
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

    def get_quat_groundt(self, n):   # METODO  
        # pose in quaternion
        x = self.imu_mat[n, 6]
        y = self.imu_mat[n, 7]
        z = self.imu_mat[n, 8]
        w = self.imu_mat[n, 9]
        return float(x), float(y), float(z), float(w)

    def get_ang_groundt(self, n):   # METODO  
        # pose in quaternion
        x, y, z, w = self.get_quat_groundt(n)	
        roll, pitch, yaw = self.quaternion_to_euler(x, y, z, w)
        return roll, pitch, yaw


#######################################################################################
#Dataset collected using Matlab for Phone
#######################################################################################
class datasetMatlabIMU(Dataset):

    def __init__(self, path="./data/Dati_iphone/"):
        self.path = path
        with open(self.path+"PhoneMatrix.csv") as imudata:
            imu_iter = csv.reader(imudata)
            imulist = [line for line in imu_iter]
            self.imu_mat = np.array(imulist)    # ho convertito la lista di liste in una matrice
        self.len = self.imu_mat.shape[0]

    def __len__(self):
        return self.len

    def gettime(self, i):
        time = self.imu_mat[i, 0]
        return time

    def __getitem__(self, i):   # METODO
        Gx = float(self.imu_mat[i, 3])
        Gy = float(self.imu_mat[i, 4])
        Gz = float(self.imu_mat[i, 5])
        Ax = float(self.imu_mat[i, 6])
        Ay = float(self.imu_mat[i, 7])
        Az = float(self.imu_mat[i, 8])
        Mx = float(self.imu_mat[i, 9])
        My = float(self.imu_mat[i, 10])
        Mz = float(self.imu_mat[i, 11])
        return Gx, Gy, Gz, Ax, Ay, Az, Mx, My, Mz

    def get_orient(self, i):   # METODO
        roll = float(self.imu_mat[i, 2]) * pi / 180.0 
        pitch = float(self.imu_mat[i, 1]) * pi / 180.0
        yaw = float(self.imu_mat[i, 0]) * pi / 180.0 
        return pitch, roll, -yaw

    def get_acc_angles(self, i):
        [_, _, _, ax, ay, az, _, _, _] = self.__getitem__(i)
        phi = math.atan2(ay, math.sqrt(ax ** 2.0 + az ** 2.0))
        theta = math.atan2(-ax, math.sqrt(ay ** 2.0 + az ** 2.0))
        psi = math.atan2(math.sqrt(ax ** 2.0 + ay ** 2.0), az)
        return [phi, theta, psi]

    def get_ang_groundt(self, i):   # METODO
        return self.get_orient(i)


class DatasetPhils(Dataset):
    def __init__(self, path="./data/Attitude-Estimation/"):
        self.path = path
        with open(self.path+"imu_data.csv") as imudata:
            imu_iter = csv.reader(imudata)
            imulist = [line for line in imu_iter]
            imulist.pop(0)  # rimuovo il primo elemento della lista visto che non contiene numeri!
            self.imu_mat = np.array(imulist)    # ho convertito la lista di liste in una matrice
        self.len = self.imu_mat.shape[0]

    def __len__(self):
        return self.len

    def gettime(self, n):
        time = self.imu_mat[n, 0]
        return time

    def __getitem__(self, i):   # METODO
        Ax = float(self.imu_mat[i, 1]) / 16384.0
        Ay = float(self.imu_mat[i, 2]) / 16384.0
        Az = float(self.imu_mat[i, 3]) / 16384.0
        Gx = float(self.imu_mat[i, 4]) * math.pi / (180.0 * 131.0)
        Gy = float(self.imu_mat[i, 5]) * math.pi / (180.0 * 131.0)
        Gz = float(self.imu_mat[i, 6]) * math.pi / (180.0 * 131.0)
        Mx = 0
        My = 0
        Mz = 0
        return Gx, Gy, Gz, Ax, Ay, Az, Mx, My, Mz

    def get_gyro_bias(self, N=100):
        bx = 0.0
        by = 0.0
        bz = 0.0
        for i in range(N):
            [_, _, _, _, gx, gy, gz] = self.__getitem__(i)
            bx += gx
            by += gy
            bz += gz
        return [bx / float(N), by / float(N), bz / float(N)] 

    def get_acc_angles(self, i):
        [_, _, _, ax, ay, az, _, _, _] = self.__getitem__(i)
        phi = math.atan2(ay, math.sqrt(ax ** 2.0 + az ** 2.0))
        theta = math.atan2(-ax, math.sqrt(ay ** 2.0 + az ** 2.0))
        psi = math.atan2(math.sqrt(ax ** 2.0 + ay ** 2.0), az)
        return [phi, theta, psi]

#######################################################################################
# GAN Datasets
#######################################################################################
	
class Dataset_pred_for_GAN(Dataset):
    def __init__(self, path = "./data/Oxio_Dataset/",seq_length=10,angle="theta"):
        key_gt = angle+"_gt"
        key_kf = angle+"_kf"
        files = glob.glob(path+"*.pkl")
        self.gt_list = []
        self.kf_list = []
        for i in range(len(files)):
            with open(files[i], "rb") as f: dict = pickle.load(f) 
            self.gt_list.append( dict[key_gt].squeeze()) 
            self.kf_list.append( dict[key_kf].squeeze()) #.squeeze per rimuovere unwanted extra dimensions
        self.gt = np.concatenate(self.gt_list, axis=0)
        self.kf = np.concatenate(self.kf_list, axis=0)
        self.valid_indexes = []
        acc = 0
        for i in range(len(files)):
            mat_indexes = [j for j in range(self.kf_list[i].shape[0]-seq_length)]
            mat_indexes = [elem + acc for elem in mat_indexes]
            self.valid_indexes += mat_indexes #sommare le liste vuol dire fare un append
            acc += self.kf_list[i].shape[0]
 
        self.len = len(self.valid_indexes)
        self.seq_length = seq_length

    def __len__(self):
        return self.len

    def __getitem__(self, i):   # METODO
        true_idx = self.valid_indexes[i] 
        gt = self.gt[true_idx:true_idx+self.seq_length]
        kf = self.kf[true_idx:true_idx+self.seq_length]
        return torch.from_numpy(kf).view(1,-1), torch.from_numpy(gt).view(1,-1)

class Dataset_GAN_2(Dataset):
    def __init__(self, path = "./data/Oxio_Dataset/", seq_length=10, angle="theta"):
        files = glob.glob(path+"*.pkl")
        shape_0 = []
        with open(files[0], "rb") as f: 
            self.dict = pickle.load(f)
        for key in self.dict.keys():
            self.dict[key] = []
        for i in range(len(files)):
            with open(files[i], "rb") as f: f_dict = pickle.load(f) 
            for key in f_dict.keys():
                self.dict[key].append(f_dict[key].squeeze()) 
        shape_0 = [self.dict["theta_gt"][i].shape[0] for i in range(len(files))] #capturing all the .shape[0] of the matrices of the self.dict["theta_gt"] list
        for key in self.dict.keys():
            self.dict[key] = np.concatenate(self.dict[key], axis=0)
        self.valid_indexes = []
        acc = 0
        for i in range(len(files)):
            mat_indexes = [j for j in range(shape_0[i]-seq_length)] #
            mat_indexes = [elem + acc for elem in mat_indexes]
            self.valid_indexes += mat_indexes #sommare le liste vuol dire fare un append
            acc += shape_0[i]
 
        self.len = len(self.valid_indexes)
        self.seq_length = seq_length

    def __len__(self):
        return self.len

    def __getitem__(self, i):   # METODO
        dictionary = dict.fromkeys(self.dict.keys(),None)
        true_idx = self.valid_indexes[i] 
        for key in self.dict.keys():
            dictionary[key] = self.dict[key][true_idx:true_idx+self.seq_length]
            dictionary[key] = torch.from_numpy(dictionary[key]).view(1,-1)
        return dictionary


class Dataset_GAN_3ang(Dataset):
    def __init__(self, path = "./data/Oxio_Dataset/",seq_length=10,phi="phi", theta="theta", psi="psi"):
        files = glob.glob(path+"*.pkl")
        shape_0 = []
        with open(files[0], "rb") as f: 
            self.dict = pickle.load(f)
        for key in self.dict.keys():
            self.dict[key] = []
        for i in range(len(files)):
            with open(files[i], "rb") as f: f_dict = pickle.load(f) 
            for key in f_dict.keys():
                self.dict[key].append(f_dict[key].squeeze()) 
        shape_0 = [self.dict["theta_gt"][i].shape[0] for i in range(len(files))] #capturing all the .shape[0] of the matrices of the self.dict["theta_gt"] list
        for key in self.dict.keys():
            self.dict[key] = np.concatenate(self.dict[key], axis=0)
        self.valid_indexes = []
        acc = 0
        for i in range(len(files)):
            mat_indexes = [j for j in range(shape_0[i]-seq_length)] #
            mat_indexes = [elem + acc for elem in mat_indexes]
            self.valid_indexes += mat_indexes #sommare le liste vuol dire fare un append
            acc += shape_0[i]
 
        self.len = len(self.valid_indexes)
        self.seq_length = seq_length

    def __len__(self):
        return self.len

    def __getitem__(self, i):   # METODO
        dictionary = dict.fromkeys(self.dict.keys(),None)
        true_idx = self.valid_indexes[i] 
        for key in self.dict.keys():
            dictionary[key] = self.dict[key][true_idx:true_idx+self.seq_length]
            dictionary[key] = torch.from_numpy(dictionary[key]).view(1,-1)
        return dictionary


