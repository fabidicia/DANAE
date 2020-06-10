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


def butter_lowpass(cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order):
    b, a = butter_lowpass(cutoff, fs, order=order)
    fil = filtfilt(b, a, data)
#    fil = lfilter(b, a, data)
    return fil


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
        Mx = float(self.imu_mat[i, 13])
        My = float(self.imu_mat[i, 14])
        Mz = float(self.imu_mat[i, 15])
        return Gx, Gy, Gz, Ax, Ay, Az, Mx, My, Mz

    def get_acc_angles(self, i):
        [_, _, _, ax, ay, az, _, _, _] = self.__getitem__(i)
        phi = math.atan2(ay, math.sqrt(ax ** 2.0 + az ** 2.0))
        theta = math.atan2(-ax, math.sqrt(ay ** 2.0 + az ** 2.0))
        psi = math.atan2(math.sqrt(ax ** 2.0 + ay ** 2.0), az)
        return [phi, theta, psi]

    def get_orient(self, i):   # METODO
        roll = float(self.imu_mat[i, 1]) #* pi / 180.0
        pitch = float(self.imu_mat[i, 2]) # * pi / 180.0
        yaw = float(self.imu_mat[i, 3]) #* pi / 180.0
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
        return roll, pitch, yaw


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
        return roll, pitch, yaw

    def get_acc_angles(self, i):
        [_, _, _, ax, ay, az, _, _, _] = self.__getitem__(i)
        phi = math.atan2(ay, math.sqrt(ax ** 2.0 + az ** 2.0))
        theta = math.atan2(-ax, math.sqrt(ay ** 2.0 + az ** 2.0))
        return [phi, theta]

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
        return [phi, theta]


class Dataset9250(Dataset):
    def __init__(self, path="./data/9250/"):
        self.path = path
        with open(self.path+"9250Data.csv") as imudata:
            imu_iter = csv.reader(imudata)
            imulist = [line for line in imu_iter]
            self.imu_mat = np.array(imulist)    # ho convertito la lista di liste in una matrice
        self.len = self.imu_mat.shape[0]

    def __len__(self):
        return self.len

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

        # train set [m/S^2] and [rad/s]
        roll = float(self.imu_mat[i, 2]) * pi / 180.0
        pitch = float(self.imu_mat[i, 1]) * pi / 180.0
        yaw = float(self.imu_mat[i, 0]) * pi / 180.0
        return roll, pitch, yaw

    def get_gyro_bias(self, N=100):
        bx = 0.0
        by = 0.0
        bz = 0.0
        for i in range(N):
            [gx, gy, gz, _, _, _, _, _, _] = self.__getitem__(i)
            bx += gx
            by += gy
            bz += gz
        return [bx / float(N), by / float(N), bz / float(N)]

    def get_acc_angles(self, i):
        [_, _, _, ax, ay, az, _, _, _] = self.__getitem__(i)
        phi = math.atan2(ay, math.sqrt(ax ** 2.0 + az ** 2.0))
        theta = math.atan2(-ax, math.sqrt(ay ** 2.0 + az ** 2.0))
        return [phi, theta]

    def get_ang_groundt(self, i):   # METODO
        return self.get_orient(i)


class MotherOfIMUdata(Dataset):
    def __init__(self,path,seq_len=10):
       self.imudata = IMUdata(path)
       self.seq_len = seq_len
    def __len__(self):
        return self.imudata.len - self.seq_len - 1 ##DA CONTROLLARE SE SI PUO' ELIMINARE IL -1

    def __getitem__(self,n):
        train_sample = []
        gt_sample = []
        for j in range(self.seq_len):
            time, acc_v, gyr, mag, gt_rot, gt_transl = self.imudata[n+j]
            acc_v = [float(e) for e in acc_v]
            gyr = [float(e) for e in gyr]
            mag = [float(e) for e in mag]
            gt_rot = [float(e) for e in gt_rot]
            gt_transl = [float(e) for e in gt_transl]
            train_sample.append(torch.tensor([acc_v[0],acc_v[1],acc_v[2],gyr[0],gyr[1],gyr[2],mag[0],mag[1],mag[2]]))
            gt_sample.append(torch.tensor([gt_rot[0],gt_rot[1],gt_rot[2],gt_rot[3],gt_transl[0],gt_transl[1],gt_transl[2]]))
        train_sample = torch.stack(train_sample) #size: 10x3
        gt_sample = torch.stack(gt_sample) #size: 10x2
        return train_sample, gt_sample


class SimpleDataset(Dataset):
    def __init__(self,seq_len=10):
        self.train_list, self.gt_list = simple_data()
        self.seq_len = seq_len
        self.len = len(self.train_list) - seq_len -1 #if I have 1000 element of data, the maximum index is 990=1000-10

    def __len__(self):
        return self.len

    def __getitem__(self, n):   # METODO
        train_sample = self.train_list[n:n+self.seq_len]
        gt_sample = self.gt_list[n:n+self.seq_len]
        input_list = [torch.Tensor(elem) for elem in train_sample]
        gt_list = [torch.Tensor(elem) for elem in gt_sample]
        input_list = torch.stack(input_list)
        gt_list = torch.stack(gt_list)
        return input_list, gt_list



class DatasetPhi_gt_kf(Dataset):
    def __init__(self, path = "./data/Oxio_Dataset/",seq_length=10):
        self.path_gt = path + "theta_gt.npy"
        self.path_kf = path + "theta_kf.npy"
        self.phi_gt = np.load(self.path_gt)
        self.phi_kf = np.load(self.path_kf)
        self.len = self.phi_gt.shape[0]-seq_length
        self.seq_length = seq_length
    def __len__(self):
        return self.len

    def __getitem__(self, i,seq_length=10):   # METODO

        phi_gt = self.phi_gt[i:i+self.seq_length]
        phi_kf = self.phi_kf[i:i+self.seq_length]
        return torch.from_numpy(phi_kf).view(1,-1), torch.from_numpy(phi_gt).view(1,-1)
	
class Dataset_pred_for_GAN(Dataset):
    def __init__(self, path = "./data/Oxio_Dataset/",seq_length=10):
        files = glob.glob(path+"*.pkl")
        #files_gt = [file.replace("kf","gt") for file in files_kf]
        self.gt_list = []
        self.kf_list = []
        for i in range(len(files)):
            with open(files[i], "rb") as f: dict = pickle.load(f) 

            self.gt_list.append( dict["theta_gt"].squeeze()) ## I'M ASSUMING I WANNA WORK ONLY WITH THETA 
            self.kf_list.append( dict["theta_kf"].squeeze()) #.squeeze per rimuovere unwanted extra dimensions
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

