import numpy as np
from torch.utils.data import Dataset
import torch
import csv
from math import sin, cos, atan
import scipy.io
import math
from time import sleep

class IMUdata(Dataset):
    def __init__(self, path):
        self.path = path
        with open(self.path+"imu2.csv") as imudata:
            imu_iter = csv.reader(imudata)
            imulist = [line for line in imu_iter]
            self.imu_mat = np.array(imulist)    # ho convertito la lista di liste in una matrice
        with open(self.path+"vi2.csv") as gtdata:
            gt_iter = csv.reader(gtdata)
            gtlist = [line for line in gt_iter]
            self.gt_mat = np.array(gtlist)  # ho convertito la lista di liste in una matrice
        self.len = self.imu_mat.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, n):   # METODO

        # train set
        roll = self.imu_mat[n, 1]
        pitch = self.imu_mat[n, 2]
        yaw = self.imu_mat[n, 3]

        gyr_x = self.imu_mat[n, 4]
        gyr_y = self.imu_mat[n, 5]
        gyr_z = self.imu_mat[n, 6]

        acc_x = self.imu_mat[n, 10] # in questo modo la matrice da leggere con getdata è diventato un attributo
        acc_y = self.imu_mat[n, 11]
        acc_z = self.imu_mat[n, 12]

        mag_x = self.imu_mat[n, 13]
        mag_y = self.imu_mat[n, 14]
        mag_z = self.imu_mat[n, 15]

        time = self.imu_mat[n, 0]

        # ground thruth
        transl_x = self.gt_mat[n, 2]
        transl_y = self.gt_mat[n, 3]
        transl_z = self.gt_mat[n, 4]

        # pose in quaternion to euler
        rot_x = self.gt_mat[n, 5]
        rot_y = self.gt_mat[n, 6]
        rot_z = self.gt_mat[n, 7]
        rot_w = self.gt_mat[n, 8]

        gyr = gyr_x.astype(np.float), gyr_y.astype(np.float), gyr_z.astype(np.float)
        acc_v = acc_x.astype(np.float), acc_y.astype(np.float), acc_z.astype(np.float) # è una tupla! non ha parentesi infatti. oppure potrebbe averle tonde
        mag = mag_x.astype(np.float), mag_y.astype(np.float), mag_z.astype(np.float)
        orient = roll.astype(np.float), pitch.astype(np.float), yaw.astype(np.float)
        gt_transl = transl_x.astype(np.float), transl_y.astype(np.float), transl_z.astype(np.float)
        #gt_rot = phi.astype(np.float), theta.astype(np.float), psi.astype(np.float)
        gt_rot = rot_x.astype(np.float), rot_y.astype(np.float), rot_z.astype(np.float), rot_w.astype(np.float)

        return time, orient, acc_v, gyr, mag, gt_rot, gt_transl # decreta chiusura del metodo, deve essere l'ultima riga

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


class datasetMatlabIMU(Dataset):

    def __init__(self, path="/mnt/c/Users/fabia/OneDrive/Desktop/Deep_Learning/dataMatrix.mat"):
        self.path = path
        with open(self.path) as imudata:
            data = scipy.io.loadmat(path)
        self.orient = data['orient']
        self.gyr_s = data['gyr_s']
        self.mag_s = data['mag_s']
        self.acc_s = data['acc_s']
        self.epoch_acc = data['epoch_acc']
        self.len = self.orient.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, n):   # METODO
        orient = tuple(elem for elem in self.orient[n])
        gyr = tuple(elem for elem in self.gyr_s[n])
        acc_v = tuple(elem for elem in self.acc_s[n])
        mag = tuple(elem for elem in self.mag_s[n])

        time = self.epoch_acc[n, 0] # BOOH non so cosa ci sta qua dentro!
        gt_rot = None
        gt_transl = None
        return time, orient, acc_v, gyr, mag, gt_rot, gt_transl #decreta chiusura del metodo, deve essere l'ultima riga


class DatasetMPU9250(Dataset):
    def __init__(self, path = "./data/Attitude-Estimation/"):
        self.path = path
        with open(self.path+"imu_data.csv") as imudata:
            imu_iter = csv.reader(imudata)
            imulist = [line for line in imu_iter]
            imulist.pop(0) # rimuovo il primo elemento della lista visto che non contiene numeri!
            self.imu_mat = np.array(imulist)    # ho convertito la lista di liste in una matrice
        self.len = self.imu_mat.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, i):   # METODO

        # train set [m/S^2] and [rad/s]
        time = self.imu_mat[i, 0]
        Ax = float(self.imu_mat[i, 1]) / 16384.0
        Ay = float(self.imu_mat[i, 2]) / 16384.0
        Az = float(self.imu_mat[i, 3]) / 16384.0
        Gx = float(self.imu_mat[i, 4]) * math.pi / (180.0 * 131.0)
        Gy = float(self.imu_mat[i, 5]) * math.pi / (180.0 * 131.0)
        Gz = float(self.imu_mat[i, 6]) * math.pi / (180.0 * 131.0)

        return time, Ax, Ay, Az, Gx, Gy, Gz

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
        [_, ax, ay, az, _, _, _] = self.__getitem__(i)
        phi = math.atan2(ay, math.sqrt(ax ** 2.0 + az ** 2.0))
        theta = math.atan2(-ax, math.sqrt(ay ** 2.0 + az ** 2.0))
        return [phi, theta]


class Dataset9250(Dataset):
    def __init__(self, path = "./data/9250/"):
        self.path = path
        with open(self.path+"9250Data.csv") as imudata:
            imu_iter = csv.reader(imudata)
            imulist = [line for line in imu_iter]
            self.imu_mat = np.array(imulist)    # ho convertito la lista di liste in una matrice
        self.len = self.imu_mat.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, i):   # METODO

        # train set [m/S^2] and [rad/s]
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
        roll = float(self.imu_mat[i, 2])
        pitch = float(self.imu_mat[i, 1])
        yaw = float(self.imu_mat[i, 0])
        return roll, pitch, yaw

    def get_gyro_bias(self, N=100):
        bx = 0.0
        by = 0.0
        bz = 0.0
        for i in range(N):
            [gx, gy, gz, _, _, _, _, _, _,] = self.__getitem__(i)
            bx += gx
            by += gy
            bz += gz
        return [bx / float(N), by / float(N), bz / float(N)] 

    def get_acc_angles(self, i):
        [_, _, _, ax, ay, az, _, _, _] = self.__getitem__(i)
        phi = math.atan2(ay, math.sqrt(ax ** 2.0 + az ** 2.0))
        theta = math.atan2(-ax, math.sqrt(ay ** 2.0 + az ** 2.0))
        return [phi, theta]
