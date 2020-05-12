import numpy as np
from torch.utils.data import Dataset
import torch
import csv
from math import sin,cos,atan

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

        # pose in quaternion format
        rot_x = self.gt_mat[n, 5]
        rot_y = self.gt_mat[n, 6]
        rot_z = self.gt_mat[n, 7]
        rot_w = self.gt_mat[n, 8]

        gyr = gyr_x, gyr_y, gyr_z
        acc_v = acc_x, acc_y, acc_z # è una tupla! non ha parentesi infatti. oppure potrebbe averle tonde
        mag = mag_x, mag_y, mag_z
        orient = roll, pitch, yaw
        gt_transl = transl_x, transl_y, transl_z
        gt_rot = rot_x, rot_y, rot_z, rot_w

        return time, roll, pitch, yaw, gyr_x, gyr_y, gyr_z, acc_x, acc_y, acc_z, mag_x, mag_y, mag_z