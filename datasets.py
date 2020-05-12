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

        gyr = gyr_x.astype(np.float), gyr_y.astype(np.float), gyr_z.astype(np.float)
        acc_v = acc_x.astype(np.float), acc_y.astype(np.float), acc_z.astype(np.float) # è una tupla! non ha parentesi infatti. oppure potrebbe averle tonde
        mag = mag_x.astype(np.float), mag_y.astype(np.float), mag_z.astype(np.float)
        orient = roll.astype(np.float), pitch.astype(np.float), yaw.astype(np.float)
        gt_transl = transl_x.astype(np.float), transl_y.astype(np.float), transl_z.astype(np.float)
        gt_rot = rot_x.astype(np.float), rot_y.astype(np.float), rot_z.astype(np.float), rot_w.astype(np.float)

        return time, orient, acc_v, gyr, mag, gt_rot, gt_transl #decreta chiusura del metodo, deve essere l'ultima riga


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
