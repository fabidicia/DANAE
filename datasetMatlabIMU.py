import numpy as np
from torch.utils.data import Dataset
import torch
import csv
from math import sin,cos,atan
import scipy.io

class datasetMatlabIMU(Dataset):

    def __init__(self, path="./dataMatrix.mat"):
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