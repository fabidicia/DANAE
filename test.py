import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.tensorboard import SummaryWriter
import csv
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import chain
from datasets import IMUdata
from networks import MyLSTM, MyLSTM2
from vio_master.LKF import LKF
from tqdm import tqdm

parser = argparse.ArgumentParser("script to show i-value of IMU data")
parser.add_argument('--folder', type=str, default="/mnt/c/Users/fabia/OneDrive/Desktop/Deep learning/Oxford Inertial Odometry Dataset/handheld/data5/syn/")
parser.add_argument('--arch', type=str, default="MyLSTM")
parser.add_argument('--seed', type=int)
parser.add_argument('--epochs', type=int, default=200)
# parser.add_argument ('--n', type=int, required=True) # i_th value to display
args = parser.parse_args()

exper_path = 'runs/'+args.arch+'_experiment_'+str(args.seed)
writer = SummaryWriter(exper_path)

imudata = IMUdata(args.folder)   # istanza della classe

acc_list = []   # la lista si definisce con le quadre
gyr_list = []
mag_list = []
gt_rotlist = []
gt_transllist = []
X_gt = []
Y_gt = []
X_pred = []
Y_pred = []
past_meas = torch.zeros([10, 1, 3])     # 10 è dim0, 1 è dim1, 3 è dim2
past_meas = past_meas.view(1, 30)
# past_meas = torch.zeros([10])

# creating my LSTM deep model
if args.arch == "MyLSTM":
    model = MyLSTM()
elif args.arch == "MyLSTM2":
    model = MyLSTM2()

# creating my linear kalman filter:
# lkf = LKF()

loss_function = nn.MSELoss()
loss_vector = []
rel_error_v = []
total_loss = 0.0
counter = 0

# import pdb; pdb.set_trace() # mette i breakpoints
# torch.cat(inputs).view(len(inputs), 1, 9) #se come terzo valore metto -1 funziona con tutto perchè chiedo a lui di farlo arbitrariamente

model.load_state_dict(torch.load(exper_path+'/model.pth'))
model.eval()     # #require every time i wanna TEST a model

for i in tqdm(range(0, imudata.len, 1)):
    # hidden = (torch.rand(1, 1, 3), torch.rand(1, 1, 3))
    _, acc, gyr, mag, gt_rot, gt_tran = imudata.__getitem__(i)   # l'_ all'inizio dice che ci andrebbe una variabile ma non mi serve e non la metto. questo perchè bisogna che tutte le chiamate corrispondano a quelle della function __getitem__
    input_list = [acc[0], acc[1], acc[2], gyr[0], gyr[1], gyr[2], mag[0], mag[1], mag[2]]
    input_list = [float(i) for i in input_list]     # quando leggo valori in una lista servono sempre le quadre
    grt_transl = [gt_tran[0], gt_tran[1], gt_tran[2]]
    grt_transl = [float(i) for i in grt_transl]
    input_tensor = torch.Tensor(input_list)
    input_tensor = input_tensor.view(1, 9)
    newinput_tensor = torch.cat([input_tensor, past_meas], -1)    # tensore 1*19
    grt_transltensor = torch.Tensor(grt_transl)  # tensore 1x3
    with torch.no_grad():
        out = model(newinput_tensor)
        loss = loss_function(out.view(1, 1, 3), grt_transltensor.view(1, 1, 3))
        rel_error = (out.view(1, 1, 3) - grt_transltensor.view(1, 1, 3)).abs() / grt_transltensor.view(1, 1, 3)

    # out = torch.Tensor(out)
    X_out = out[0, 0, 0]
    Y_out = out[0, 0, 1]
    X_gt.append(gt_tran[0].item())
    Y_gt.append(gt_tran[1].item())
    X_pred.append(X_out.item())
    Y_pred.append(Y_out.item())

    writer.add_scalar('test loss (point by point)',loss.item(), imudata.len + i)
    # import pdb; pdb.set_trace()
    writer.add_scalar('relative error X(point by point)', rel_error[0, 0, 0].item(), imudata.len + i)
    writer.add_scalar('relative error Y(point by point)', rel_error[0, 0, 1].item(), imudata.len + i)
    writer.add_scalar('relative error Z(point by point)', rel_error[0, 0, 2].item(), imudata.len + i)
    writer.add_scalar('relative error Mean(point by point)',torch.mean(rel_error).item(), imudata.len + i)
    past_meas = torch.roll(past_meas, shifts=3)
    past_meas[0, 0:3] = out[0, 0:3]  # fino al 2ndo
    loss_vector.append(torch.mean(loss).item())
    rel_error_v.append(torch.mean(rel_error).item())

#plt.plot(loss_vector)
#plt.plot(rel_error_v)
import pdb; pdb.set_trace()
X_gt = [float(elem) for elem in X_gt]
Y_gt = [float(elem) for elem in Y_gt]
X_pred = [float(elem) for elem in X_pred]
Y_pred = [float(elem) for elem in Y_pred]


# fig1 = plt.figure()
# ax1 = fig.add_subplot(2, 1, 1)
# plt.plot(np.array(loss_vector), 'r')
# plt.show()
fig = plt.figure()
plt.plot(X_gt, Y_gt, 'r')
plt.plot(X_pred, Y_pred, 'b')
# plt.legend()
plt.show()
# torch.manual_seed(1)    # seed è un numero che si da in input ad una funzione o libreria in modo che sia riproducibile
