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

if args.folder == "fabiana":
    args.folder = "/mnt/c/Users/fabia/OneDrive/Desktop/Deep learning/Oxford Inertial Odometry Dataset/handheld/data5/syn/"
elif args.folder == "paolo":
    args.folder = "/home/paolo/datasets/Oxford_Inertial_Odometry_Dataset/handheld/data5/syn/"
else:
    raise Exception("Are u paolo or fabiana? Write the answer to define the folder :)")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
exper_path = 'runs/'+args.arch+'_experiment_'+str(args.seed)
writer = SummaryWriter(exper_path)
MyDataset = MotherOfIMUdata(args.folder,SEQ_LEN)
MyDataLoader = DataLoader(MyDataset, batch_size=args.batch_size,shuffle=True, num_workers=1)

# creating my LSTM deep model
if args.arch == "MyLSTM":
    model = MyLSTM(device=device,n_inputs=33)
elif args.arch == "MyLSTM2":
    model = MyLSTM2()
elif args.arch == "MyLSTMCell":
    model = MyLSTMCell()
elif args.arch == "YOLO_LSTM":
    model = YOLO_LSTM(input_dim=INPUT_DIM, output_size=OUT_DIM, hidden_dim=args.hidden_dim, n_layers=2, drop_prob=0.0)
model = model.to(device)    # casting the model to the correct device, cpu or gpu

loss_function = nn.MSELoss()
loss_vector = []
total_loss = 0.0
total_rel_error = 0.0
print_freq = 9000
X_gt = []
Y_gt = []
X_pred = []
Y_pred = []

# import pdb; pdb.set_trace() # mette i breakpoints
# torch.cat(inputs).view(len(inputs), 1, 9) #se come terzo valore metto -1 funziona con tutto perchè chiedo a lui di farlo arbitrariamente

model.load_state_dict(torch.load(exper_path+'/model.pth'))
model.eval()     # #require every time i wanna TEST a model

for i, (input_tensor, gt_tensor) in tqdm(enumerate(MyDataLoader)):
    input_tensor = input_tensor.to(device)
    gt_tensor = gt_tensor.to(device)

    with torch.no_grad():
        out = model(input_tensor)
        rel_error = ((out.view(-1, SEQ_LEN, OUT_DIM) - gt_tensor.view(-1, SEQ_LEN, OUT_DIM)).abs() / gt_tensor.view(-1, SEQ_LEN, OUT_DIM).abs()) #now it has BATCH_SIZE x SEQ_LEN x OUT_DIM shape
        total_rel_error += torch.mean(rel_error,dim=[0,1]) # now it has OUT_DIM shape
        loss = loss_function(out.view(-1, SEQ_LEN, OUT_DIM), gt_tensor.view(-1, SEQ_LEN, OUT_DIM))

    X_out = out[0, 0, 0]
    Y_out = out[0, 0, 1]
    X_gt.append(gt_tran[0].item())
    Y_gt.append(gt_tran[1].item())
    X_pred.append(X_out.item())
    Y_pred.append(Y_out.item())

    writer.add_scalar('test loss (point by point)',loss.item(), MyDataLoader.len + i)
    # import pdb; pdb.set_trace()
    writer.add_scalar('relative error X(point by point)', rel_error[0, 0, 0].item(), MyDataLoader.len + i)
    writer.add_scalar('relative error Y(point by point)', rel_error[0, 0, 1].item(), MyDataLoader.len + i)
    writer.add_scalar('relative error Z(point by point)', rel_error[0, 0, 2].item(), MyDataLoader.len + i)
    writer.add_scalar('relative error Mean(point by point)',torch.mean(rel_error).item(), MyDataLoader.len + i)
    loss_vector.append(torch.mean(loss).item())
    rel_error.append(torch.mean(rel_error).item())


# plt.plot(loss_vector)
# plt.plot(rel_error)
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
