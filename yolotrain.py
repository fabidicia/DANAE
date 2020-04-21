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
from datasets import MotherOfIMUdata
from networks import *
import LKF
import random
from datasets import SimpleDataset
from torch.optim.lr_scheduler import MultiStepLR
# import pdb; pdb.set_trace() # mette i breakpoints

parser = argparse.ArgumentParser("script to show i-value of IMU data")
parser.add_argument('--folder', type=str, default="/mnt/c/Users/fabia/OneDrive/Desktop/Deep learning/Oxford Inertial Odometry Dataset/handheld/data2/syn/")
parser.add_argument('--arch', type=str, default="MyLSTM")
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--optim', type=str, default="SGD")
# parser.add_argument ('--n', type=int, required=True) # i_th value to display
args = parser.parse_args()

if args.folder == "fabiana":
    args.folder = "/mnt/c/Users/fabia/OneDrive/Desktop/Deep learning/Oxford Inertial Odometry Dataset/handheld/data2/syn/"
elif args.folder == "paolo":
    args.folder = "/home/paolo/datasets/Oxford_Inertial_Odometry_Dataset/handheld/data2/syn/"
#else:
 #   raise Exception("Are u paolo or fabiana? Write the answer to define the folder :)")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = random.randint(0, 1000)
print("SEED EXPERIMENT: "+str(seed))
# Tensorboard writer to plot loss and images and whatever
exper_path = 'runs/'+args.arch+'_experiment_'+str(seed)

INPUT_DIM = 9
OUT_DIM = 7
SEQ_LEN = 10

writer = SummaryWriter(exper_path)
MyDataset = MotherOfIMUdata(args.folder,SEQ_LEN)
MyDataLoader = DataLoader(MyDataset, batch_size=32,shuffle=True, num_workers=1)

# creating my LSTM deep model
if args.arch == "MyLSTM":
    model = MyLSTM(device=device,n_inputs=33)
elif args.arch == "MyLSTM2":
    model = MyLSTM2()
elif args.arch == "MyLSTMCell":
    model = MyLSTMCell()
elif args.arch == "YOLO_LSTM":
    model = YOLO_LSTM(input_dim=INPUT_DIM, output_size=OUT_DIM, hidden_dim=128, n_layers=2, drop_prob=0.0)
model = model.to(device)    # casting the model to the correct device, cpu or gpu


loss_function = nn.MSELoss()

if args.optim == "Adam":
    optimizer = optim.Adam(chain(*[model.parameters()]), lr=0.0001)
elif args.optim == "SGD":
    optimizer = optim.SGD(chain(*[model.parameters()]), lr=0.01, momentum=0.9)
else:
    raise Exception("Invalid value for args.optim! Chooose between Adam and SGD :-p ")

scheduler = MultiStepLR(optimizer, milestones=[100,200], gamma=0.1)
loss_vector = []
total_loss = 0.0
total_rel_error = 0.0

# fai X_gt.append(value.item().numpy())
# torch.cat(inputs).view(len(inputs), 1, 9) #se come terzo valore metto -1 funziona con tutto perchè chiedo a lui di farlo arbitrariamente

for epoch in range(args.epochs):
    for i,(input_tensor, gt_tensor) in enumerate(MyDataLoader):
        input_tensor = input_tensor.to(device)
        gt_tensor = gt_tensor.to(device)
    
        out = model(input_tensor)
        total_rel_error += ((out.view(-1, SEQ_LEN, OUT_DIM) - gt_tensor.view(-1, SEQ_LEN, OUT_DIM)).abs() / gt_tensor.view(-1, SEQ_LEN, OUT_DIM).abs())[0, 0]

        loss = loss_function(out.view(-1, SEQ_LEN, OUT_DIM), gt_tensor.view(-1, SEQ_LEN, OUT_DIM))
        loss.backward()  # calcola i gradienti
        optimizer.step()    # aggiorna i pesi della rete a partire dai gradienti calcolati
        # hidden = (torch.rand(1,1,3), torch.rand(1,1,3)) #we reset hidden state every 30 iters
        total_loss += loss.item()
        optimizer.zero_grad()
        writer.add_scalar('training loss (point by point)', loss.item(), epoch * len(MyDataLoader.dataset) + i)
        loss_vector.append(loss.item())

        if i % 9000 == 0:
            print("epoch: " + str(epoch) + ", loss: " + str(total_loss/100) )
            # print("epoch: " + str(epoch) + ", REL_ERROR X,Y,Z: %.2f, %.2f, %.2f" % 
            writer.add_scalar('training loss (mean over 100)', total_loss/100, epoch * len(MyDataLoader.dataset) + i)
            #      total_rel_error[0].item()/2999, total_rel_error[1].item()/2999, total_rel_error[2].item()/2999)
            rel_error_X = total_rel_error[0].item() / 9000
            rel_error_Y = total_rel_error[1].item() / 9000
            rel_error_Z = total_rel_error[2].item() / 9000

            print("REL_ERROR: %.3f, %.3f, %.3f" % (rel_error_X, rel_error_Y, rel_error_Z) )
            total_loss = 0.0
            total_rel_error = 0.0
    scheduler.step()
    torch.save(model.state_dict(), exper_path+'/model.pth')
        # DA AGGIUNGERE IL CODICE PER SALVARE I PESI DEL MODELLO: torch.save(model,path_to_the_file)
print("SEED EXPERIMENT: "+str(seed))

#X_gt = [float(X_gt) for X_gt in X_gt]
#Y_gt = [float(Y_gt) for Y_gt in Y_gt]
#X_pred = [float(X_pred) for X_pred in X_pred]
#Y_pred = [float(Y_pred) for Y_pred in Y_pred]


fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
plt.plot(np.array(loss_vector), 'r')
plt.show()

#    del loss,input_list,grt_transl,input_tensor
# torch.manual_seed(1)    # seed è un numero che si da in input ad una funzione o libreria in modo che sia riproducibile

