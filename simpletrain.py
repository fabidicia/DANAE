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
import LKF
import random
# import pdb; pdb.set_trace() # mette i breakpoints

parser = argparse.ArgumentParser("script to show i-value of IMU data")
parser.add_argument('--folder', type=str, default="/mnt/c/Users/fabia/OneDrive/Desktop/Deep learning/Oxford Inertial Odometry Dataset/handheld/data2/syn/")
parser.add_argument('--arch', type=str, default="MyLSTM")
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--update_step', type=int, default=200)
parser.add_argument('--optim', type=str, default="Adam")
# parser.add_argument ('--n', type=int, required=True) # i_th value to display
args = parser.parse_args()

if args.folder == "fabiana":
    args.folder = "/mnt/c/Users/fabia/OneDrive/Desktop/Deep learning/Oxford Inertial Odometry Dataset/handheld/data2/syn/"
elif args.folder == "paolo":
    args.folder = "/home/paolo/datasets/Oxford_Inertial_Odometry_Dataset/handheld/data2/syn/"
#else:
 #   raise Exception("Are u paolo or fabiana? Write the answer to define the folder :)")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
imudata = IMUdata(args.folder)  # istanza della classe
seed = random.randint(0, 1000)
print("SEED EXPERIMENT: "+str(seed))
# Tensorboard writer to plot loss and images and whatever
exper_path = 'runs/'+args.arch+'_experiment_'+str(seed)

writer = SummaryWriter(exper_path)
train_list = []   # la lista si definisce con le quadre
gt_list = []
past_gt = torch.zeros([10, 1, 3])   # 10 è dim0, 1 è dim1, 3 è dim2
past_gt = past_gt.view(1, 30)
# past_gt = torch.zeros([10])

# creating my LSTM deep model
if args.arch == "MyLSTM":
    model = MyLSTM(device=device,n_inputs=33)
elif args.arch == "MyLSTM2":
    model = MyLSTM2()
model = model.to(device)    # casting the model to the correct device, cpu or gpu


loss_function = nn.MSELoss()

if args.optim == "Adam":
    optimizer = optim.Adam(chain(*[model.parameters()]), lr=0.0001)
elif args.optim == "SGD":
    optimizer = optim.SGD(chain(*[model.parameters()]), lr=0.00001, momentum=0.9)
else:
    raise Exception("Invalid value for args.optim! Chooose between Adam and SGD :-p ")

loss_vector = []
total_loss = 0.0
total_rel_error = 0.0
counter = 0

# fai X_gt.append(value.item().numpy())
# torch.cat(inputs).view(len(inputs), 1, 9) #se come terzo valore metto -1 funziona con tutto perchè chiedo a lui di farlo arbitrariamente

# # Train and GT Data cretor
for i in range(0, imudata.len, 1):
    inputs = [i, i + 1, i - 2]
    train_list.append(inputs)
    summ = sum(inputs)
    tmp = [float(j)*float(j) for j in inputs]
    quad = sum(tmp)
    tmp2 = [float(j)*float(j)*float(j) for j in inputs]  
    cube = sum(tmp2)
    gt = [summ,quad,cube]
    gt_list.append(gt)

for epoch in range(args.epochs):
     #counter = 0
    model.ResetHiddenState()
    for i in range(0, imudata.len, 1):
        # hidden = (torch.rand(1, 1, 3), torch.rand(1, 1, 3))
        # _, acc, gyr, mag, gt_rot, gt_tran = imudata.__getitem__(i)  # l'_ all'inizio dice che ci andrebbe una variabile ma non mi serve e non la metto. questo perchè bisogna che tutte le chiamate corrispondano a quelle della function __getitem__
        input_list = train_list[i]
        # input_list = [float(i) for i in input_list]     # quando leggo valori in una lista servono sempre le quadre
        input_tensor = torch.Tensor(input_list)
        input_tensor = input_tensor.view(1, -1)
        newinput_tensor = torch.cat([input_tensor, past_gt], -1)     # tensore 1*19
        gt = gt_list[i]
        # grt_transl = [float(i) for i in grt_transl]
        gt_tensor = torch.Tensor(gt).to(device) #tensore 1x3
        newinput_tensor = newinput_tensor.to(device) #casting input data to device
        out = model(newinput_tensor)
        total_rel_error += ((out.view(1, 1, 3) - gt_tensor.view(1, 1, 3)).abs() / gt_tensor.view(1, 1, 3).abs())[0, 0]

        if i % args.update_step == 0:
            loss = loss_function(out.view(1, 1, 3), gt_tensor.view(1, 1, 3))
            loss.backward(retain_graph=True)  # calcola i gradienti
            optimizer.step()    # aggiorna i pesi della rete a partire dai gradienti calcolati
            # hidden = (torch.rand(1,1,3), torch.rand(1,1,3)) #we reset hidden state every 30 iters
            total_loss += loss.item()

            counter = counter + 1
            optimizer.zero_grad()
            writer.add_scalar('training loss (point by point)', loss.item(), epoch * imudata.len + i)
            writer.add_scalar('training loss (mean over 100)', total_loss/100, epoch * imudata.len + i)
        past_gt = torch.roll(past_gt, shifts=3)
        past_gt[0, 0:3] = gt_tensor[0:3]     # fino al 2ndo
        loss_vector.append(loss.item())

        if i % 3000 == 2999:
            print("epoch: " + str(epoch) + ", loss: " + str(total_loss/100))
            # print("epoch: " + str(epoch) + ", REL_ERROR X,Y,Z: %.2f, %.2f, %.2f" % 
            #      total_rel_error[0].item()/2999, total_rel_error[1].item()/2999, total_rel_error[2].item()/2999)
            rel_error_X = total_rel_error[0].item()/2999.0
            rel_error_Y = total_rel_error[1].item()/2999.0
            rel_error_Z = total_rel_error[2].item()/2999.0

            print("REL_ERROR: %.2f, %.2f, %.2f" % (rel_error_X, rel_error_Y, rel_error_Z) )
            total_loss = 0.0
            total_rel_error = 0.0
            torch.save(model.state_dict(), exper_path+'/model.pth')
            # DA AGGIUNGERE IL CODICE PER SALVARE I PESI DEL MODELLO: torch.save(model,path_to_the_file)
print("SEED EXPERIMENT: "+str(seed))

#X_gt = [float(X_gt) for X_gt in X_gt]
#Y_gt = [float(Y_gt) for Y_gt in Y_gt]
#X_pred = [float(X_pred) for X_pred in X_pred]
#Y_pred = [float(Y_pred) for Y_pred in Y_pred]


fig1 = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
plt.plot(np.array(loss_vector), 'r')
plt.show()

#    del loss,input_list,grt_transl,input_tensor
# torch.manual_seed(1)    # seed è un numero che si da in input ad una funzione o libreria in modo che sia riproducibile

