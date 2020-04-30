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
from tqdm import tqdm

parser = argparse.ArgumentParser("script to show i-value of IMU data")
parser.add_argument('--folder', type=str, default="/mnt/c/Users/fabia/OneDrive/Desktop/Deep_learning/Oxio_Dataset/handheld/data2/syn/")
parser.add_argument('--arch', type=str, default="MyLSTM")
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--seq_len', type=int, default=10)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--optim', type=str, default="SGD")
parser.add_argument('--past_gt', type=bool, default=False)
# parser.add_argument ('--n', type=int, required=True) # i_th value to display
args = parser.parse_args()

if args.folder == "fabiana":
    args.folder = "/mnt/c/Users/fabia/OneDrive/Desktop/Deep_learning/Oxio_Dataset/handheld/data2/syn/"
elif args.folder == "paolo":
    args.folder = "/home/paolo/datasets/Oxford_Inertial_Odometry_Dataset/handheld/data2/syn/"
elif args.folder == "fabianassh":
    args.folder = "~/Datasets/Oxio_Dataset/handheld/data2/syn/"
else:
   raise Exception("Are u paolo or fabiana? Write the answer to define the folder :)")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = random.randint(0, 1000)
print("SEED EXPERIMENT: "+str(seed))
# Tensorboard writer to plot loss and images and whatever
exper_path = 'runs/'+args.arch+'_experiment_'+str(seed)

INPUT_DIM = 9
OUT_DIM = 7
SEQ_LEN = args.seq_len

if args.past_gt:
    PAST_DIM = 1 * OUT_DIM #used for past_gt which is the past ground truth values
    INPUT_DIM = INPUT_DIM + PAST_DIM

writer = SummaryWriter(exper_path)
MyDataset = MotherOfIMUdata(args.folder,SEQ_LEN+1) if args.past_gt else MotherOfIMUdata(args.folder,SEQ_LEN)
MyDataLoader = DataLoader(MyDataset, batch_size=args.batch_size,shuffle=True, num_workers=1)

# creating my LSTM deep model
if args.arch == "MyLSTM":
    model = MyLSTM(device=device,n_inputs=33)
elif args.arch == "MyLSTM2":
    model = MyLSTM2()
elif args.arch == "MyLSTMCell":
    model = MyLSTMCell()
elif args.arch == "YOLO_LSTM":
    model = YOLO_LSTM(input_dim=INPUT_DIM, output_size=OUT_DIM, hidden_dim=args.hidden_dim, n_layers=2, drop_prob=args.dropout)
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
print_freq = 900
# fai X_gt.append(value.item().numpy())
# torch.cat(inputs).view(len(inputs), 1, 9) #se come terzo valore metto -1 funziona con tutto perchè chiedo a lui di farlo arbitrariamente


for epoch in tqdm(range(args.epochs)):
    for i,(input_tensor, gt_tensor) in enumerate(MyDataLoader):
        
        input_tensor = input_tensor.to(device)
        gt_tensor = gt_tensor.to(device)
        if args.past_gt:
            past_gt = torch.roll(gt_tensor, shifts=1, dims=1).detach() #rolling along sequence axis by 1 step.
            input_tensor = input_tensor[:,1:,:] #i'm removing first element, which belongs to seq_len+1 sequence, so that I have now a seq_len tensor :)
            gt_tensor = gt_tensor[:,1:,:]       # same
            past_gt = past_gt[:,1:,:]       # same
            input_tensor = torch.cat([input_tensor, past_gt], -1) #concatenating along last axis, which is the input data axis! -2(equivalent to 1) would be sequences, -3(equivalent to 0) to  batch
        out = model(input_tensor)
        rel_error = ((out.view(-1, SEQ_LEN, OUT_DIM) - gt_tensor.view(-1, SEQ_LEN, OUT_DIM)).abs() / gt_tensor.view(-1, SEQ_LEN, OUT_DIM).abs()) #now it has BATCH_SIZE x SEQ_LEN x OUT_DIM shape
        total_rel_error += torch.mean(rel_error,dim=[0,1]) # now it has OUT_DIM shape
        loss = loss_function(out.view(-1, SEQ_LEN, OUT_DIM), gt_tensor.view(-1, SEQ_LEN, OUT_DIM))
        loss.backward()  # calcola i gradienti
        optimizer.step()    # aggiorna i pesi della rete a partire dai gradienti calcolati
        # hidden = (torch.rand(1,1,3), torch.rand(1,1,3)) #we reset hidden state every 30 iters
        total_loss += loss.item()
        optimizer.zero_grad()
        writer.add_scalar('training loss (point by point)', loss.item(), epoch * len(MyDataLoader.dataset) + i)
        loss_vector.append(loss.item())

        if i % print_freq == 0:
            print("epoch: " + str(epoch) + ", loss: " + str(total_loss/print_freq) )
            # print("epoch: " + str(epoch) + ", REL_ERROR X,Y,Z: %.2f, %.2f, %.2f" % 
            writer.add_scalar('training loss, mean over ' + str(print_freq), total_loss/print_freq, epoch * len(MyDataLoader.dataset) + i)
            #      total_rel_error[0].item()/2999, total_rel_error[1].item()/2999, total_rel_error[2].item()/2999)
            total_rel_error = (total_rel_error / print_freq)

            print("REL_ERROR: %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f" % tuple([elem.item() for elem in total_rel_error]) )
            print("MEAN_REL_ERROR: %.1f" %(torch.mean(total_rel_error).item() * 100.0))
            total_loss = 0.0
            total_rel_error = 0.0
    scheduler.step()
    torch.save(model.state_dict(), exper_path+'/model.pth')
        # DA AGGIUNGERE IL CODICE PER SALVARE I PESI DEL MODELLO: torch.save(model,path_to_the_file)
print("SEED EXPERIMENT: "+str(seed))

# X_gt = [float(X_gt) for X_gt in X_gt]
# Y_gt = [float(Y_gt) for Y_gt in Y_gt]
# X_pred = [float(X_pred) for X_pred in X_pred]
# Y_pred = [float(Y_pred) for Y_pred in Y_pred]


fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
plt.plot(np.array(loss_vector), 'r')
plt.show()
