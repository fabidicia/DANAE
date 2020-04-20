import os
import torch
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt serviva solo per plottare 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import csv
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import chain


class MyLSTM(nn.Module):
    def __init__(self,device = torch.device("cpu"),n_inputs=39):
        super(MyLSTM, self).__init__()
        self.ninput = n_inputs
        self.device = device
        self.lstm = nn.LSTM(self.ninput, 3).to(device)   # lstm = nn.LSTM(9, 3)
        self.fc1 = torch.nn.Linear(self.ninput, 256).to(device)    # fc1 = torch.nn.Linear(9,9) #fully connected layer
        self.fcm = torch.nn.Linear(256, self.ninput).to(device)
        self.fc2 = torch.nn.Linear(3, 3).to(device)    # fc2 = torch.nn.Linear(3,3) #fully connected layer
        self.ReLU = torch.nn.ReLU()     # o ci va F.relu(output)??
        self.hidden = (torch.rand(1, 1, 3).to(device), torch.rand(1, 1, 3).to(device))
        self.hidden[0].requires_grad = False
        self.hidden[1].requires_grad = False
        # hidden = (torch.rand(1, 1, 3), torch.rand(1, 1, 3))
        # optimizer = optim.Adam(chain(*[lstm.parameters(),fc1.parameters(),fc2.parameters()]), lr=0.0001)
        # optimizer = optim.SGD(lstm.parameters(), lr=0.001,momentum=True)

    def ResetHiddenState(self):   #metodo che resetta l'hidden ???
        self.hidden = (torch.rand(1,1,3).to(self.device),torch.rand(1,1,3).to(self.device))

    def forward(self, input):  #dato l'input fornisce l'output
        input = self.fc1(input)
        input = self.ReLU(input)
        input = self.fcm(input)
        input = self.ReLU(input)
        out, self.hidden = self.lstm(input.view(1, 1, self.ninput), self.hidden)   # ci va il self?
        out = self.ReLU(out)    # out = F.relu(out)
        out = self.fc2(out)
        return out


class MyLSTM2(nn.Module):
    def __init__(self):
        super(MyLSTM2, self).__init__()
        self.ninput = 39
        self.nlayers = 4
        self.lstm = nn.LSTM(self.ninput, 3,num_layers=self.nlayers)   # lstm = nn.LSTM(9, 3)
        self.hidden = (torch.rand(self.nlayers, 1, 3), torch.rand(self.nlayers, 1, 3))
        self.hidden[0].requires_grad = False
        self.hidden[1].requires_grad = False

    def ResetHiddenState(self):   #metodo che resetta l'hidden ???
        self.hidden = (torch.rand(self.nlayers, 1, 3), torch.rand(self.nlayers, 1, 3))

    def forward(self, input):  #dato l'input fornisce l'output

        out, self.hidden = self.lstm(input.view(1, 1, self.ninput), self.hidden)   # ci va il self?
        return out


class MyLSTMCell(nn.Module):
    def __init__(self):
        super(MyLSTMCell, self).__init__()
        self.ninput = 39
        self.nlayers = 4
        self.lstm = nn.LSTMCell(self.ninput, 3,num_layers=self.nlayers)   # lstm = nn.LSTM(9, 3)
        self.hidden = (torch.rand(self.nlayers, 1, 3), torch.rand(self.nlayers, 1, 3))
        self.hidden[0].requires_grad = False
        self.hidden[1].requires_grad = False

    def ResetHiddenState(self):   # metodo che resetta l'hidden ???
        self.hidden = (torch.rand(self.nlayers, 1, 3), torch.rand(self.nlayers, 1, 3))

    def forward(self, input):  #dato l'input fornisce l'output

        out, self.hidden = self.lstm(input.view(1, 1, self.ninput), self.hidden)   # ci va il self?
        return out

#model = torch.nn.Sequential(lstm,
#                            torch.nn.ReLU(),
#                            torch.nn.Linear(3,3)) # fully connected layer
# import pdb; pdb.set_trace() # mette i breakpoints
# torch.cat(inputs).view(len(inputs), 1, 9) #se come terzo valore metto -1 funziona con tutto perchè chiedo a lui di farlo arbitrariamente
