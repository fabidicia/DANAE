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
    def __init__(self, device=torch.device("cpu"), n_inputs=39):
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

    def ResetHiddenState(self):   # metodo che resetta l'hidden ???
        self.hidden = (torch.rand(1, 1, 3).to(self.device), torch.rand(1, 3).to(self.device))

    def forward(self, input):  # dato l'input fornisce l'output
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
        self.lstm = nn.LSTM(self.ninput, 3, num_layers=self.nlayers)   # lstm = nn.LSTM(9, 3)
        self.hidden = (torch.rand(self.nlayers, 1, 3), torch.rand(self.nlayers, 1, 3))
        self.hidden[0].requires_grad = False
        self.hidden[1].requires_grad = False

    def ResetHiddenState(self):   # metodo che resetta l'hidden ???
        self.hidden = (torch.rand(self.nlayers, 1, 3), torch.rand(self.nlayers, 1, 3))

    def forward(self, input):  # dato l'input fornisce l'output

        out, self.hidden = self.lstm(input.view(1, 1, self.ninput), self.hidden)
        return out


class MyLSTMCell(nn.Module):
    def __init__(self):
        super(MyLSTMCell, self).__init__()
        self.ninput = 39
        self.lstm = nn.LSTMCell(self.ninput, 3)    # lstm = nn.LSTM(9, 3)
        self.hidden = (torch.rand(self.nlayers, 1, 3), torch.rand(self.nlayers, 1, 3))
        self.cell = (torch.rand(self.nlayers, 1, 3), torch.rand(self.nlayers, 1, 3))    # same dim of hidden
        self.hidden[0].requires_grad = False
        self.hidden[1].requires_grad = False

    def ResetHiddenState(self):   # metodo che resetta l'hidden ???
        self.hidden = (torch.rand(self.nlayers, 1, 3), torch.rand(self.nlayers, 1, 3))

    def forward(self, input):  # dato l'input fornisce l'output

        out, self.hidden, self.cell = self.lstm(input.view(1, 1, self.ninput), self.hidden, self.cell)
        return out


class MyLSTMgru(nn.Module):
    def __init__(self):
        super(MyLSTMgru, self).__init__()
        self.ninput = 39    # input of shape (seq_len, batch, input_size): tensor containing the features of the input sequence. 
        self.lstm = nn.LSTMgru(self.ninput, 3)    # lstm = nn.LSTM(9, 3)
        self.hidden = (torch.rand(self.nlayers, 1, 3), torch.rand(self.nlayers, 1, 3))
        self.hidden[0].requires_grad = False
        self.hidden[1].requires_grad = False

    def ResetHiddenState(self):   # metodo che resetta l'hidden ???
        self.hidden = (torch.rand(self.nlayers, 1, 3), torch.rand(self.nlayers, 1, 3))

    def forward(self, input):  # dato l'input fornisce l'output

        out, self.hidden = self.lstm(input.view(1, 1, self.ninput), self.hidden)
        return out

# model = torch.nn.Sequential(lstm,
#                            torch.nn.ReLU(),
#                            torch.nn.Linear(3,3)) # fully connected layer
# torch.cat(inputs).view(len(inputs), 1, 9) #se come terzo valore metto -1 funziona con tutto perchè chiedo a lui di farlo arbitrariamente


class MyLSTM_integrated(nn.Module):
    def __init__(self,device = torch.device("cpu"),n_inputs=39):
        super(MyLSTM_integrated, self).__init__()
        self.ninput = n_inputs
        self.device = device
        self.lstm = nn.LSTM(self.ninput, 3).to(device)   # lstm = nn.LSTM(9, 3)
        self.fc2 = torch.nn.Linear(3, 3).to(device)    # fc2 = torch.nn.Linear(3,3) #fully connected layer
        self.ReLU = torch.nn.ReLU()     # o ci va F.relu(output)??
        self.hidden = (torch.rand(1, 1, 3).to(device), torch.rand(1, 1, 3).to(device))
        self.hidden[0].requires_grad = False
        self.hidden[1].requires_grad = False
        # hidden = (torch.rand(1, 1, 3), torch.rand(1, 1, 3))
        # optimizer = optim.Adam(chain(*[lstm.parameters(),fc1.parameters(),fc2.parameters()]), lr=0.0001)
        # optimizer = optim.SGD(lstm.parameters(), lr=0.001,momentum=True)

    def ResetHiddenState(self):   # metodo che resetta l'hidden ???
        self.hidden = (torch.rand(1, 1, 3).to(self.device), torch.rand(1, 1, 3).to(self.device))

    def forward(self, input):  # dato l'input fornisce l'output
        input = self.fc1(input)
        input = self.ReLU(input)
        input = self.fcm(input)
        input = self.ReLU(input)
        out, self.hidden = self.lstm(input.view(1, 1, self.ninput), self.hidden)   # ci va il self?
        out = self.ReLU(out)    # out = F.relu(out)
        out = self.fc2(out)
        return out


class YOLO_LSTM(nn.Module):
    def __init__(self, input_dim, output_size, hidden_dim, n_layers, drop_prob=0.0):
        super(YOLO_LSTM, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden_state=None):
        if hidden_state is None:
            lstm_out, hidden_state = self.lstm(x) # dimensione di x; batch_size * seq * element
            lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
            out = self.fc(lstm_out)
            return out
        else:
            lstm_out, hidden_state = self.lstm(x, hidden_state) # dimensione di x; batch_size * seq * element
            lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
            lstm_out = self.dropout(lstm_out)
            out = self.fc(lstm_out)
            return out, hidden_state

class GeneratorBIG(nn.Module):
    def __init__(self):
        super(GeneratorBIG, self).__init__()
        self.conv1 = nn.Conv1d(1,128,kernel_size=3,stride=1,padding=1, bias=True)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv1d(128,128,kernel_size=3,dilation=3,stride=1, padding=1,bias=True)
        self.conv3 = nn.Conv1d(128,128,kernel_size=3,dilation=3,stride=1, padding=1,bias=True)
        self.conv4 = nn.Conv1d(128,128,kernel_size=3,dilation=3,stride=1, padding=1,bias=True)
        self.conv5 = nn.Conv1d(128,128,kernel_size=3,dilation=3,stride=1, padding=1,bias=True)

        self.conv6 = nn.Conv1d(128,128,kernel_size=4,stride=1, padding=0,bias=True)
        self.deconv1 = nn.ConvTranspose1d(128,128,kernel_size=4,stride=1, padding=0,bias=True)

        self.deconv2 = nn.ConvTranspose1d(128,128,kernel_size=3,dilation=3,stride=1, padding=1,bias=True)
        self.deconv3 = nn.ConvTranspose1d(128,128,kernel_size=3,dilation=3,stride=1, padding=1,bias=True)
        self.deconv4 = nn.ConvTranspose1d(128,128,kernel_size=3,dilation=3,stride=1, padding=1,bias=True)
        self.deconv5 = nn.ConvTranspose1d(128,128,kernel_size=3,dilation=3,stride=1, padding=1,bias=True)

        self.conv1d = nn.Conv1d(128,128,kernel_size=3,stride=1, padding=1,bias=True)
        self.conv2d = nn.Conv1d(128,128,kernel_size=3,stride=1, padding=1,bias=True)
        self.conv3d = nn.Conv1d(128,128,kernel_size=3,stride=1, padding=1,bias=True)
        self.conv4d = nn.Conv1d(128,128,kernel_size=3,stride=1, padding=1,bias=True)
        self.conv5d = nn.Conv1d(128,128,kernel_size=3,stride=1, padding=1,bias=True)
        self.conv_final = nn.Conv1d(128,1,kernel_size=3,stride=1, padding=1,bias=True)

##E SE APPLICASSI UN GRANDE RESIDUAL FRA INPUT E OUTPUT??
    def forward(self, input):
        out1 = self.relu(self.conv1(input))
        out2 = self.relu(self.conv2(out1))
        out3 = self.relu(self.conv3(out2))
        out4 = self.relu(self.conv4(out3))
        out5 = self.relu(self.conv5(out4))
        out6 = self.relu(self.conv6(out5))
        out_1 = self.relu(self.deconv1(out6))

        out_11 = self.relu(self.conv1d(out_1 + out5))
        out_2 = self.relu(self.deconv2(out_11))
        out_22 = self.relu(self.conv2d(out_2 + out4))
        out_3 = self.relu(self.deconv3(out_22))
        out_33 = self.relu(self.conv3d(out_3 + out3))
        out_4 = self.relu(self.deconv4(out_33))
        out_44 = self.relu(self.conv4d(out_4 + out2))
        out_5 = self.relu(self.deconv5(out_44))
        out_55 = self.relu(self.conv5d(out_5+ out1))
        out = self.conv_final(out_55)
        return out, out4

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv1d(1,128,kernel_size=3,stride=1,padding=1, bias=True)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv1d(128,128,kernel_size=3,dilation=3,stride=1, padding=1,bias=True)
        self.conv3 = nn.Conv1d(128,128,kernel_size=3,dilation=3,stride=1, padding=1,bias=True)
        self.conv4 = nn.Conv1d(128,128,kernel_size=3,dilation=6,stride=1, padding=1,bias=True)

        self.deconv1 = nn.ConvTranspose1d(128,128,kernel_size=3,dilation=6,stride=1, padding=1,bias=True)
        self.deconv2 = nn.ConvTranspose1d(128,128,kernel_size=3,dilation=3,stride=1, padding=1,bias=True)
        self.deconv3 = nn.ConvTranspose1d(128,128,kernel_size=3,dilation=3,stride=1, padding=1,bias=True)

        self.conv5 = nn.Conv1d(128,128,kernel_size=3,stride=1, padding=1,bias=True)
        self.conv6 = nn.Conv1d(128,128,kernel_size=3,stride=1, padding=1,bias=True)
        self.conv7 = nn.Conv1d(128,128,kernel_size=3,stride=1, padding=1,bias=True)
        self.conv_final = nn.Conv1d(128,1,kernel_size=3,stride=1, padding=1,bias=True)

##E SE APPLICASSI UN GRANDE RESIDUAL FRA INPUT E OUTPUT??
    def forward(self, input):
        out1 = self.relu(self.conv1(input))
        out2 = self.relu(self.conv2(out1))
        out3 = self.relu(self.conv3(out2))
        out4 = self.relu(self.conv4(out3))
        out_1 = self.relu(self.deconv1(out4))
        out_11 = self.relu(self.conv5(out_1 + out3))
        out_2 = self.relu(self.deconv2(out_11))
        out_22 = self.relu(self.conv6(out_2 + out2))
        out_3 = self.relu(self.deconv3(out_22))
        out_33 = self.relu(self.conv7(out_3 + out1))
        out = self.conv_final(out_33)
        return out, out4

class Discriminator(nn.Module):
    def __init__(self, use_sigmoid=False):
        super(Discriminator, self).__init__()
        #self.conv1 = nn.Conv1d(128,512,kernel_size=3,stride=1, bias=True)
        #self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.relu = nn.ReLU(True)
        self.fc1=nn.Linear(128,150,bias=True)
        self.fc2=nn.Linear(150,150,bias=True)
        self.fc_final=nn.Linear(150,2,bias=True)
        self.use_sigmoid = use_sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        #x = self.lrelu(self.conv1(input))
        x = F.adaptive_avg_pool1d(input,(1))
        x = x.reshape(-1,128)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc_final(x)
        if self.use_sigmoid:
            x = self.sigmoid(x)
        return x

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'switchable':
        norm_layer = SwitchNorm2d
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

def update_learning_rate(scheduler, optimizer):
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    print('learning rate = %.7f' % lr)
