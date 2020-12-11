import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from datasets import Dataset_pred_for_GAN, Dataset_GAN_2
from math import log10, sqrt
from sklearn.metrics import mean_squared_error
import numpy as np
from numpy import mean
from torch.utils.tensorboard import SummaryWriter
from utils import plot_tensorboard
from pathlib import Path
import matplotlib.pyplot as plt
from random import randint
import torch.nn.functional as F
from networks import Generator, Discriminator, weights_init, GANLoss, update_learning_rate, GeneratorBIG

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=32, type=int, help='batch_size')
parser.add_argument('--path', default="./data/preds/slow_walking_train/", help='full path of dataset')
parser.add_argument('--angle', default="theta", help='full path of dataset')
parser.add_argument('--dataset', default="oxford", help='useless parameter')
parser.add_argument('--lr', default=0.0002, type=float, help='lr')
parser.add_argument('--lamb', default=10.0, type=float, help='lambda')
parser.add_argument('--length', default=20,type=int, help='signal length')
parser.add_argument('--epochs', default=1,type=int, help='number of epochs')
parser.add_argument('--input_type', default="kf", help='type of input.choose between kf, kf_est,est')

n_inputs = { "kf":1, "est":6, "kf_est": 7}
args = parser.parse_args()
print(args)

seed = randint(0,1000)
print("experiment seed: "+str(seed))
exper_path = "./runs/KF_9250_"+str(seed)+"/"
Path(exper_path).mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(exper_path)

cudnn.benchmark = True

#dataset = Dataset_pred_for_GAN(seq_length=args.length,path=args.path,angle=args.angle.lower())
dataset = Dataset_GAN_2(seq_length=args.length,path=args.path,angle=args.angle.lower())
test_path = args.path.replace("train","test")
#dataset_test = Dataset_pred_for_GAN(seq_length=args.length,path=test_path,angle=args.angle.lower())
dataset_test = Dataset_GAN_2(seq_length=args.length,path=test_path,angle=args.angle.lower())
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                         shuffle=True, num_workers=0)
test_dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=32,
                                         shuffle=False, num_workers=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

netG = GeneratorBIG(n_inputs=n_inputs[args.input_type]).to(device)
netD = Discriminator().to(device)

netD.apply(weights_init)
netG.apply(weights_init)

criterionGAN = GANLoss().to(device)
criterionL1 = nn.MSELoss().to(device)
#criterionL1 = nn.L1Loss().to(device)
criterionMSE = nn.MSELoss().to(device)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))

def test(args,dataset_test,writer):
    # test
    avg_psnr = 0
    error_list = []
    error_GAN_list = []
    gt_list = []
    kf_list = []
    gan_list = []
    with torch.no_grad():
        for i in range(0,len(dataset_test),args.length): #QUESTO DA IL PROBLEMA CHE SALTERO' ALCUNI SAMPLES DEL TEST SET, ANDRA' FIXATO
            batch = dataset_test[i]
            phi_acc = batch["phi_acc"]
            theta_acc = batch["theta_acc"]
            psi_acc = batch["psi_acc"]
            phi_dot = batch["phi_dot"]
            theta_dot = batch["theta_dot"]
            psi_dot = batch["psi_dot"]
            real_a = batch[args.angle.lower()+"_kf"]
            if args.input_type == 'kf':
                real_a_stack = torch.cat([real_a,],dim=0)
            if args.input_type == 'kf_est':
                real_a_stack = torch.cat([real_a,phi_acc,theta_acc,psi_acc,phi_dot,theta_dot,psi_dot],dim=0)
            if args.input_type == 'est':
                real_a_stack = torch.cat([phi_acc,theta_acc,psi_acc,phi_dot,theta_dot,psi_dot],dim=0)

            real_b = batch[args.angle.lower()+"_gt"]


            real_a, real_a_stack, real_b = real_a.to(device, dtype=torch.float), real_a_stack.to(device, dtype=torch.float), real_b.to(device, dtype=torch.float)
            real_a, real_a_stack, real_b = real_a[None,...], real_a_stack[None,...], real_b[None,...] # se real_a era 6x20, ora diventa 1x6x20 grazie al trick [None,...]

            pred,_ = netG(real_a_stack)
            mse = criterionMSE(pred, real_b)
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
            error_list.append(torch.mean(torch.abs(real_b - real_a)).item())
            error_GAN_list.append(torch.mean(torch.abs(real_b - pred)).item())
            gt_list += real_b[0,0,:].tolist()
            kf_list += real_a[0,0,:].tolist()
            gan_list += pred[0,0,:].tolist()

    print("mean error: " + str(mean(error_list)))
    print("mean GAN error: " + str(mean(error_GAN_list)))
    print("mean deviation gt-kf: %.4f" % np.mean(np.abs(np.asarray(gt_list) - np.asarray(kf_list) )))
    print("mean deviation gt-GAN: %.4f" % np.mean(np.abs(np.asarray(gt_list) - np.asarray(gan_list) )))
    print("max deviation gt-kf: %.4f" % np.max(np.abs(np.asarray(gt_list) - np.asarray(kf_list) )))
    print("max deviation gt-GAN: %.4f" % np.max(np.abs(np.asarray(gt_list) - np.asarray(gan_list) )))
    print("RMS error gt-kf: %.4f" % sqrt(mean_squared_error(gt_list, kf_list)) )
    print("RMS error gt-GAN: %.4f" % sqrt(mean_squared_error(gt_list, gan_list)) )
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(test_dataloader)))
    ### FOR UCS
#    kf_list = kf_list[0:1400]
#    gt_list = gt_list[0:1400]
#    gan_list = gan_list[0:1400]
#    plot_tensorboard(writer,[kf_list[0:1400], gt_list[0:1400]],['b','r'],Labels=["Kalman filter estimation","Ground Truth"],Name="Image_kf0",ylabel=args.angle+" [rad]")
#    plot_tensorboard(writer,[gan_list[0:1400], gt_list[0:1400]],['b','r'],Labels=["DANAE estimation","Ground Truth"],Name="Image_DANAE0",ylabel=args.angle+" [rad]")

     ###FOR OXIOD
    kf_list = kf_list[10000:]
    gt_list = gt_list[10000:]
    gan_list = gan_list[10000:]
    plot_tensorboard(writer,[kf_list[2500:5000], gt_list[2500:5000]],['b','r'],Labels=["Kalman filter estimation","Ground Truth"],Name="Image_kf0",ylabel=args.angle+" [rad]")
    plot_tensorboard(writer,[gan_list[2500:5000], gt_list[2500:5000]],['b','r'],Labels=["DANAE estimation","Ground Truth"],Name="Image_DANAE0",ylabel=args.angle+" [rad]")

    #plot_tensorboard(writer,[gt_list[2500:5000], kf_list[2500:5000]],['r','b'],Labels=["Ground truth","Kalman filter estimation"],Name="Image_kf1",ylabel=args.angle+" [rad]")
    #plot_tensorboard(writer,[gt_list[2500:5000], gan_list[2500:5000]],['r','b'],Labels=["Ground truth","DANAE estimation"],Name="Image_DANAE1",ylabel=args.angle+" [rad]")

    #plot_tensorboard(writer,[gt_list[5000:7500], kf_list[5000:7500]],['r','b'],Labels=["Ground truth","Kalman filter estimation"],Name="Image_kf2",ylabel=args.angle+" [rad]")
    #plot_tensorboard(writer,[gt_list[5000:7500], gan_list[5000:7500]],['r','b'],Labels=["Ground truth","DANAE estimation"],Name="Image_DANAE2",ylabel=args.angle+" [rad]")

    #plot_tensorboard(writer,[gt_list[7500:10000], kf_list[7500:10000]],['r','b'],Labels=["Ground truth","Kalman filter estimation"],Name="Image_kf3",ylabel=args.angle+" [rad]")
    #plot_tensorboard(writer,[gt_list[7500:10000], gan_list[7500:10000]],['r','b'],Labels=["Ground truth","DANAE estimation"],Name="Image_DANAE3",ylabel=args.angle+" [rad]")

    #plot_tensorboard(writer,[gt_list[7500:10000], kf_list[10000:12500]],['r','b'],Labels=["Ground truth","Kalman filter estimation"],Name="Image_kf4",ylabel=args.angle+" [rad]")
    #plot_tensorboard(writer,[gt_list[7500:10000], gan_list[10000:12500]],['r','b'],Labels=["Ground truth","DANAE estimation"],Name="Image_DANAE4",ylabel=args.angle+" [rad]")
##fine della funzione di test


for epoch in range(args.epochs):
    netG.eval() #before testing, always put your network in eval mode!
    test(args,dataset_test,writer)
    netG.train() ##before training,always put your network in train mode!
    # train
    for i, batch in enumerate(train_dataloader, 1):
        # forward
        phi_acc = batch["phi_acc"]
        theta_acc = batch["theta_acc"]
        psi_acc = batch["psi_acc"]
        phi_dot = batch["phi_dot"]
        theta_dot = batch["theta_dot"]
        psi_dot = batch["psi_dot"]
        real_a = batch[args.angle.lower()+"_kf"]
        if args.input_type == 'kf':
            real_a_stack = torch.cat([real_a,],dim=1)
        if args.input_type == 'kf_est':
            real_a_stack = torch.cat([real_a,phi_acc,theta_acc,psi_acc,phi_dot,theta_dot,psi_dot],dim=1)
        if args.input_type == 'est':
            real_a_stack = torch.cat([phi_acc,theta_acc,psi_acc,phi_dot,theta_dot,psi_dot],dim=1)


        real_b = batch[args.angle.lower()+"_gt"]
        real_b, real_a_stack = real_b.to(device, dtype=torch.float), real_a_stack.to(device, dtype=torch.float)
#        real_a, real_b = batch[0].to(device, dtype=torch.float), batch[1].to(device, dtype=torch.float)
        fake_b,fake_b_int = netG(real_a_stack)
#        _,real_b_int = netG(real_b)

        ######################
        # (1) Update D network
        ######################

#        optimizerD.zero_grad()
        
#        pred_fake = netD.forward(fake_b_int.detach())
#        loss_d_fake = criterionGAN(pred_fake, False)

#        pred_real = netD.forward(real_b_int)
#        loss_d_real = criterionGAN(pred_real, True)
        
        # Combined D loss
#        loss_d = (loss_d_fake + loss_d_real) * 0.5

#        loss_d.backward()
       
#        optimizerD.step()

        ######################
        # (2) Update G network
        ######################

        optimizerG.zero_grad()

        # First, G(A) should fake the discriminator
#        pred_fake = netD.forward(fake_b_int)
#        loss_g_gan = criterionGAN(pred_fake, True)

        # Second, G(A) = B
        loss_g_l1 = criterionL1(fake_b, real_b) * args.lamb # se uso lr=0.001 e args.lamb 10 è la stessa cosa che 0.01 e args.lamb 1. Quindi args.lamb è un paarmetro ridondante!!
        
        loss_g = loss_g_l1 #+ loss_g_gan
        
        loss_g.backward() # calcolo dei gradienti

        optimizerG.step() # ottimizzazione rispetto al gradiente
        if i % 100 == 0:
            print("===> Epoch[{}]({}/{}): Loss_G: {:.4f}".format(
                  epoch, i, len(train_dataloader), loss_g.item()))

#    update_learning_rate(netG_scheduler, optimizerG)
#    update_learning_rate(netD_scheduler, optimizerD)



    #checkpoint
    if epoch % 50 == 0:
        if not os.path.exists("checkpoint"):
            os.mkdir("checkpoint")
        if not os.path.exists(os.path.join("checkpoint", args.dataset)):
            os.mkdir(os.path.join("checkpoint", args.dataset))
        netG_model_out_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(args.dataset, epoch)
        netD_model_out_path = "checkpoint/{}/netD_model_epoch_{}.pth".format(args.dataset, epoch)
        torch.save(netG, netG_model_out_path)
        torch.save(netD, netD_model_out_path)
        print("Checkpoint saved to {}".format("checkpoint" + args.dataset))
netG.eval()
test(args,dataset_test,writer)
