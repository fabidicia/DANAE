import argparse
import os
import logging
#logging.disable(logging.WARNING) 
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
#import silence_tensorflow.auto
#import tensorflow as tf
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
from networks import Generator, Discriminator, weights_init, GANLoss, update_learning_rate, GeneratorBIG, GeneratorBIG2, GeneratorBIG3



parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=32, type=int, help='batch_size')
parser.add_argument('--path', default="./data/preds/slow_walking_train/", help='full path of dataset')
parser.add_argument('--angle', default="theta", help='full path of dataset')
parser.add_argument('--dataset', default="oxford", help='useless parameter')
parser.add_argument('--lr', default=0.0002, type=float, help='lr')
parser.add_argument('--lamb', default=10.0, type=float, help='lambda')
parser.add_argument('--length', default=20,type=int, help='signal length')
parser.add_argument('--epochs', default=1,type=int, help='number of epochs')
parser.add_argument('--input_type', default="lkf", help='type of input.choose between kf, kf_est,est')
parser.add_argument('--arch', default="GeneratorBIG", help='danae architecture')

args = parser.parse_args()

ang = args.angle.lower()
input_dict = {}
input_dict["lkf"] = [args.angle.lower()+"_kf"]
input_dict["lest"] = ["phi_acc","theta_acc", "psi_acc","phi_dot", "theta_dot", "psi_dot"]
input_dict["lkf_est"] = [args.angle.lower()+"_kf","phi_acc","theta_acc", "psi_acc","phi_dot", "theta_dot", "psi_dot"]
input_dict["lkf_est_complete"] = ["phi_kf","theta_kf", "psi_kf","phi_acc","theta_acc", "psi_acc","phi_dot", "theta_dot", "psi_dot"]
input_dict["lkf_3input_3output"] = ["phi_kf","theta_kf", "psi_kf"]
##THE CODE ACTUALLY DOESNOT WORK IF THERE AREN'T THE THREE ANGLES IN INPUT
input_dict["ekf"] = [args.angle.lower()+"_kf"]
input_dict["eest"] = ["phi_interm_Gyro","phi_interm_AccMag","theta_interm_Gyro","theta_interm_AccMag","psi_interm_Gyro","psi_interm_AccMag"]
input_dict["ekf_est"] = [args.angle.lower()+"_kf","phi_interm_Gyro","phi_interm_AccMag","theta_interm_Gyro","theta_interm_AccMag","psi_interm_Gyro","psi_interm_AccMag"]
input_dict["ekf_est_single_angle"] = [ang+"_kf",ang+"_interm_Gyro",ang+"_interm_AccMag"]
input_dict["ekf_est_complete"] = ["phi_kf","theta_kf","psi_kf","phi_interm_Gyro","phi_interm_AccMag","theta_interm_Gyro","theta_interm_AccMag","psi_interm_Gyro","psi_interm_AccMag"]

n_inputs = { "lkf":1, "lkf_3input_3output":3,"lest":6, "lkf_est": 7, "ekf":1, "eest":6, "ekf_est":7,"ekf_est_single_angle":3, "ekf_est_complete":9, "ekf_est_complete_single_angle":7, "lkf_est_complete":9}
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

if args.arch == "GeneratorBIG":
    netG = GeneratorBIG(n_inputs=n_inputs[args.input_type],n_angles=3).to(device)
if args.arch == "GeneratorBIG2":
    netG = GeneratorBIG2(n_inputs=n_inputs[args.input_type],n_angles=3).to(device)
if args.arch == "GeneratorBIG3":
    netG = GeneratorBIG3(n_inputs=n_inputs[args.input_type],n_angles=3).to(device)

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

def test(args,dataset_test,writer,epoch=0):
    s_epoch = str(epoch)
    # test
    avg_psnr = 0
    error_list = []
    error_GAN_list = []
    gt_list_phi = []
    kf_list_phi = []
    gan_list_phi = []
    gt_list_theta = []
    kf_list_theta = []
    gan_list_theta = []
    gt_list_psi = []
    kf_list_psi = []
    gan_list_psi = []
    fil_list_phi = []
    fil_list_theta = []
    fil_list_psi = []
    uniform_list_phi = []
    uniform_list_theta = []
    uniform_list_psi = []
    with torch.no_grad():
        for i in range(0,len(dataset_test),args.length): #QUESTO DA IL PROBLEMA CHE SALTERO' ALCUNI SAMPLES DEL TEST SET, ANDRA' FIXATO
            batch = dataset_test[i]

            input = []
            input_list = input_dict[args.input_type]
            for elem in input_list:
                input.append(batch[elem])
            real_a_stack = torch.cat(input,dim=0) ## because here i'm doing forward on each sample, dim=0, differently to train phase, where dim=1
            real_b_phi = batch["phi_gt"]
            real_b_theta = batch["theta_gt"]
            real_b_psi = batch["psi_gt"]
            real_b = torch.cat([real_b_phi,real_b_theta,real_b_psi],dim=0)

            real_a_stack, real_b =  real_a_stack.to(device, dtype=torch.float), real_b.to(device, dtype=torch.float)
            real_a_stack, real_b = real_a_stack[None,...], real_b[None,...] # se real_a era 6x20, ora diventa 1x6x20 grazie al trick [None,...]

            pred,_ = netG(real_a_stack)
            mse = criterionMSE(pred, real_b)
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
#            error_list.append(torch.mean(torch.abs(real_b - real_a)).item())
#            error_GAN_list.append(torch.mean(torch.abs(real_b - pred)).item())
            gt_list_phi += real_b[0,0,:].tolist()
            kf_list_phi += batch["phi_kf"][0,:].tolist()
            gan_list_phi += pred[0,0,:].tolist()

            gt_list_theta += real_b[0,1,:].tolist()
            kf_list_theta += batch["theta_kf"][0,:].tolist()
            gan_list_theta += pred[0,1,:].tolist()

            gt_list_psi += real_b[0,2,:].tolist()
            kf_list_psi += batch["psi_kf"][0,:].tolist()
            gan_list_psi += pred[0,2,:].tolist()

            try:
                fil_list_phi +=batch["phi_kf_fil"][0,:].tolist()
                fil_list_theta += batch["theta_kf_fil"][0,:].tolist()
                fil_list_psi += batch["psi_kf_fil"][0,:].tolist()
                uniform_list_phi +=batch["phi_kf_uniform"][0,:].tolist()
                uniform_list_theta += batch["theta_kf_uniform"][0,:].tolist()
                uniform_list_psi += batch["psi_kf_uniform"][0,:].tolist()

            except:
                pass
 #   print("mean error: " + str(mean(error_list)))
 #   print("mean GAN error: " + str(mean(error_GAN_list)))
    print("mean deviation gt-kf_phi: %.4f" % np.mean(np.abs(np.asarray(gt_list_phi) - np.asarray(kf_list_phi) )))
    print("mean deviation gt-GAN_phi: %.4f" % np.mean(np.abs(np.asarray(gt_list_phi) - np.asarray(gan_list_phi) )))
    print("max deviation gt-kf_phi: %.4f" % np.max(np.abs(np.asarray(gt_list_phi) - np.asarray(kf_list_phi) )))
    print("max deviation gt-GAN_phi: %.4f" % np.max(np.abs(np.asarray(gt_list_phi) - np.asarray(gan_list_phi) )))
    print("RMS error gt-kf_phi: %.4f" % sqrt(mean_squared_error(gt_list_phi, kf_list_phi)) )
    print("RMS error gt-GAN_phi: %.4f" % sqrt(mean_squared_error(gt_list_phi, gan_list_phi)) )
    print("===> Avg. PSNR_phi: {:.4f} dB".format(avg_psnr / len(test_dataloader)))

    print("mean deviation gt-kf_theta: %.4f" % np.mean(np.abs(np.asarray(gt_list_theta) - np.asarray(kf_list_theta) )))
    print("mean deviation gt-GAN_theta: %.4f" % np.mean(np.abs(np.asarray(gt_list_theta) - np.asarray(gan_list_theta) )))
    print("max deviation gt-kf_theta: %.4f" % np.max(np.abs(np.asarray(gt_list_theta) - np.asarray(kf_list_theta) )))
    print("max deviation gt-GAN_theta: %.4f" % np.max(np.abs(np.asarray(gt_list_theta) - np.asarray(gan_list_theta) )))
    print("RMS error gt-kf_theta: %.4f" % sqrt(mean_squared_error(gt_list_theta, kf_list_theta)) )
    print("RMS error gt-GAN_theta: %.4f" % sqrt(mean_squared_error(gt_list_theta, gan_list_theta)) )
    print("===> Avg. PSNR_theta: {:.4f} dB".format(avg_psnr / len(test_dataloader)))

    print("mean deviation gt-kf_psi: %.4f" % np.mean(np.abs(np.asarray(gt_list_psi) - np.asarray(kf_list_psi) )))
    print("mean deviation gt-GAN_psi: %.4f" % np.mean(np.abs(np.asarray(gt_list_psi) - np.asarray(gan_list_psi) )))
    print("max deviation gt-kf_psi: %.4f" % np.max(np.abs(np.asarray(gt_list_psi) - np.asarray(kf_list_psi) )))
    print("max deviation gt-GAN_psi: %.4f" % np.max(np.abs(np.asarray(gt_list_psi) - np.asarray(gan_list_psi) )))
    print("RMS error gt-kf_psi: %.4f" % sqrt(mean_squared_error(gt_list_psi, kf_list_psi)) )
    print("RMS error gt-GAN_psi: %.4f" % sqrt(mean_squared_error(gt_list_psi, gan_list_psi)) )
    print("===> Avg. PSNR_psi: {:.4f} dB".format(avg_psnr / len(test_dataloader)))

    try:
        print("mean deviation gt-fil_phi: %.4f" % np.mean(np.abs(np.asarray(gt_list_phi) - np.asarray(fil_list_phi) )))
        print("max deviation gt-fil_phi: %.4f" % np.max(np.abs(np.asarray(gt_list_phi) - np.asarray(fil_list_phi) )))
        print("RMS error gt-fil_phi: %.4f" % sqrt(mean_squared_error(gt_list_phi, fil_list_phi)) )

        print("mean deviation gt-fil_theta: %.4f" % np.mean(np.abs(np.asarray(gt_list_theta) - np.asarray(fil_list_theta) )))
        print("max deviation gt-fil_theta: %.4f" % np.max(np.abs(np.asarray(gt_list_theta) - np.asarray(fil_list_theta) )))
        print("RMS error gt-fil_theta: %.4f" % sqrt(mean_squared_error(gt_list_theta, fil_list_theta)) )

        print("mean deviation gt-fil_psi: %.4f" % np.mean(np.abs(np.asarray(gt_list_psi) - np.asarray(fil_list_psi) )))
        print("max deviation gt-fil_psi: %.4f" % np.max(np.abs(np.asarray(gt_list_psi) - np.asarray(fil_list_psi) )))
        print("RMS error gt-fil_psi: %.4f" % sqrt(mean_squared_error(gt_list_psi, fil_list_psi)) )

        print("mean deviation gt-uniform_phi: %.4f" % np.mean(np.abs(np.asarray(gt_list_phi) - np.asarray(uniform_list_phi) )))
        print("max deviation gt-uniform_phi: %.4f" % np.max(np.abs(np.asarray(gt_list_phi) - np.asarray(uniform_list_phi) )))
        print("RMS error gt-uniform_phi: %.4f" % sqrt(mean_squared_error(gt_list_phi, uniform_list_phi)) )

        print("mean deviation gt-uniform_theta: %.4f" % np.mean(np.abs(np.asarray(gt_list_theta) - np.asarray(uniform_list_theta) )))
        print("max deviation gt-uniform_theta: %.4f" % np.max(np.abs(np.asarray(gt_list_theta) - np.asarray(uniform_list_theta) )))
        print("RMS error gt-uniform_theta: %.4f" % sqrt(mean_squared_error(gt_list_theta, uniform_list_theta)) )

        print("mean deviation gt-uniform_psi: %.4f" % np.mean(np.abs(np.asarray(gt_list_psi) - np.asarray(uniform_list_psi) )))
        print("max deviation gt-uniform_psi: %.4f" % np.max(np.abs(np.asarray(gt_list_psi) - np.asarray(uniform_list_psi) )))
        print("RMS error gt-uniform_psi: %.4f" % sqrt(mean_squared_error(gt_list_psi, uniform_list_psi)) )
    except:
        print("Low pass filters not found, skipping their plot")

     ###FOR OXIOD
    if args.dataset == "oxford":
        kf_list_theta = kf_list_theta[10000:]
        kf_list_phi = kf_list_phi[10000:]
        kf_list_psi = kf_list_psi[10000:]
        gt_list_theta = gt_list_theta[10000:]
        gt_list_phi = gt_list_phi[10000:]
        gt_list_psi = gt_list_psi[10000:]
        gan_list_phi = gan_list_phi[10000:]
        gan_list_theta = gan_list_theta[10000:]
        gan_list_psi = gan_list_psi[10000:]
        plot_tensorboard(writer,[kf_list_phi[2500:5000], gt_list_phi[2500:5000]],['#a87813','#490101'],Labels=["Kalman Filter estimation","Ground Truth"],Name="kf_phi_"+s_epoch,ylabel="phi [rad]",ylim=[-0.6,-0.1])
        writer.flush()
        plot_tensorboard(writer,[kf_list_theta[2500:5000], gt_list_theta[2500:5000]],['#a87813','#490101'],Labels=["Kalman Filter estimation","Ground Truth"],Name="kf_theta_"+s_epoch,ylabel="theta [rad]",ylim=[-0.1,0.3])
        writer.flush()
        plot_tensorboard(writer,[kf_list_psi[2500:5000], gt_list_psi[2500:5000]],['#a87813','#490101'],Labels=["Kalman Filter estimation","Ground Truth"],Name="kf_psi_"+s_epoch,ylabel="psi [rad]")
        writer.flush()
        plot_tensorboard(writer,[gan_list_phi[2500:5000], gt_list_phi[2500:5000]],['#43a5ff','#490101'],Labels=["DANAE++ estimation","Ground Truth"],Name="DANAE++_phi_"+s_epoch,ylabel="phi [rad]",ylim=[-0.6,-0.1])
        writer.flush()
        plot_tensorboard(writer,[gan_list_theta[2500:5000], gt_list_theta[2500:5000]],['#43a5ff','#490101'],Labels=["DANAE++ estimation","Ground Truth"],Name="DANAE++_theta_"+s_epoch,ylabel="theta [rad]",ylim=[-0.1,0.3])
        writer.flush()
        plot_tensorboard(writer,[gan_list_psi[2500:5000], gt_list_psi[2500:5000]],['#43a5ff','#490101'],Labels=["DANAE++ estimation","Ground Truth"],Name="DANAE++_psi_"+s_epoch,ylabel="psi [rad]")
        writer.flush()

        try:
            fil_list_theta = fil_list_theta[10000:]
            fil_list_phi = fil_list_phi[10000:]
            fil_list_psi = fil_list_psi[10000:]
            uniform_list_phi = uniform_list_phi[10000:]
            uniform_list_theta = uniform_list_theta[10000:]
            uniform_list_psi = uniform_list_psi[10000:]
            plot_tensorboard(writer,[gan_list_phi[2500:5000], gt_list_phi[2500:5000],fil_list_phi[2500:5000]],['#43a5ff','#490101','#a87813'],Labels=["DANAE++ estimation","Ground Truth","Butter Low-pass Filter estimation"],Name="fil_phi_"+s_epoch,ylabel="phi [rad]",ylim=[-0.6,-0.1])
            writer.flush()
            plot_tensorboard(writer,[gan_list_theta[2500:5000], gt_list_theta[2500:5000],fil_list_theta[2500:5000]],['#43a5ff','#490101','#a87813'],Labels=["DANAE++ estimation","Ground Truth","Butter Low-pass Filter estimation"],Name="fil_theta_"+s_epoch,ylabel="theta [rad]",ylim=[-0.1,0.3])
            writer.flush()
            plot_tensorboard(writer,[gan_list_psi[2500:5000], gt_list_psi[2500:5000],fil_list_psi[2500:5000]],['#43a5ff','#490101','#a87813'],Labels=["DANAE++ estimation","Ground Truth","Butter Low-pass Filter estimation"],Name="fil_psi_"+s_epoch,ylabel="psi [rad]")
            writer.flush()
            plot_tensorboard(writer,[gan_list_phi[2500:5000], gt_list_phi[2500:5000],uniform_list_phi[2500:5000]],['#43a5ff','#490101','#a87813'],Labels=["DANAE++ estimation","Ground Truth","Uniform Filter estimation"],Name="uniform_phi_"+s_epoch,ylabel="phi [rad]",ylim=[-0.6,-0.1])
            writer.flush()
            plot_tensorboard(writer,[gan_list_theta[2500:5000], gt_list_theta[2500:5000],uniform_list_theta[2500:5000]],['#43a5ff','#490101','#a87813'],Labels=["DANAE++ estimation","Ground Truth","Uniform Filter estimation"],Name="uniform_theta_"+s_epoch,ylabel="theta [rad]",ylim=[-0.1,0.3])
            writer.flush()
            plot_tensorboard(writer,[gan_list_psi[2500:5000], gt_list_psi[2500:5000],uniform_list_psi[2500:5000]],['#43a5ff','#490101','#a87813'],Labels=["DANAE++ estimation","Ground Truth", "Uniform Filter Estimation"],Name="uniform_psi_"+s_epoch,ylabel="psi [rad]")
            writer.flush()
        except:
            pass
    else:
    ### FOR UCS
#    kf_list = kf_list[0:1400]
#    gt_list = gt_list[0:1400]
#    gan_list = gan_list[0:1400]
#    plot_tensorboard(writer,[kf_list[0:1400], gt_list[0:1400]],['b','r'],Labels=["Kalman filter estimation","Ground Truth"],Name="Image_kf0",ylabel=args.angle+" [rad]")
#    plot_tensorboard(writer,[gan_list[0:1400], gt_list[0:1400]],['b','r'],Labels=["DANAE estimation","Ground Truth"],Name="Image_DANAE0",ylabel=args.angle+" [rad]")
        kf_list_theta = kf_list_theta[0:1400]
        kf_list_phi = kf_list_phi[0:1400]
        kf_list_psi = kf_list_psi[0:1400]
        gt_list_theta = gt_list_theta[0:1400]
        gt_list_phi = gt_list_phi[0:1400]
        gt_list_psi = gt_list_psi[0:1400]
        gan_list_phi = gan_list_phi[0:1400]
        gan_list_theta = gan_list_theta[0:1400]
        gan_list_psi = gan_list_psi[0:1400]
        plot_tensorboard(writer,[kf_list_phi[0:1400], gt_list_phi[0:1400]],['#a87813','#490101'],Labels=["Kalman Filter estimation", "Ground Truth"],Name="Image_kf_phi_"+s_epoch,ylabel="phi [rad]")
        plot_tensorboard(writer,[kf_list_theta[0:1400], gt_list_theta[0:1400]],['#a87813','#490101'],Labels=["Kalman Filter estimation","Ground Truth"],Name="Image_kf_theta_"+s_epoch,ylabel="theta [rad]")
        plot_tensorboard(writer,[kf_list_psi[0:1400], gt_list_psi[0:1400]],['#a87813','#490101'],Labels=["Kalman Filter estimation","Ground Truth"],Name="Image_kf_psi_"+s_epoch,ylabel="psi [rad]")

        plot_tensorboard(writer,[gan_list_phi[0:1400], gt_list_phi[0:1400]],['#43a5ff','#490101'],Labels=["DANAE++ estimation","Ground Truth"],Name="Image_DANAE2_phi_"+s_epoch,ylabel="phi [rad]")
        plot_tensorboard(writer,[gan_list_theta[0:1400], gt_list_theta[0:1400]],['#43a5ff','#490101'],Labels=["DANAE++ estimation","Ground Truth"],Name="Image_DANAE2_theta_"+s_epoch,ylabel="theta [rad]")
        plot_tensorboard(writer,[gan_list_psi[0:1400], gt_list_psi[0:1400]],['#43a5ff','#490101'],Labels=["DANAE++ estimation","Ground Truth"],Name="Image_DANAE2_psi_"+s_epoch,ylabel="psi [rad]")



for epoch in range(args.epochs):
    netG.train() ##before training,always put your network in train mode!

    # train
    for i, batch in enumerate(train_dataloader, 1):
        input = []
        input_list = input_dict[args.input_type]
        for elem in input_list:
            input.append(batch[elem])
        real_a_stack = torch.cat(input,dim=1)

        real_b_phi = batch["phi_gt"]
        real_b_theta = batch["theta_gt"]
        real_b_psi = batch["psi_gt"]
        real_b = torch.cat([real_b_phi,real_b_theta,real_b_psi],dim=1)

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
        loss_g_l1 = criterionL1(fake_b, real_b) * args.lamb

        loss_g = loss_g_l1 #+ loss_g_gan

        loss_g.backward() # calcolo dei gradienti

        optimizerG.step() # ottimizzazione rispetto al gradiente
        if i % 100 == 0:
            print("===> Epoch[{}]({}/{}): Loss_G: {:.4f}".format(
                  epoch, i, len(train_dataloader), loss_g.item()))

#    update_learning_rate(netG_scheduler, optimizerG)
#    update_learning_rate(netD_scheduler, optimizerD)


    netG.eval() #before testing, always put your network in eval mode!
    test(args,dataset_test,writer,epoch)

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
