'''
This script performs few shot testing in image denoising domain.
Gradient based update is done using MSE loss

'''

import sys
import os
import copy
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import random
import numpy as np
import matplotlib.pyplot as plt
# from skimage.metrics import mean_squared_error
# from scipy.io import loadmat
import torch
from torch.nn import MSELoss
from torch import autograd
from torch.optim import Adam, optimizer
# from torch.nn.modules import loss

from model_config import DnCNN
from utilities.utils import set_seed
from utilities import FineTune

device = torch.device('cuda')
set_seed(100)
compute_mse_loss = MSELoss()
result_path    = "Results/noise2trueMTL2"

class Meta_train():
    def __init__(self,args):
        self.args            = args
        self.args.device     = device
        self.loss_fn         = FineTune(args)
        self.num_of_tasks    = args.num_of_tasks
        self.num_of_updates  = args.num_of_updates
        self.model           = DnCNN(1,8)
        self.model           = self.model.to(device)
        self.pretrained      = args.pretrained
        if self.pretrained is not None:
            self.model.load_state_dict(torch.load(self.pretrained, map_location='cpu'))
            print("Pretrained Model Loaded.")
        
        self.lr_beta         = args.lr_beta
        self.step_size       = args.step_size
        self.gamma           = args.gamma

        # Training stopping criterion
        self.stop_training   = False
        self.prev_loss       = 1e-1

    def train(self):

        task_list = random.sample(range(0,self.num_of_tasks),self.num_of_tasks)


        # losses_a = [] #  Before inner gradient update 
        # rmses_a = [] #  Before inner gradient update 

        losses_b    = [] #  After inner gradient update 
        rmses_b     = [] #  After inner gradient update 
        param_error = []
        # losses_c = [] # after meta update
        # rmses_c = []  # After meta update

        for itr in range(self.args.num_of_epochs):
            optimizer = Adam(self.model.parameters(), lr=self.lr_beta)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma =self.gamma)

            #-------------------------------------------------------
            data1,data2 =[],[] 
            copy_model = copy.deepcopy(self.model)
            for param1 in copy_model.parameters():
                data1.append(param1.data)
            del copy_model
            #---------------------------------------------------------
            # train_rmse_a = 0 
            # train_loss_a = 0

            train_rmse_b = 0 
            train_loss_b = 0 

            # train_rmse_c = 0 
            # train_loss_c = 0 
            sp_var = 0
            for idx in (task_list):
                # task_output = self.loss_fn.task_ft(weights=self.model.state_dict(),task_idx = idx)
                noisy,cln,task_loss_b, task_rmse_b,loss_te = self.loss_fn.task_ft_custom(weights=self.model.state_dict(),task_idx = idx)
                # _,te_samples,task_loss_a,task_loss_b,task_rmse_a,task_rmse_b = task_output
                # train_loss_b += task_loss_b[self.num_of_updates-1]
                # train_rmse_b += task_rmse_b[self.num_of_updates-1]
                sp_var += loss_te
                train_loss_b += task_loss_b
                train_rmse_b += task_rmse_b

                # train_loss_a += task_loss_a
                # train_rmse_a += task_rmse_a

                # self.plot_result(task_output,idx)
            self.model.train()
            meta_loss2 = sp_var/self.num_of_tasks
            
            denoised = self.model(noisy)
            meta_loss1 = compute_mse_loss(denoised,cln)
            meta_loss1 = 0.0
            meta_loss = 0.00001*meta_loss1+meta_loss2
            # meta_loss =meta_loss1+meta_loss2-meta_loss1

            # meta_loss = torch.Tensor([train_loss_b]).to(device)
            # meta_loss = torch.Tensor([0])
            # meta_loss.to(device)
            # print("Before ",meta_loss)
            # meta_loss.set_(torch.Tensor([train_loss_b]).to(device))
            # meta_loss.set_(torch.Tensor([train_loss_b]))
            # meta_loss.requires_grad = True
            optimizer.zero_grad()
            meta_loss.backward()
            for paramm in self.model.parameters():
                print(paramm.grad.data)
            optimizer.step()
            scheduler.step()
     
            #--------------------------
            for param1 in self.model.parameters():
                data2.append(param1.data)
        
            diff = 0
            for k in range(len(data1)):
                d = compute_mse_loss(data1[k],data2[k])
                diff+=d.cpu().detach().numpy()

            # print(diff)
            param_error.append(diff)
            #------------------------------------------------------------
            total_loss = meta_loss2.item()/(self.num_of_tasks)
            total_train_rmse=train_rmse_b/self.num_of_tasks

            losses_b.append(total_loss)
            rmses_b.append(total_train_rmse)

            # ttl = train_loss_a/self.num_of_tasks
            # ttb = train_rmse_a/self.num_of_tasks

            # print("(", (itr + 1), ")  Loss before update: %.1f" % ttl,", RMSE before update, : %.1f" % ttb)
            
            print("(", (itr + 1), ") Training Loss after update: %.1f" % total_loss, ", RMSE, : %.1f" % total_train_rmse)
            
            # self.stopping_criterion(total_loss)
            # losses_a.append(ttl)
            # rmses_a.append(ttb)


            # filename =result_path+"_lr_b_"+str(self.args.lr_beta) \
            # +"updates_"+str(self.args.num_of_updates)+"step_"+str(self.args.step_size)+ "gamma_"+str(self.args.gamma)
                
            # torch.save(self.model.state_dict(), filename + ".pt")
            # np.savez(filename, losses_b=losses_b, rmses_b=rmses_b, param_error = param_error)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.suptitle("lr_b_"+str(self.lr_beta))
        ax1.plot(losses_b)
        # ax1.xlabel("Epochs")
        ax1.set_title("Meta Update Loss Curve")
        ax2.plot(rmses_b)
        # ax2.xlabel("Epochs")
        ax2.set_title("Meta Update RMSE Curve")
        ax3.plot(param_error)
        ax3.set_title("Param rel error Curve")
        plt.show()

        filename ="Test_Results/lr_b"+str(self.lr_beta)+"_/Task_"+"_lr_a_"+str(self.args.lr_alpha) \
        +"updates_"+str(self.args.num_of_updates)+".png"
        # # plt.savefig(filename)
        # fig.set_size_inches((8.5,5), forward=False)
        # fig.savefig(filename, dpi=500)
        # plt.close(fig)

    def stopping_criterion(self,new_loss):
        error = np.abs((self.prev_loss-new_loss)/self.prev_loss)
        print(error)
        if error<1e-5:
            self.stop_training =True
        self.prev_loss = new_loss

    # Ghapla ache ei block a-------------------------------
    def plot_result(self,task_output,idx):
        noise_lvl, loss_a, loss_b , rmse_a, rmse_b = task_output
        loss_a = loss_a.item()
        loss_b = [idx.item() for idx in loss_b]
        filename = "temp/revisited_noise2trueMTL_task_"+str(idx)+"_updates_"+str(self.num_of_updates)
        # np.savez(filename, noise_lvl=noise_lvl, 
        #     loss_a = loss_a, loss_b=loss_b, rmse_a=rmse_a, rmse_b=rmse_b)
        # plt.figure('Loss Curve')
        # plt.plot(loss)
        # plt.show()
        print(loss_b[self.num_of_updates-1])

