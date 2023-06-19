'''
This script performs few shot testing in image denoising domain.
Gradient based update is done using MSE loss

'''
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import random
import numpy as np
import matplotlib.pyplot as plt
# from skimage.metrics import mean_squared_error
# from scipy.io import loadmat
import torch
from torch.nn.modules import loss

from model_config import DnCNN
from utilities.utils import set_seed
from utilities import FineTune

device = torch.device('cuda')
set_seed(100)

class Meta_test():
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

    def test(self):
        task_list = random.sample(range(0,self.num_of_tasks),self.num_of_tasks)

        for itr in range(1):  
            test_rmse_b = 0 
            test_loss_b = 0 
            for idx in (task_list):
                task_output = self.loss_fn.task_ft(weights=self.model.state_dict(),task_idx = idx)
                self.plot_result(task_output,idx)
    
    # Ghapla ache ei block a-------------------------------
    def plot_result(self,task_output,idx):
        noise_lvl, loss_a, loss_b , rmse_a, rmse_b = task_output
        loss_a = loss_a.item()
        loss_b = [idx.item() for idx in loss_b]
        filename = "temp/noise2true_task_"+str(idx)+"_updates_"+str(self.num_of_updates)
        np.savez(filename, noise_lvl=noise_lvl, 
            loss_a = loss_a, loss_b=loss_b, rmse_a=rmse_a, rmse_b=rmse_b)
        # plt.figure('Loss Curve')
        # plt.plot(loss)
        # plt.show()

