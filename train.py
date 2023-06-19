import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error
from scipy.io import loadmat
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset as dset
from utilities.utils import set_seed
from utilities.mydataset import mydataset
from utilities.loss_fn import Noise2Self as Loss_Func
import torch.nn as nn

from mask import Masker
from torch.nn import MSELoss
from torch.optim import Adam

from model_config.dncnn import DnCNN
import wandb

    
    

if __name__=='__main__':
    wandb.init(project="mtl_ssrl",
           config={
    "num_of_layers" : 8,
    "batch_size" : 2,
    "lr" : 1e-1,
    "step_size" : 10,
    "gamma" : 0.95,
    "num_epoch" : 500,})    
    train_mode = 'noise2self'


    device = torch.device('cuda')

    set_seed(100)

    num_of_layers = 8
    batch_size = 2
    lr = 1e-1
    step_size = 10
    gamma = 0.95

    num_epoch = 500 #1000


    model = DnCNN(1, num_of_layers=num_of_layers)
    model = model.to(device)



    loss_function_mean = MSELoss(reduction='mean')
    loss_function_sum = MSELoss(reduction='sum')

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    train_dataset = mydataset('data/train_imgs.mat')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False)


    losses = []
    train_rmses = []
    result_path = 'Results/'+train_mode

    if train_mode=='noise2self':
        masker = Masker(width=4, mode='interpolate')
        # loss = Loss_Func(model,train_mode)
        loss = Loss_Func()
    elif train_mode=='noise2same':
        masker = Masker(width=14, mode='interpolate')
        # loss = Loss_Func(model,train_mode)
        loss = Loss_Func()    

    for epoch in range(num_epoch):
        model.train()
        total_loss = 0
        total_train_rmse = 0

        for idx, batch in enumerate(train_loader):
            clean_images, noisy_images = batch

            noisy_images = noisy_images.to(device, dtype=torch.float)

            net_input, mask = masker.mask(noisy_images, idx)

            net_output = model(net_input)
            denoised = model(noisy_images)



            loss = loss_function_sum(net_output * mask, noisy_images * mask) / (batch_size * torch.sum(mask))
            wandb.log({"loss_basic": loss})
            denoised = np.squeeze(denoised.detach().cpu().numpy().astype(np.float64))
            clean_image = np.squeeze(clean_images.cpu().numpy().astype(np.float64))

            train_rmse = np.sqrt(mean_squared_error(denoised, clean_image))
            total_train_rmse += train_rmse

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        total_loss = total_loss / (idx + 1)
        total_train_rmse = total_train_rmse / (idx + 1)

        losses.append(total_loss)
        train_rmses.append(total_train_rmse)
        wandb.log({"Training Loss":  total_loss, "RMSE" : total_train_rmse })
        torch.save(model.state_dict(), result_path + ".pt")
        np.savez(result_path, losses=losses, train_rmses=train_rmses)

        print("(", (epoch + 1), ") Training Loss: %.1f" % total_loss, ", RMSE, : %.1f" % total_train_rmse)
        
    
