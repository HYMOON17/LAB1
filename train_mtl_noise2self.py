'''
This code is written to implement MAML in noise2self framework. Single gradient Update
It works!!!!!!
Problems in loss_b and train_rmse. Should be outside the loop.(solved)
'''
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import random
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error
from scipy.io import loadmat
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset as dset
from model_config import DnCNN
from mask import Masker
from utilities.mydataset import mydataset
from utilities.utils import set_seed
import wandb
wandb.init(project="mtl_ssrl",
           config={
    "num_of_layers" : 8,
    "lr_alpha" : 1e-1,
    "lr_beta" : 1e-1,
    'step_size' : 10,
    "gamma" : 0.95,
    "num_epoch" : 500,})

device = torch.device('cuda')
set_seed(100)



num_of_layers = 8
# batch_size = 2
lr_alpha = 1e-1
lr_beta = 1e-1
step_size = 10
gamma = 0.95

num_epoch = 100 #500




masker = Masker(width=4, mode='interpolate')



model = DnCNN(1, num_of_layers=num_of_layers)
model = model.to(device)
model.load_state_dict(torch.load("trained_models/noise2self.pt", map_location='cpu'))

from torch.nn import MSELoss
from torch.optim import Adam

loss_function_mean = MSELoss(reduction='mean')
loss_function_sum = MSELoss(reduction='sum')

optimizer = Adam(model.parameters(), lr=lr_beta)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


losses = []
train_rmses = []
result_path = "Results/noise2selfMTL"



# MTL Parameters
train_dataset = mydataset('data/train_imgs_mtl.mat')
num_of_tasks = 100
sample_per_task = np.floor(len(train_dataset)/num_of_tasks).astype(np.int64)


model_temp = DnCNN(1, num_of_layers=num_of_layers)
model_temp.to(device)

for epoch in range(num_epoch):

    # Get Tasks from the dataset
    # task_data = train_dataset.get_task(sample_per_task)
    # a: elements related to task training 
    # b: elements related to meta update

    task_list = random.sample(range(0,100),num_of_tasks)
    total_loss = 0
    total_train_rmse = 0
    loss_b = 0
    train_rmse_b = 0
    wandb_clean_img, wandb_noisy_img, wandb_net_input = [],[],[] #
    
    # For each task, generate theta_i
    # for idx in range(len(task_data)):
    for idx in (task_list):

        #Temporary model creation for theta_i
        # model_temp = DnCNN(1, num_of_layers=num_of_layers)
        # model_temp.to(device)
        model_temp.load_state_dict(model.state_dict())

        optimizer_temp = Adam(model_temp.parameters(), lr=lr_alpha)
        # scheduler_temp = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        model_temp.train()
        
        # Get task specific data
        # task_i = (task_data[idx])

        task_i = train_dataset.get_task_samples(idx)
        
        # Randomly selectly train and test samples for each task
        num_of_tr_samples = np.floor(sample_per_task*0.4).astype(np.int16)
        tr_sample_idx = random.sample(range(0, sample_per_task), num_of_tr_samples)
        ts_sample_idx = [i for i in range(5)  if i not in tr_sample_idx ] 
        
        # Convert the images into tensor and form them as batch
        clean_img, noisy_img = [],[]
        for i in tr_sample_idx:
            clean_img_tmp, noisy_img_tmp = task_i[i]
            
            clean_img.append(clean_img_tmp)
            noisy_img.append(noisy_img_tmp)

        clean_img = np.asarray(clean_img)
        noisy_img = np.asarray(noisy_img)

        clean_images = torch.from_numpy(clean_img)
        noisy_images = torch.from_numpy(noisy_img)
        

        # move the data into the device
        noisy_images = noisy_images.to(device, dtype=torch.float)  
        wandb_clean_img.append(wandb.Image(noisy_images)) #
        # Compute the loss function   
        
        net_input, mask = masker.mask(noisy_images, idx)

        wandb_net_input.append(wandb.Image(net_input)) #
        net_output = model_temp(net_input)
        denoised = model_temp(noisy_images)   

        

        loss_temp = loss_function_sum(net_output * mask, noisy_images * mask) / (2 * torch.sum(mask))

        # denoised = np.squeeze(denoised.detach().cpu().numpy().astype(np.float64))
        # clean_image = np.squeeze(clean_images.cpu().numpy().astype(np.float64))

        # train_rmse = np.sqrt(mean_squared_error(denoised, clean_image))
        # total_train_rmse += train_rmse

        optimizer_temp.zero_grad()
        loss_temp.backward()
        optimizer_temp.step()


        # Generate losses for each task's testing samples. It will be used to update the theta param
        # Collect test samples and Convert the images into tensor and form them as batch
        
        clean_img, noisy_img = [],[]
        for i in ts_sample_idx:
            clean_img_tmp, noisy_img_tmp = task_i[i]
            
            clean_img.append(clean_img_tmp)
            noisy_img.append(noisy_img_tmp)

        clean_img = np.asarray(clean_img)
        noisy_img = np.asarray(noisy_img)

        clean_images = torch.from_numpy(clean_img)
        noisy_images = torch.from_numpy(noisy_img)

        noisy_images = noisy_images.to(device, dtype=torch.float)

        net_input, mask = masker.mask(noisy_images, idx)
        
        
        with torch.no_grad():    
            net_output = model_temp(net_input)
            denoised = model_temp(noisy_images)


        #Collects loss for each task testing samples
            task_loss_b = loss_function_sum(net_output * mask, noisy_images * mask) / (3 * torch.sum(mask))


        loss_b += task_loss_b.clone()
        # print("loss_b : " ,loss_b)
        wandb.log({"loss_b": loss_b})
        denoised = np.squeeze(denoised.detach().cpu().numpy().astype(np.float64))
        clean_image = np.squeeze(clean_images.cpu().numpy().astype(np.float64))

        train_rmse = np.sqrt(mean_squared_error(denoised, clean_image))
        train_rmse_b += (train_rmse)


# Re Check this section
    meta_loss = loss_function_sum(net_output * mask, noisy_images * mask) / (3 * torch.sum(mask)) #dummy loss
    # print("meta_loss : " ,  meta_loss)
    
    meta_loss.set_(torch.Tensor([loss_b]).to(device)) 
    meta_loss.requires_grad = True 
    model.train()
    optimizer.zero_grad()
    meta_loss.backward()
    optimizer.step()

    total_loss = meta_loss.item()/(num_of_tasks)
    scheduler.step()
    wandb.log({"meta_loss": meta_loss})
    # total_loss = total_loss / (idx + 1)
    # total_train_rmse = total_train_rmse / (idx + 1)
    total_train_rmse=train_rmse_b/num_of_tasks


    losses.append(total_loss)
    train_rmses.append(total_train_rmse)

    torch.save(model.state_dict(), result_path + ".pt")
    np.savez(result_path, losses=losses, train_rmses=train_rmses)

    print("(", (epoch + 1), ") Training Loss: %.1f" % total_loss, ", RMSE, : %.1f" % total_train_rmse)
    wandb.log({"Training Loss":  total_loss, "RMSE" : total_train_rmse })


wandb.finish()

