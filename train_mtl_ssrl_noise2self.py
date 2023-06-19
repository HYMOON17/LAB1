'''
This code is written to implement MAML in SSRL noise2self framework. Single gradient Update
It works!!!!!!
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
from torch.nn import MSELoss
from torch.optim import Adam

device = torch.device('cuda')
set_seed(100)

num_of_layers = 8
# batch_size = 2
lr_alpha = 1e-2
lr_beta = 1e-1
step_size = 10
gamma = 0.95

num_epoch = 500




masker = Masker(width=4, mode='interpolate')



model = DnCNN(1, num_of_layers=num_of_layers)
model = model.to(device)
model.load_state_dict(torch.load("trained_models/ours_ssrl_noise2self.pt", map_location='cpu'))

model_gx = DnCNN(1, num_of_layers=num_of_layers)
model_gx.load_state_dict(torch.load("./trained_models/noise2self_checkerboard.pt", map_location='cpu'))

model = model.to(device)
model_gx = model_gx.to(device)
model_gx.eval()

loss_function = MSELoss()
# loss_function_mean = MSELoss(reduction='mean')
# loss_function_sum = MSELoss(reduction='sum')

optimizer = Adam(model.parameters(), lr=lr_beta)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


losses = []
train_rmses = []
result_path = "Results/ours_ssrl_noise2selfMTL"



# MTL Parameters
train_dataset = mydataset('data/train_imgs_mtl.mat')
num_of_tasks = 100
sample_per_task = np.floor(len(train_dataset)/num_of_tasks).astype(np.int64)


model_temp = DnCNN(1, num_of_layers=num_of_layers)
model_temp.to(device)

# SSRL specific process

kernel = torch.Tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], (0.0, 1.0, 0.0)])
kernel_inv = torch.ones(kernel.shape) - kernel

kernel = kernel[np.newaxis, np.newaxis, :, :]
kernel = kernel / kernel.sum()

kernel_inv = kernel_inv[np.newaxis, np.newaxis, :, :]
kernel_inv = kernel_inv / kernel_inv.sum()

kernel = kernel.to(device)
kernel_inv = kernel_inv.to(device)

replicate_unit = torch.Tensor([[0.0, 1.0], [1.0, 0.0]])
S_mask = replicate_unit.repeat(256, 256)
S_mask = S_mask.to(device)
S_mask_inv = torch.ones(S_mask.shape).to(device) - S_mask

# reflection padding
m = torch.nn.ReflectionPad2d(1)





for epoch in range(num_epoch):

    # Get Tasks from the dataset
    # task_data = train_dataset.get_task(sample_per_task)
    # a: elements related to task training 
    # b: elements related to meta update
    # model.train()
    task_list = random.sample(range(0,100),num_of_tasks)
    total_loss = 0
    total_train_rmse = 0
    loss_b = 0
    train_rmse_b = 0
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

        # Compute the loss function   
        
        # reflection padding
        noisy_images_padded = m(noisy_images)

        # interpolation
        filtered_tensor = torch.nn.functional.conv2d(noisy_images_padded, kernel, stride=1, padding=0)
        net_input1 = filtered_tensor * S_mask + noisy_images * S_mask_inv
        net_input2 = filtered_tensor * S_mask_inv + noisy_images * S_mask

        net_output1 = model_temp(net_input1)
        net_output2 = model_temp(net_input2)

        with torch.no_grad():
            net_output_gx1 = model_gx(net_input1)
            net_output_gx2 = model_gx(net_input2)

        loss1a = loss_function(net_output1, net_output_gx2)
        loss2a = loss_function(net_output2, net_output_gx1)
        loss_temp = (loss1a + loss2a) / 2.0

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

        # Compute Loss for task test samples
        # reflection padding
        noisy_images_padded = m(noisy_images)

        # interpolation
        filtered_tensor = torch.nn.functional.conv2d(noisy_images_padded, kernel, stride=1, padding=0)
        net_input1 = filtered_tensor * S_mask + noisy_images * S_mask_inv
        net_input2 = filtered_tensor * S_mask_inv + noisy_images * S_mask


        with torch.no_grad():
            denoised = model_temp(noisy_images)

            net_output1 = model_temp(net_input1)
            net_output2 = model_temp(net_input2)

            net_output_gx1 = model_gx(net_input1)
            net_output_gx2 = model_gx(net_input2)

        loss1b = loss_function(net_output1, net_output_gx2)
        loss2b = loss_function(net_output2, net_output_gx1)
        task_loss_b = (loss1b + loss2b) / 2.0        
        
        loss_b += task_loss_b.clone()
        denoised = np.squeeze(denoised.detach().cpu().numpy().astype(np.float64))
        clean_image = np.squeeze(clean_images.cpu().numpy().astype(np.float64))

        train_rmse = np.sqrt(mean_squared_error(denoised, clean_image))
        train_rmse_b += (train_rmse)


# Re Check this section
    loss1b = loss_function(net_output1, net_output_gx2)
    loss2b = loss_function(net_output2, net_output_gx1)
    meta_loss = (loss1b + loss2b) / 2.0     #dummy loss
           
    meta_loss.set_(torch.Tensor([loss_b]).to(device)) 
    meta_loss.requires_grad = True 
    model.train()
    optimizer.zero_grad()
    meta_loss.backward()
    optimizer.step()

    total_loss = meta_loss.item()/(num_of_tasks)
    scheduler.step()

    # total_loss = total_loss / (idx + 1)
    # total_train_rmse = total_train_rmse / (idx + 1)
    total_train_rmse=train_rmse_b/num_of_tasks


    losses.append(total_loss)
    train_rmses.append(total_train_rmse)

    torch.save(model.state_dict(), result_path + ".pt")
    np.savez(result_path, losses=losses, train_rmses=train_rmses)

    print("(", (epoch + 1), ") Training Loss: %.1f" % total_loss, ", RMSE, : %.1f" % total_train_rmse)