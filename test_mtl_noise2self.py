
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


device = torch.device('cuda')
set_seed(100)



num_of_layers = 8
# batch_size = 2
lr_alpha = 1e-1
lr_beta = 1e-1
step_size = 10
gamma = 0.95

num_epoch = 500




masker = Masker(width=4, mode='interpolate')

model = DnCNN(1, num_of_layers=num_of_layers)
model = model.to(device)
model.load_state_dict(torch.load("Results/noise2selfMTL.pt", map_location='cpu'))

from torch.nn import MSELoss
from torch.optim import Adam

loss_function_mean = MSELoss(reduction='mean')
loss_function_sum = MSELoss(reduction='sum')

# MTL Parameters
test_dataset = mydataset('data/test_imgs_mtl.mat')
num_of_tasks = 20
sample_per_task = np.floor(len(test_dataset)/num_of_tasks).astype(np.int64)


model_temp = DnCNN(1, num_of_layers=num_of_layers)
model_temp.to(device)



# Get Tasks from the dataset
# task_data = train_dataset.get_task(sample_per_task)
# a: elements related to task training 
# b: elements related to meta update

task_list = random.sample(range(0,20),num_of_tasks)
total_test_rmse_b = []

    # For each task, generate theta_i
    # for idx in range(len(task_data))
for itr in range(10):  
    test_rmse_b = 0  
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
        task_i = test_dataset.get_task_samples(idx)

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
        
        net_input, mask = masker.mask(noisy_images, idx)

        
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

        # net_input, mask = masker.mask(noisy_images, idx)
        
        model_temp.eval()
        with torch.no_grad():    
            # net_output = model_temp(net_input)
            denoised = model_temp(noisy_images)


        denoised = np.squeeze(denoised.detach().cpu().numpy().astype(np.float64))
        clean_image = np.squeeze(clean_images.cpu().numpy().astype(np.float64))

        test_rmse_b += np.sqrt(mean_squared_error(denoised, clean_image))


        # total_loss = total_loss / (idx + 1)
        # total_train_rmse = total_train_rmse / (idx + 1)
    total_test_rmse_b.append(test_rmse_b/num_of_tasks)
    print("%s: %.1f" % ("Noise2Self", total_test_rmse_b[itr]))

print("%s: %.1f  %s: %.1f" % ("Mean", np.mean(total_test_rmse_b),"Std: ", np.std(total_test_rmse_b)))