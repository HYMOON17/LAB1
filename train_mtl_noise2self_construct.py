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
import torch.nn  as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset as dset
# from model_config.dncnn_revised import DnCNN
from torchmeta.model_hsm import ConvolutionalNeuralNetwork as DnCNN
from mask import Masker
from utilities.mydataset import mydataset
from utilities.utils import set_seed
from torch.nn import MSELoss
from torch.optim import Adam
from collections import OrderedDict
from torchmeta.utils.gradient_based import gradient_update_parameters
import wandb
'''
# wandb.init(project="mtl_ssrl",
#            config={
#     "num_of_layers" : 8,
#     "lr_alpha" : 1e-1,
#     "lr_beta" : 1e-1,
#     'step_size' : 10,
#     "gamma" : 0.95,
#     "num_epoch" : 500,})
'''
device = torch.device('cuda')
set_seed(100)
'''
    # For each task, generate theta_i
    # for idx in range(len(task_data)):
    #     for idx in (task_list):

    #         #Temporary model creation for theta_i
    #         model_temp.load_state_dict(model.state_dict())
    #         optimizer_temp = Adam(model_temp.parameters(), lr=lr_alpha)
    #         model_temp.train()
        
    #         # Get task specific data

    #         task_i = train_dataset.get_task_samples(idx)
        
    #         # Randomly selectly train and test samples for each task
    #         num_of_tr_samples = np.floor(sample_per_task*0.4).astype(np.int16)
    #         tr_sample_idx = random.sample(range(0, sample_per_task), num_of_tr_samples)
    #         ts_sample_idx = [i for i in range(5)  if i not in tr_sample_idx ] 
        
    #         # Convert the images into tensor and form them as batch
    #         clean_img, noisy_img = [],[]
    #         for i in tr_sample_idx:
    #             clean_img_tmp, noisy_img_tmp = task_i[i]
    #             clean_img.append(clean_img_tmp)
    #             noisy_img.append(noisy_img_tmp)

    #         clean_img = np.asarray(clean_img)
    #         noisy_img = np.asarray(noisy_img)
    #         clean_images = torch.from_numpy(clean_img)
    #         noisy_images = torch.from_numpy(noisy_img)

    #         # move the data into the device
    #         noisy_images = noisy_images.to(device, dtype=torch.float)  
        
    #         # wandb_noisy_img.append(wandb.Image(noisy_images)) #
    #         # wandb.log({"First Noisy_images" : wandb_noisy_img})
        
    #         # Compute the loss function   
    #         net_input, mask = masker.mask(noisy_images, idx)
    #         net_output = model_temp(net_input)
    #         denoised = model_temp(noisy_images)   
        
    #         # wandb_mask.append(wandb.Image(mask)) #
    #         # wandb.log({"First mask" : wandb_mask})
    #         # wandb_net_input.append(wandb.Image(net_input)) #
    #         # wandb.log({"First input images" : wandb_net_input})
    #         # wandb_net_output.append(wandb.Image(net_output)) #
    #         # wandb.log({"First output images" : wandb_net_output})
    #         # wandb_denoised.append(wandb.Image(denoised)) #
    #         # wandb.log({"First denoised images" : wandb_denoised})

    #         loss_temp = loss_function_sum(net_output * mask, noisy_images * mask) / (2 * torch.sum(mask))
    #         wandb.log({"loss_temp(First)":loss_temp})
        
    #         optimizer_temp.zero_grad()
    #         loss_temp.backward()
    #         optimizer_temp.step()
        
    #         # Generate losses for each task's testing samples. It will be used to update the theta param
    #         # Collect test samples and Convert the images into tensor and form them as batch
        
    #         clean_img, noisy_img = [],[]
    #         for i in ts_sample_idx:
    #             clean_img_tmp, noisy_img_tmp = task_i[i]
            
    #             clean_img.append(clean_img_tmp)
    #             noisy_img.append(noisy_img_tmp)

    #         clean_img = np.asarray(clean_img)
    #         noisy_img = np.asarray(noisy_img)
    #         clean_images = torch.from_numpy(clean_img)
    #         noisy_images = torch.from_numpy(noisy_img)

    #         # inner_noisy_img.append(wandb.Image(noisy_images)) #
    #         # wandb.log({"inner Noisy_images" : inner_noisy_img})
    #         noisy_images = noisy_images.to(device, dtype=torch.float)
    #         net_input, mask = masker.mask(noisy_images, idx)
        
    #         with torch.no_grad():    
    #             net_output = model_temp(net_input)
    #             denoised = model_temp(noisy_images)

    #         #Collects loss for each task testing samples
    #             task_loss_b = loss_function_sum(net_output * mask, noisy_images * mask) / (3 * torch.sum(mask))
        
    #         # inner_net_input.append(wandb.Image(net_input)) #
    #         # wandb.log({"inner input images" : inner_net_input})
    #         # inner_net_output.append(wandb.Image(net_output)) #
    #         # wandb.log({"inner output images" : inner_net_output})
    #         # inner_denoised.append(wandb.Image(denoised)) #
    #         # wandb.log({"inner denoised images" : inner_denoised})

    #         loss_b += task_loss_b.clone()
    #         # print("loss_b : " ,loss_b)
    #         wandb.log({"loss_b": loss_b})
    #         denoised = np.squeeze(denoised.detach().cpu().numpy().astype(np.float64))
    #         clean_image = np.squeeze(clean_images.cpu().numpy().astype(np.float64))

    #         train_rmse = np.sqrt(mean_squared_error(denoised, clean_image))
    #         train_rmse_b += (train_rmse)
    #         for (name, param), grad in zip(params.items(), grads):
    #             params[name] = param - step_size * grad


    # # Re Check this section
    #     meta_loss = loss_function_sum(net_output * mask, noisy_images * mask) / (3 * torch.sum(mask)) #dummy loss
    #     # print("meta_loss : " ,  meta_loss)

    #     meta_loss.set_(torch.Tensor([loss_b]).to(device)) 
    #     meta_loss.requires_grad = True 
    #     model.train()
    #     optimizer.zero_grad()
    #     meta_loss.backward()
    #     optimizer.step()
    #     # wandb.log({"opt_loss": optimizer}) - 이거 안됨
    #     total_loss = meta_loss.item()/(num_of_tasks)
    #     scheduler.step()
    #     wandb.log({"meta_loss": meta_loss})
    #     # total_loss = total_loss / (idx + 1)
    #     # total_train_rmse = total_train_rmse / (idx + 1)
    #     total_train_rmse=train_rmse_b/num_of_tasks


    #     losses.append(total_loss)
    #     train_rmses.append(total_train_rmse)

    #     torch.save(model.state_dict(), result_path + ".pt")
    #     np.savez(result_path, losses=losses, train_rmses=train_rmses)

    #     print("(", (epoch + 1), ") Training Loss: %.1f" % total_loss, ", RMSE, : %.1f" % total_train_rmse)
    #     wandb.log({"Training Loss":  total_loss, "RMSE" : total_train_rmse })


    # wandb.finish()
'''
    # 여기 K 수정해야됨
class MAML():
    def __init__(self, model,  inner_lr, meta_lr, K=10, inner_steps=1, tasks_per_meta_batch=1000):
        
        # important objects
        # self.tasks = tasks -- origin
        self.model = model
        self.weights = list(model.parameters()) # the maml weights we will be meta-optimising
        self.criterion = nn.MSELoss()
        self.meta_optimiser = torch.optim.Adam(self.weights, meta_lr)
        
        # hyperparameters
        self.idx=0
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.K = K
        self.inner_steps = inner_steps # with the current design of MAML, >1 is unlikely to work well 
        self.tasks_per_meta_batch = tasks_per_meta_batch 
        
        # metrics
        self.plot_every = 10
        self.print_every = 500
        self.meta_losses = []
    
    def task_sample_task(self,task_i, sample_idx):
        clean_img, noisy_img = [],[]
        for i in sample_idx:
            clean_img_tmp, noisy_img_tmp = task_i[i]
            
            clean_img.append(clean_img_tmp)
            noisy_img.append(noisy_img_tmp)

        clean_img = np.asarray(clean_img)
        noisy_img = np.asarray(noisy_img)
        clean_images = torch.Tensor(clean_img)
        noisy_images = torch.Tensor(noisy_img)
        # print(noisy_images.size())
        return clean_images,noisy_images

    
    def inner_loop(self,task_i, tr_sample_idx,ts_sample_idx,idx):
        # reset inner model to current maml weights
        temp_weights = [w.clone() for w in self.weights]

        # perform training on data sampled from task
        # Train
        Train_clean_images,Test_noisy_images=[],[]
        Train_clean_images, Train_noisy_images = self.task_sample_task(task_i,tr_sample_idx)
        # net_input, mask = masker.mask(Train_noisy_images, idx)
        Train_noisy_images = Train_noisy_images.to(device, dtype=torch.float) 
        Train_clean_images = Train_clean_images.to(device, dtype=torch.float)   
        # net_output = self.model(net_input)
        Train_denoised = self.model(Train_noisy_images)
        
        # Train_denoised = torch.tensor(Train_denoised, dtype = torch.float32)
        for step in range(self.inner_steps):
            # loss = self.criterion(Train_denoised,Train_clean_images) / self.K
            
            loss = self.criterion(Train_denoised,Train_clean_images) / self.K
            # compute grad and update inner loop weights
            # grad = torch.autograd.grad(loss, temp_weights) --origin
            grad = torch.autograd.grad(loss, temp_weights,allow_unused=True)
            model.zero_grad()
            # temp_weights = [w - self.inner_lr * g for w, g in zip(temp_weights, grad)]
            # 여기 업데이트 하는게 conv 꼴이 아니여서 문제가 있음 타 파일 필요
            params = gradient_update_parameters(self.model,
                                                    loss,
                                                    step_size=0.4,
                                                    first_order=True)
        # sample new data for meta-update and compute loss
        
        Test_clean_images,Test_noisy_images=[],[]
        Test_clean_images, Test_noisy_images = self.task_sample_task(task_i,ts_sample_idx)
        Test_noisy_images = Test_noisy_images.to(device, dtype=torch.float) 
        Test_clean_images = Test_clean_images.to(device, dtype=torch.float)   
        Test_denoised = self.model(Test_noisy_images,params=params)
        loss = self.criterion(Test_clean_images,Test_denoised) / self.K
        model.zero_grad()
        return loss
    
    def main_loop(self, num_iterations):
        epoch_loss = 0
        
        for iteration in range(1, num_iterations+1):
            model.zero_grad()
            # compute meta loss
            meta_loss = 0
            task_list = random.sample(range(0,100),num_of_tasks)
            # print(task_list)
            # for i in range(self.tasks_per_meta_batch):
            for idx in (task_list):
                task_i = train_dataset.get_task_samples(idx)
                num_of_tr_samples = np.floor(sample_per_task*0.4).astype(np.int16)
                tr_sample_idx = random.sample(range(0, sample_per_task), num_of_tr_samples)
                ts_sample_idx = [i for i in range(5)  if i not in tr_sample_idx ]
                meta_loss += self.inner_loop(task_i, tr_sample_idx,ts_sample_idx,idx)
            
            # compute meta gradient of loss with respect to maml weights
            meta_grads = torch.autograd.grad(meta_loss, self.weights)
            
            # assign meta gradient to weights and take optimisation step
            for w, g in zip(self.weights, meta_grads):
                w.grad = g
            self.meta_optimiser.step()
            
            # log metrics
            epoch_loss += meta_loss.item() / self.tasks_per_meta_batch
            
            if iteration % self.print_every == 0:
                print("{}/{}. loss: {}".format(iteration, num_iterations, epoch_loss / self.plot_every))
            
            if iteration % self.plot_every == 0:
                self.meta_losses.append(epoch_loss / self.plot_every)
                epoch_loss = 0



    # Get Tasks from the dataset
    # task_data = train_dataset.get_task(sample_per_task)
    # a: elements related to task training 
    # b: elements related to meta update


    
if __name__ == '__main__':
    # wandb.init(project="mtl_ssrl")
    wandb.init(mode="disabled")
    import argparse
    set_seed(100)
    

    # MTL Parameters
    train_dataset = mydataset('data/train_imgs_mtl.mat')
    num_of_tasks = 100
    sample_per_task = np.floor(len(train_dataset)/num_of_tasks).astype(np.int64)

    '''
    # parser = argparse.ArgumentParser('Model-Agnostic Meta-Learning (MAML)')

    # parser.add_argument('folder', type=str,
    #     help='Path to the folder the data is downloaded to.')
    # parser.add_argument('--num-shots', type=int, default=5,
    #     help='Number of examples per class (k in "k-shot", default: 5).')
    # parser.add_argument('--num-ways', type=int, default=5,
    #     help='Number of classes per task (N in "N-way", default: 5).')

    # parser.add_argument('--first-order', action='store_true',
    #     help='Use the first-step approximation of MAML.')
    # parser.add_argument('--step-size', type=float, default=0.4,
    #     help='Step-size for the gradient step for adaptation (default: 0.4).')
    # parser.add_argument('--hidden-size', type=int, default=64,
    #     help='Number of channels for each convolutional layer (default: 64).')

    # parser.add_argument('--output-folder', type=str, default=None,
    #     help='Path to the output folder for saving the model (optional).')
    # parser.add_argument('--batch-size', type=int, default=2,
    #     help='Number of tasks in a mini-batch of tasks (default:2 --origin 16).')
    # parser.add_argument('--num-batches', type=int, default=100,
    #     help='Number of batches the model is trained over (default: 100).')
    # parser.add_argument('--num-workers', type=int, default=1,
    #     help='Number of workers for data loading (default: 1).')
    # parser.add_argument('--num-layers', type=int, default=8,
    #     help='Number of layers for dncnn (default: 8).')
    # # parser.add_argument('--download', action='store_true',
    # #     help='Download the Omniglot dataset in the data folder.')
    # parser.add_argument('--use-cuda', action='store_true',
    #     help='Use CUDA if available.')

    # args = parser.parse_args()
    # args.device = torch.device('cuda' if args.use_cuda
    #     and torch.cuda.is_available() else 'cpu')

    # train(args)
    '''

    num_of_layers = 8
    # batch_size = 2
    lr_alpha = 1e-1
    lr_beta = 1e-1
    step_size = 10
    gamma = 0.95
    num_epoch = 100 #500
    masker = Masker(width=4, mode='interpolate')

    losses = []
    train_rmses = []
    result_path = "Results/noise2selfMTL"
    model = DnCNN(1, 1)
    model = model.to(device)
    # model = model.to(args.device)
    wandb.watch(model)
    # model.load_state_dict(torch.load("trained_models/noise2self.pt", map_location='cpu'))
    model.train()
    maml = MAML(model, inner_lr=0.01, meta_lr=0.01)
    maml.main_loop(num_iterations=100)


    # model = DnCNN(1, num_of_layers=num_of_layers)
    # model = DnCNN()
    # optimizer = Adam(model.parameters(), lr=lr_beta)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    

    

    # model_temp = DnCNN(1, num_of_layers=num_of_layers)
    # model_temp.to(device)

    # wanb 용 그림 시각화
    # wandb_clean_img, wandb_noisy_img, wandb_net_input,wandb_net_output,wandb_mask,wandb_denoised = [],[],[],[],[],[]#
    # inner_clean_img, inner_noisy_img, inner_net_input,inner_net_output,inner_mask,inner_denoised = [],[],[],[],[],[]#

    # maml = MAML(DnCNN(), tasks, inner_lr=0.01, meta_lr=0.001)
    # maml.main_loop(num_iterations=10000) --origin
    
    
    