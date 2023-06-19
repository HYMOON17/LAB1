'''
This script performs Meta learning in image denoising domain.
Gradient based update is done using MSE loss.
The core motivation is taken from MAML (Model Agnostic Meta Learning) paper.

'''
# Import Libraries
import os
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from skimage.metrics import mean_squared_error

from utilities.mydataset import mydataset
from model_config import DnCNN
from utilities.utils import set_seed

os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device('cuda')
set_seed(100)
compute_mse_loss = MSELoss()

class Meta_train():
    def __init__(self,args):
        self.args            = args
        self.dataset         = mydataset(args.data_path)# Dataset Object 
        self.num_of_tasks    = args.num_of_tasks       # total number of tasks in the dataset. For our case, it is 100.
        self.meta_batch_size = args.meta_batch_size    # Number of tasks per meta update
        self.step_size       = args.step_size          # Meta update scheduler parameter
        self.gamma           = args.gamma              # Meta update schudeler parameter
        self.lr_beta         = args.lr_beta            # Meta update learning rate
        self.lr_alpha        = args.lr_alpha           # Inner update learning rate
        self.num_of_updates  = args.num_of_updates     # Number of inner updates
        
        self.model           = DnCNN(1,8)              # Initialize model
        self.model           = self.model.to(device)   # Place model into GPU

        # if the starting point of meta training is not random, load the weights from the provided pretrained model.
        if args.pretrained is not None:
            self.model.load_state_dict(torch.load(args.pretrained, map_location='cpu'))
            print("Pretrained Model Loaded.")

        # Training stopping criterion
        self.stop_training   = False
        self.prev_loss       = 1e-1


    def train(self):
        '''
        Performs Meta training.
        '''
        # Creates results directory
        result_dir ="Test_Results/lr_b_"+str(self.lr_beta)
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)
            print("Created folder: ",result_dir)
        else:
            print(result_dir, " folder laready exists.")

        filename =result_dir+"/Meta_Batch_size_"+str(self.args.meta_batch_size)+"_lr_a_"+str(self.args.lr_alpha) \
        +"_updates_"+str(self.args.num_of_updates)        
        #-----------------------------------------

        losses_b    = [] #  Stores all loss values numpy value for each epoch After inner gradient update 
        rmses_b     = [] #  Stores all loss values numpy value for each epoch After inner gradient update 
        param_diff = []  #  Stores the parameter differences between models before and after meta update 


        # Optimizer and scheduler for meta update
        optimizer = Adam(self.model.parameters(), lr=self.lr_beta)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma =self.gamma)

        for itr in range(self.args.num_of_epochs):

            #3. Sample batch of tasks    
            # task_list = random.sample(range(0,self.num_of_tasks),self.args.meta_batch_size)
            task_list = [t for t in range(self.meta_batch_size)]

            #-------------------------------------------------------
            # This block is written to keep ttrack of the previous model parameters.
            # this info will be used to compute the parameter difference to see the model progress
            data1,data2 =[],[] 
            copy_model = copy.deepcopy(self.model)
            for param1 in copy_model.parameters():
                data1.append(param1.data)
            del copy_model
            #---------------------------------------------------------

            train_rmse_b = 0.0 # stores rmse for all tasks per epoch
            meta_loss = 0  # stores loss for all tasks per epoch

            for task_idx in (task_list): #4. For all the tasks in the batch do:
                # 5. Collect data for the task 
                # Get training and test samples
                tr_samples, te_samples = self.generate_data(task_idx)

                # Separate input noisy images and output clean images
                clean_images_te, noisy_images_te = te_samples
                clean_images_te_cpu = np.squeeze(clean_images_te.cpu().numpy().astype(np.float64))
                clean_images_tr, noisy_images_tr = tr_samples

                # end data gathering and processing
                
                # Create a temp copy of the model of inner update
                task_model = copy.deepcopy(self.model)
                task_optimizer = Adam(task_model.parameters(), lr=self.lr_alpha)

                for i in range(self.num_of_updates):
                    task_model.train()
                    denoised_tr = task_model(noisy_images_tr) 
                    loss_temp = compute_mse_loss(denoised_tr,clean_images_tr)
                    task_optimizer.zero_grad()
                    loss_temp.backward()
                    task_optimizer.step()
                    
                    denoised_te = task_model(noisy_images_te) 
                    loss_te = compute_mse_loss(denoised_te,clean_images_te)

                    denoised_te_cpu = np.squeeze(denoised_te.detach().cpu().numpy().astype(np.float64))
                    rmse_te = np.sqrt(mean_squared_error(denoised_te_cpu, clean_images_te_cpu))
  
                del task_model        
                meta_loss += loss_te
                train_rmse_b += rmse_te

            # Update Model with the meta loss
            self.model.train()            
            denoised = self.model(noisy_images_te)
            loss = compute_mse_loss(denoised,clean_images_te)
            
            meta_loss = meta_loss/self.args.meta_batch_size
            rmses_b.append(train_rmse_b/self.args.meta_batch_size)
            loss.data = torch.tensor(0.0)
            total_loss = loss +meta_loss

            total_loss.backward()
            optimizer.step()
            scheduler.step()
            #------------------------------------
            for param1 in self.model.parameters():
                data2.append(param1.data)
            d=0.0

            for i in range (len(data1)):
                d+=compute_mse_loss(data1[i],data2[i])
            #------------------------------------
            print("Epoch: ", itr+1,"Meta_loss: ",meta_loss, "Param Erorr",d)
            param_diff.append(d.detach().cpu().numpy())
            losses_b.append(total_loss.detach().cpu().numpy())

            # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            # fig.suptitle("lr_b_"+str(self.lr_beta))
            # ax1.plot(losses_b)
            # # ax1.xlabel("Epochs")
            # ax1.set_title("Meta Update Loss Curve")
            # ax2.plot(rmses_b)
            # # ax2.xlabel("Epochs")
            # ax2.set_title("Meta Update RMSE Curve")
            # ax3.plot(param_error)
            # ax3.set_title("Param rel error Curve")
            # plt.show()



            torch.save(self.model.state_dict(), filename + ".pt")
            np.savez(filename, losses_b=losses_b, rmses_b=rmses_b, param_error = param_diff)


    def generate_data(self,task_idx):
        '''
        Args:
            task_idx (int) : index of the task
        Returns:
            tr_samples (tensor.float.device) : task training samples (cln, noisy)
            te_samples (tensor.float.device) : task test samples (cln, noisy)
        '''
        # Randomly selectly train and test samples for each task
        task_i = self.dataset.get_task_samples(task_idx)

        if self.args.tr_sample_pr is None:
            tr_sample_idx = [0,1]
            te_sample_idx = [2,3,4]
        else:
            # Randomly selectly train and test samples for each task
            num_of_tr_samples = np.floor(self.args.samples_per_task*self.args.tr_sample_pr).astype(np.int16)
            tr_sample_idx = random.sample(range(0, self.args.samples_per_task), num_of_tr_samples)
            te_sample_idx = [i for i in range(self.args.samples_per_task)  if i not in tr_sample_idx ] 

        # Convert the images into tensor and form them as batch
        clean_img_tr, noisy_img_tr = [],[]
        for i in tr_sample_idx:
            clean_img_tmp, noisy_img_tmp = task_i[i]
            
            clean_img_tr.append(clean_img_tmp)
            noisy_img_tr.append(noisy_img_tmp)

        clean_img_tr = np.asarray(clean_img_tr)
        noisy_img_tr = np.asarray(noisy_img_tr)

        clean_images_tr = torch.from_numpy(clean_img_tr)
        noisy_images_tr = torch.from_numpy(noisy_img_tr) 

        # move the data into the device
        noisy_images_tr = noisy_images_tr.to(device, dtype=torch.float)  
        clean_images_tr = clean_images_tr.to(device, dtype=torch.float)   

        tr_samples  = [clean_images_tr, noisy_images_tr]

        clean_img_te, noisy_img_te = [],[]
        for i in te_sample_idx:
            clean_img_tmp, noisy_img_tmp = task_i[i]
            
            clean_img_te.append(clean_img_tmp)
            noisy_img_te.append(noisy_img_tmp)

        clean_img_te = np.asarray(clean_img_te)
        noisy_img_te = np.asarray(noisy_img_te)

        clean_images_te = torch.from_numpy(clean_img_te)
        noisy_images_te = torch.from_numpy(noisy_img_te) 

        # move the data into the device
        noisy_images_te = noisy_images_te.to(device, dtype=torch.float)  
        clean_images_te = clean_images_te.to(device, dtype=torch.float)   

        te_samples  = [clean_images_te, noisy_images_te]

        return tr_samples, te_samples



    def stopping_criterion(self,new_loss):
        '''
        Args:
        new_loss = latest loss
        Return:
        compute ths relative error between current and previous loss and changes the flag of stop_training variable
        On closing, sets the new loss to prev_loss for next iteration.
        '''
        error = np.abs((self.prev_loss-new_loss)/self.prev_loss)
        print(error)
        if error<1e-5:
            self.stop_training =True
        self.prev_loss = new_loss