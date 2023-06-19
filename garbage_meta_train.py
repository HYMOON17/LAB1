# Import Libraries
from torch.nn.modules import loss
from utilities import Noise2Self
from model_config import DnCNN

import torch
from torch.optim import Adam
import random
import numpy as np

device = torch.device('cuda')

class Meta_trainer():
    def __init__(self, args):
        '''
        Args:
            loss_criterion (str)   : Loss Function to be used for meta training


        '''
        super(Meta_trainer,self).__init__()
        self.meta_batch_size = args.meta_batch_size
        self.loss_criterion  = args.loss_criterion
        self.pretrained      = args.pretrained
        self.num_of_epochs   = args.num_of_epochs
        self.model           = DnCNN(1,8)
        self.model           = self.model.to(device)

        if self.pretrained is not None:
            self.model.load_state_dict(torch.load(self.pretrained, map_location='cpu'))
            print("Pretrained Model Loaded.")
        self.optimizer       = Adam(self.model.parameters(), lr=args.lr_beta)
        self.work_dir = "Results/"+args.loss_criterion+"_MTL"
        # self.lr_alpha= lr_alpha
        # self.step_size = step_size
        # self.gamma = gamma
        if self.loss_criterion =='noise2self':
            self.loss_fn = Noise2Self()
        
        self.losses      = []   # stores loss per epoch
        self.train_rmses = []   # stores rmse per epoch
            


    def train(self):

        for epoch in range(self.num_of_epochs):
            task_batch = random.sample(range(0,100),self.meta_batch_size)

            # Loss_b corresponds to Sum_loss_theta_i
            meta_loss = 0
            meta_rmse = 0
            for idx in task_batch:
                loss_b = 0
                rmse_b = 0
                # Compute inner gradient and updates
                # Collect the meta-loss for each task
                loss_b, rmse_b = self.loss_fn.get_meta_loss(weights=self.model.state_dict(),task_idx= idx,num_of_updates=1)
                meta_loss += loss_b
                meta_rmse += rmse_b
            
            # meta_update
            self.meta_update(meta_loss=meta_loss)
            self.save_model(epoch=epoch,meta_loss=meta_loss,meta_rmse=meta_rmse)


    def meta_update(self,meta_loss):
        '''
        Meta Learner is updated in this function using the loss from tasks test samples.

        Args:
            meta_loss (tensor) : Accumulated loss from task test samples per meta batch.
        '''
        # meta_loss = torch.Tensor
        meta_loss.set_(torch.Tensor([meta_loss]).to(device)) 
        meta_loss.requires_grad = True 
        self.model.train()
        self.optimizer.zero_grad()
        meta_loss.backward()
        self.optimizer.step()

    
    def save_model(self,epoch,meta_loss,meta_rmse):
        '''
        Upon calls, saves the current model to a specified location.

        Args:

        '''
        meta_loss_ = meta_loss.item()/self.meta_batch_size
        meta_rmse_ = meta_rmse/self.meta_batch_size
        self.losses.append(meta_loss_)
        self.train_rmses.append(meta_rmse_)
        torch.save(self.model.state_dict(), self.work_dir + ".pt")
        np.savez(self.work_dir, losses=self.losses, train_rmses=self.train_rmses)

        print("(", (epoch + 1), ") Training Loss: %.1f" % meta_loss_, ", RMSE, : %.1f" % meta_rmse_)
    



# if __name__=="__main__":
#     trainer  = Meta_trainer()
