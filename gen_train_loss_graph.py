# Plot the training loss and RMSE from the results variables
#  stored in "model_name.npz".
# To generate the plots, please change the folder path to
# appropriate file location (e.g., Resutls/noise2true.npz) 

from numpy import result_type
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset as dset

# Check the evolution of training loss and RMSE loss
# model_name = "noise2true_chest_abdomen"
model_name = "Results"
# Load the loss data from the results folder
filename = "Results/"+model_name+str(".npz")
data = np.load(filename)
train_loss = data['losses']
train_RMSE = data['train_rmses']

# Normalize the data
# norm = np.linalg.norm(train_loss)
# train_loss = train_loss/norm

# norm = np.linalg.norm(train_RMSE)
# train_RMSE = train_RMSE/norm

min_loss = round(np.min(train_loss),3)
min_idx_loss=np.unravel_index(np.argmin(train_loss),train_loss.shape)

min_rmse = round(np.min(train_RMSE),3)
min_idx_rmse=np.unravel_index(np.argmin(train_RMSE),train_RMSE.shape)

#plt.plot(train_loss, label='Min Training Loss: '+str(min_loss)+' epoch: '+str(min_idx_loss),color='green', linestyle='dashed',linewidth=2, markersize=1)
plt.plot(train_RMSE, label='Min RMSE loss: '+str(min_rmse)+' epoch: '+str(min_idx_rmse),color='blue', linestyle='-',linewidth=2, markersize=1)
# plt.ylim([0,0.1])
plt.xlabel('Epochs')
plt.ylabel('Loss value')
plt.legend(loc='upper right')
plt.title(model_name)
# plt.savefig('figures/'+model_name+str(".png"))

plt.show()
# min_value