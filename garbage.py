from numpy import result_type
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset as dset

# Load the loss variables from results folderr
result_paths = ["noise2true","noise2self","ours_ssrl_noise2self",
                "noise2same","ours_ssrl_noise2same","noise2inverse", "ours_ssrl_noise2inverse"]
name_paths = ["Noise2True","Noise2Self","SSRL in Noise2Self setup",
                "Noise2Same","SSRL in Noise2Same setup", "Noise2Inverse", "SSRL in Noise2Inverse setup"]

line_style =['b','-g','-.g','-r','-.r','-m','-.m']

train_loss = []
train_RMSE = []

for i in range(len(result_paths)):
    # load the loss data using the model name
    model_name = result_paths[i]
    # Load the loss data from the results folder
    filename = "Results/"+model_name+str("NEW.npz")
    data = np.load(filename)
    plt.figure(1)
    plt.plot(data['losses'],line_style[i],label=model_name)
    plt.title("Training Loss Curve")
    plt.ylim([0,6000])
    plt.legend(loc='upper right')
    plt.figure(2)
    plt.plot(data['train_rmses'],line_style[i],label=model_name)
    plt.ylim([10,50])
    plt.title("Training RMSE Curve")

plt.legend(loc='upper right')
plt.show()