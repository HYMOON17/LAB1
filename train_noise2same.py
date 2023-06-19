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

device = torch.device('cuda')


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


set_seed(100)


class mydataset(dset):
    def __init__(self, folderpath_img):
        super(mydataset, self).__init__()

        self.clean_images = loadmat(folderpath_img)["xtrue"].transpose(2, 0, 1).astype(np.float64)
        self.noisy_images = loadmat(folderpath_img)["xfbp"].transpose(2, 0, 1).astype(np.float64)

    def __len__(self):
        return len(self.clean_images)

    def __getitem__(self, index):
        clean_images = np.expand_dims(self.clean_images[index], axis=0)
        noisy_images = np.expand_dims(self.noisy_images[index], axis=0)

        return (clean_images, noisy_images)


num_of_layers = 8
batch_size = 2
lr = 1e-1
step_size = 10
gamma = 0.95

num_epoch = 1000

sigma = 500

from mask_stratified_sampling import Masker

masker = Masker(width=14, mode='interpolate')

from dncnn import DnCNN

model = DnCNN(1, num_of_layers=num_of_layers)
model = model.to(device)

from torch.nn import MSELoss
from torch.optim import Adam

loss_function_mean = MSELoss(reduction='mean')
loss_function_sum = MSELoss(reduction='sum')

optimizer = Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

train_dataset = mydataset('data/train_imgs.mat')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False)

losses = []
train_rmses = []
result_path = "Results/noise2sameNEW"

for epoch in range(num_epoch):
    model.train()
    total_loss = 0
    total_train_rmse = 0

    for idx, batch in enumerate(train_loader):
        clean_images, noisy_images = batch
        clean_image = np.squeeze(clean_images.cpu().numpy().astype(np.float64))
        noisy_images = noisy_images.to(device, dtype=torch.float)

        net_input, masks = masker.mask(noisy_images)

        net_output = model(net_input)
        denoised = model(noisy_images)

        loss_rec = loss_function_mean(denoised, noisy_images)
        loss_inv = loss_function_sum(denoised * masks, net_output * masks) / torch.sum(masks)
        loss = loss_rec + (2.0*sigma) * torch.sqrt(loss_inv)

        denoised = np.squeeze(denoised.detach().cpu().numpy().astype(np.float64))
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

    torch.save(model.state_dict(), result_path + ".pt")
    np.savez(result_path, losses=losses, train_rmses=train_rmses)

    print("(", (epoch + 1), ") Training Loss: %.1f" % total_loss, ", RMSE, : %.1f" % total_train_rmse)