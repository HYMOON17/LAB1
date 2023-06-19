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
from model_config.dncnn import DnCNN
from torch.nn import MSELoss
from torch.optim import Adam

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

        self.I_true_all = loadmat(folderpath_img)["xtrue"].transpose(2, 0, 1).astype(np.float64)
        self.I_noisy_all = loadmat(folderpath_img)["xfbp"].transpose(2, 0, 1).astype(np.float64)
        self.I_noisy_odd_all = loadmat(folderpath_img)["xfbp_odd"].transpose(2, 0, 1).astype(np.float64)
        self.I_noisy_even_all = loadmat(folderpath_img)["xfbp_even"].transpose(2, 0, 1).astype(np.float64)

    def __len__(self):
        return len(self.I_true_all)

    def __getitem__(self, index):
        I_true = np.expand_dims(self.I_true_all[index], axis=0)
        I_noisy = np.expand_dims(self.I_noisy_all[index], axis=0)
        I_noisy_odd_all = np.expand_dims(self.I_noisy_odd_all[index], axis=0)
        I_noisy_even_all = np.expand_dims(self.I_noisy_even_all[index], axis=0)

        return (I_true, I_noisy, I_noisy_odd_all, I_noisy_even_all)

if __name__=='__main__':

    num_of_layers = 8
    batch_size = 2
    lr = 1e-3
    step_size = 10
    gamma = 0.95

    num_epoch = 1000


    model = DnCNN(1, num_of_layers=num_of_layers)
    model = model.to(device)


    loss_function = MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    train_dataset = mydataset('data/train_imgs_odd_even.mat')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False)

    losses = []
    train_rmses = []
    result_path = "Results/noise2inverseNEW"

    for epoch in range(num_epoch):
        model.train()
        total_loss = 0
        total_train_rmse = 0

        for idx, batch in enumerate(train_loader):
            clean_images, noisy_images, noisy_images_odd, noisy_images_even = batch

            noisy_images = noisy_images.to(device, dtype=torch.float)
            noisy_images_odd = noisy_images_odd.to(device, dtype=torch.float)
            noisy_images_even = noisy_images_even.to(device, dtype=torch.float)

            net_output = model(noisy_images_even)
            loss = loss_function(net_output, noisy_images_odd)

            denoised = model(noisy_images)
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

        torch.save(model.state_dict(), result_path + ".pt")
        np.savez(result_path, losses=losses, train_rmses=train_rmses)

        print("(", (epoch + 1), ") Training Loss: %.1f" % total_loss, ", RMSE, : %.1f" % total_train_rmse)

