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

sigma = 15

from dncnn import DnCNN

model = DnCNN(1, num_of_layers=num_of_layers)

model_gx = DnCNN(1, num_of_layers=num_of_layers)
model_gx.load_state_dict(torch.load("./trained_models/noise2same_checkerboardNEW.pt", map_location='cpu'))

model = model.to(device)
model_gx = model_gx.to(device)
model_gx.eval()

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
rec_losses = []
inv_losses = []
result_path = "Results/ours_ssrl_noise2sameNEW"

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
    model.train()
    total_loss = 0
    total_train_rmse = 0
    total_rec_loss = 0
    total_inv_loss = 0

    for idx, batch in enumerate(train_loader):
        clean_images, noisy_images = batch
        clean_image = np.squeeze(clean_images.cpu().numpy().astype(np.float64))
        noisy_images = noisy_images.to(device, dtype=torch.float)

        # reflection padding
        noisy_images_padded = m(noisy_images)

        # interpolation
        filtered_tensor = torch.nn.functional.conv2d(noisy_images_padded, kernel, stride=1, padding=0)
        net_input1 = filtered_tensor * S_mask + noisy_images * S_mask_inv
        net_input2 = filtered_tensor * S_mask_inv + noisy_images * S_mask

        net_output1 = model(net_input1)
        net_output2 = model(net_input2)
        denoised = model(noisy_images)

        with torch.no_grad():
            net_output_gx1 = model_gx(net_input1)
            net_output_gx2 = model_gx(net_input2)

        loss_rec1 = loss_function_sum(denoised * S_mask_inv, net_output_gx2 * S_mask_inv) / (
                    batch_size * torch.sum(S_mask_inv))
        loss_inv1 = loss_function_sum(denoised * S_mask_inv, net_output1 * S_mask_inv) / (
                    batch_size * torch.sum(S_mask_inv))
        loss1 = loss_rec1 + (2.0*sigma) * torch.sqrt(loss_inv1)

        loss_rec2 = loss_function_sum(denoised * S_mask, net_output_gx1 * S_mask) / (batch_size * torch.sum(S_mask))
        loss_inv2 = loss_function_sum(denoised * S_mask, net_output2 * S_mask) / (batch_size * torch.sum(S_mask))
        loss2 = loss_rec2 + (2.0*sigma) * torch.sqrt(loss_inv2)

        loss = (loss1 + loss2) / 2.0

        denoised = np.squeeze(denoised.detach().cpu().numpy().astype(np.float64))

        train_rmse = np.sqrt(mean_squared_error(denoised, clean_image))
        total_train_rmse += train_rmse

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_rec_loss += ((loss_rec1 + loss_rec2) / 2.0).item()
        total_inv_loss += ((torch.sqrt(loss_inv1) + torch.sqrt(loss_inv2)) / 2.0).item()

    scheduler.step()

    total_loss = total_loss / (idx + 1)
    total_train_rmse = total_train_rmse / (idx + 1)
    total_rec_loss = total_rec_loss / (idx + 1)
    total_inv_loss = total_inv_loss / (idx + 1)

    losses.append(total_loss)
    rec_losses.append(total_rec_loss)
    inv_losses.append(total_inv_loss)
    train_rmses.append(total_train_rmse)

    torch.save(model.state_dict(), result_path + ".pt")
    np.savez(result_path, losses=losses, rec_losses=rec_losses, inv_losses=inv_losses, train_rmses=train_rmses)

    print("(", (epoch + 1), ") Training Loss: %.1f" % total_loss, ", Loss of 1st term: %.1f" % total_rec_loss,
          ", Loss of 2nd term: %.1f" % total_inv_loss, ", RMSE: %.1f" % total_train_rmse)




