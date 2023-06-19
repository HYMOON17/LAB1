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
from model_config import DnCNN
device = torch.device('cuda')

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


test_dataset = mydataset('data/test_imgs.mat')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)




# result_paths = ["trained_models/noise2true.pt","trained_models/noise2self.pt","trained_models/ours_ssrl_noise2self.pt","trained_models/noise2same.pt","trained_models/ours_ssrl_noise2same.pt","trained_models/noise2inverse.pt"]

result_paths = ["trained_models/noise2trueNEW.pt","trained_models/noise2selfNEW.pt","trained_models/ours_ssrl_noise2selfNEW.pt","trained_models/noise2sameNEW.pt","trained_models/ours_ssrl_noise2sameNEW.pt","trained_models/noise2inverseNEW.pt"]
name_paths = ["Noise2True","Noise2Self","SSRL in Noise2Self setup","Noise2Same","SSRL in Noise2Same setup", "Noise2Inverse"]

# result_paths = ["trained_models/noise2self_checkerboardNEW.pt","trained_models/ours_ssrl_noise2selfNEW.pt"]
# name_paths = ["Noise2Self","SSRL in Noise2Self setup"]

if __name__=='__main__':
    for ind,result_path in enumerate(result_paths):
        model = DnCNN(1, num_of_layers=8)
        model.load_state_dict(torch.load(result_path, map_location='cpu'))
        model = model.to(device)
        model.eval()

        total_rmse = 0

        for val_idx, val_batch in enumerate(test_loader):
            clean_images, noisy_images = val_batch
            clean_image = np.squeeze(clean_images.cpu().numpy().astype(np.float64))
            noisy_image = np.squeeze(noisy_images.cpu().numpy().astype(np.float64))
            noisy_images = noisy_images.to(device, dtype=torch.float)

            with torch.no_grad():
                denoised = model(noisy_images)

            denoised = np.squeeze(denoised.detach().cpu().numpy().astype(np.float64))

            rmse = np.sqrt(mean_squared_error(denoised, clean_image))
            total_rmse += rmse

        total_rmse = total_rmse / (val_idx+1)

        print("%s: %.1f" % (name_paths[ind], total_rmse))


    #### Proposed SSRL in the Noise2Inverse setup

    model = DnCNN(1, num_of_layers=8)
    model_g = DnCNN(1, num_of_layers=8)

    # model.load_state_dict(torch.load("trained_models/ours_ssrl_noise2inverse.pt", map_location='cpu'))
    model.load_state_dict(torch.load("trained_models/ours_ssrl_noise2inverseNEW.pt", map_location='cpu'))

    model = model.to(device)

    # model_g.load_state_dict(torch.load("trained_models/noise2inverse.pt", map_location='cpu'))
    model_g.load_state_dict(torch.load("trained_models/noise2inverseNEW.pt", map_location='cpu'))
    model_g = model_g.to(device)

    denoised_images = []
    model.eval()
    model_g.eval()

    total_rmse = 0

    for val_idx, val_batch in enumerate(test_loader):
        clean_images, noisy_images = val_batch
        clean_image = np.squeeze(clean_images.cpu().numpy().astype(np.float64))
        noisy_image = np.squeeze(noisy_images.cpu().numpy().astype(np.float64))
        noisy_images = noisy_images.to(device, dtype=torch.float)

        with torch.no_grad():
            denoised = (model(noisy_images) + model_g(noisy_images)) / 2.0

            denoised = np.squeeze(denoised.detach().cpu().numpy().astype(np.float64))

            rmse = np.sqrt(mean_squared_error(denoised, clean_image))
            total_rmse += rmse

    total_rmse = total_rmse / (val_idx+1)

    print("SSRL in Noise2Inverse setup: %.1f" % total_rmse)

    # Test Line