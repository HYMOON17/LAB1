import numpy as np
import torch
import matplotlib.pyplot as plt


class Masker():
    """Object for masking and demasking"""

    def __init__(self, width=3, mode='zero', infer_single_pass=False, include_mask_as_input=False):
        self.grid_size = width
        self.n_masks = width ** 2

        self.mode = mode
        self.infer_single_pass = infer_single_pass
        self.include_mask_as_input = include_mask_as_input

    def mask(self, X):

        mask = get_stratified_coords2D(rand_float_coords2D(self.grid_size), box_size=self.grid_size, shape=X.shape)
        mask = mask.to(X.device)

        mask_inv = torch.ones(mask.shape).to(X.device) - mask

        if self.mode == 'interpolate':
            masked = interpolate_mask(X, mask, mask_inv)
        elif self.mode == 'zero':
            masked = X * mask_inv
        else:
            raise NotImplementedError
            
        if self.include_mask_as_input:
            net_input = torch.cat((masked, mask.repeat(X.shape[0], 1, 1, 1)), dim=1)
        else:
            net_input = masked

        return net_input, mask

    def __len__(self):
        return self.n_masks

    def infer_full_image(self, X, model):

        if self.infer_single_pass:
            if self.include_mask_as_input:
                net_input = torch.cat((X, torch.zeros(X[:, 0:1].shape).to(X.device)), dim=1)
            else:
                net_input = X
            net_output = model(net_input)
            return net_output

        else:
            net_input, mask = self.mask(X, 0)
            net_output = model(net_input)

            acc_tensor = torch.zeros(net_output.shape).cpu()

            for i in range(self.n_masks):
                net_input, mask = self.mask(X, i)
                net_output = model(net_input)
                # acc_tensor = acc_tensor + (net_output * mask).cpu()
                acc_tensor = acc_tensor + (net_output * mask).detach().cpu()

            return acc_tensor



def get_stratified_coords2D(coord_gen, box_size, shape):
    mask = torch.zeros(shape)
    box_count_y = int(np.ceil(shape[-1] / box_size))
    box_count_x = int(np.ceil(shape[-2] / box_size))

    for idx in range(shape[0]):
        for i in range(box_count_y):
            for j in range(box_count_x):
                y, x = next(coord_gen)
                y = int(i * box_size + y)
                x = int(j * box_size + x)
                if (y < shape[-1] and x < shape[-2]):
                    mask[idx, :, y, x] = 1.0

    return mask


def rand_float_coords2D(boxsize):
    while True:
        yield np.random.rand() * boxsize, np.random.rand() * boxsize


def interpolate_mask(tensor, mask, mask_inv):
    device = tensor.device

    mask = mask.to(device)

    kernel = np.array([[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], (0.5, 1.0, 0.5)])
    kernel = kernel[np.newaxis, np.newaxis, :, :]
    kernel = torch.Tensor(kernel).to(device)
    kernel = kernel / kernel.sum()

    filtered_tensor = torch.nn.functional.conv2d(tensor, kernel, stride=1, padding=1)

    return filtered_tensor * mask + tensor * mask_inv
