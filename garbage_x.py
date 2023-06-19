from model_config import DnCNN
import torch
import numpy as np 
import copy
from torch.nn import MSELoss
from model_config import DnCNN
from utilities.utils import set_seed
from utilities import FineTune

device = torch.device('cuda')
set_seed(100)
compute_mse_loss = MSELoss()

model1 =DnCNN(1,8)
model2 = DnCNN(1,8)
model_dir = 'trained_models/noise2true_chest_abdomen.pt'
model1.load_state_dict(torch.load(model_dir,map_location='cpu'))
model1.train()
tmp = filter(lambda x: x.requires_grad, model1.parameters())
num = sum(map(lambda x: np.prod(x.shape),tmp))
print("Total trainable tensors", num)