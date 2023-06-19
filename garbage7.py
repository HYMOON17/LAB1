# Code for parameter difference based relative error
import torch
import numpy as np 
import copy
from torch.nn import MSELoss
from model_config import DnCNN
# from utilities.utils import set_seed
# from utilities import FineTune

device = torch.device('cuda')
# set_seed(100)
compute_mse_loss = MSELoss()

model1 =DnCNN(1,8)
# model2 = DnCNN(1,8)
# model_dir = 'trained_models/noise2true_chest_abdomen.pt'
# model1.load_state_dict(torch.load(model_dir,map_location='cpu'))
# model2.load_state_dict(torch.load(model_dir,map_location='cpu'))


def compute_param_diff(model1,model2):
    data1,data2 =[],[] 
    for param1 in model1.parameters():
        data1.append(param1.data)
    for param2 in model2.parameters():
        data2.append(param2.data)

    diff = 0
    for i in range(len(data1)):
        d = compute_mse_loss(data1[i],data2[i])
        diff+=d


    return(diff.cpu().detach().numpy())
# print(compute_param_diff(model1,model2))

from torch.optim import Adam, optimizer
optimizer = Adam(model1.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma =0.95)

param_error = []
for i in range(100):
    #-------------------------------------------------------
    data1,data2 =[],[] 
    copy_model = copy.deepcopy(model1)
    for param1 in model1.parameters():
        data1.append(param1.data)

    #---------------------------------------------------------
    # loss = torch.Tensor([400])
    # loss.requires_grad =True
    # grad = torch.autograd.grad(loss, model1.parameters(),create_graph=True, retain_graph=True, allow_unused=True)
    # fast_weights = list(map(lambda p: p[1] - 0.01 * p[0], zip(grad, model1.parameters())))
    meta_loss = torch.Tensor([0])
    meta_loss.to(device)

    # meta_loss.set_(torch.Tensor([500]).to(device))
    meta_loss.set_(torch.Tensor([10000]))
    meta_loss.requires_grad = True
    # model1.train()
    optimizer.zero_grad()
    meta_loss.backward()
    optimizer.step()
    scheduler.step()
    #--------------------------
    # model1.load_state_dict(fast_weights)
    for param1 in model1.parameters():
        data2.append(param1.data)
        
    diff = 0
    for k in range(len(data1)):
        d = compute_mse_loss(data1[k],data2[k])
        diff+=d

    # print(diff)
    param_error.append(diff.cpu().detach().numpy())
    #------------------------------------------------------------
print("Param Error",sum(param_error))