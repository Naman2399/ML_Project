# import torch
#
# from models.inception import Inception
# model = Inception()
# t = torch.load('/raid/home/namanmalpani/final_yr/ML_Project/ckpts/debug_wild_cats_inception_lr_0.0005_bs_256_epochs_100/epoch_96.pt')
# print(type(t))
# keys = list(t.keys())
# for key in keys :
#     print(key)
#
# model_weights = t['model_state_dict']
# model.load_state_dict(model_weights)
# weights = model.get_model_weights()
# print(type(weights))
# print(weights.shape)


import torch
import torch.nn as nn
from torchsummary import summary
from utils.image_utils import resize_and_pad

from torchvision.models.resnet import resnet18
model = resnet18(weights = 'IMAGENET1K_V1')
print(model)

for param in model.parameters() :
    param.requires_grad = False


summary(model, (3, 244, 244), device='cpu')
print(model)

model.fc = nn.Linear(in_features=2048, out_features= 10, bias= True)
print(model)
print("-" * 50)
for parm in model.parameters() :
    print(parm.requires_grad)
