import torch
from models.resnet_18 import ResNet18, BasicBlock

a = torch.rand((1, 3, 224, 224))
model = ResNet18(img_channels=3, num_layers= 18, block= BasicBlock, num_classes=10)
model(a)