import torch.nn as nn
import torchvision.models as models


model = models.vgg16(pretrained=True) #130million+ parameters

#Freeze all model parameters
for param in model.parameters():
    param.requires_grad = False


#Add on classifier
# Add on classifier
n_classes=100
n_inputs=4096
model.classifier[6] = nn.Sequential(
                      nn.Linear(n_inputs, n_classes),                 
                      nn.LogSoftmax(dim=1))



