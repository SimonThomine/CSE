import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import  conv3x3, conv1x1, BasicBlock, Bottleneck
from torch.hub import load_state_dict_from_url


class teacherTimm(nn.Module):
    def __init__(
        self,
        backbone_name="resnet18",
        out_indices=[1, 2, 3]
    ):
        super(teacherTimm, self).__init__()     
        self.feature_extractor = timm.create_model(
            backbone_name,
            pretrained=True,
            features_only=True,
            out_indices=out_indices 
        )
        self.feature_extractor.eval() 
        for param in self.feature_extractor.parameters():
            param.requires_grad = False   
        
    def forward(self, x):
        features_t = self.feature_extractor(x)
        return features_t
    