import torch
from torch import nn
from torch.nn import functional as F

class fusionEmbedder(nn.Module):
    def __init__(self,in_chan=520,embeddingDim=64):
        super(fusionEmbedder, self).__init__()
        
        
        self.embedder=nn.Sequential(
            nn.Conv2d(in_chan, in_chan//2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_chan//2),
            nn.Conv2d(in_chan//2, in_chan//4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_chan//4),
            nn.Conv2d(in_chan//4, embeddingDim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(embeddingDim)
            )
        
        self.upsample=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.pooling=nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)

                                     
    def forward(self, x):
        
        x1=x[0]
        x2=self.upsample(x[1])
        x=torch.cat((x1,x2),dim=1)
        x=self.embedder(x)
        xReduced=self.pooling(x)
        
        return x,xReduced
    
    
class fusionDecoder(nn.Module):
    def __init__(self,out_chan1=136,out_chan2=384,embeddingDim=64): 
        super(fusionDecoder, self).__init__()

        self.decoder1=nn.Sequential(
            nn.Conv2d(embeddingDim, out_chan1//2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_chan1//2),
            nn.Conv2d(out_chan1//2, out_chan1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_chan1)
            )
                
        self.decoder2=nn.Sequential(
            nn.Conv2d(embeddingDim, out_chan2//2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_chan2//2),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(out_chan2//2, out_chan2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_chan2)
            )
        
        
    def forward(self, x):
        
        x1=self.decoder1(x)
        x2=self.decoder2(x) 
          
        return x1,x2
    