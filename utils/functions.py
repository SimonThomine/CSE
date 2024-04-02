import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter
from torch.nn.functional import cosine_similarity
from skimage.measure import label

def cal_loss(fs_list, ft_list):
    t_loss = 0
    N = len(fs_list)
    for i in range(N):
        fs = fs_list[i]
        ft = ft_list[i]
        _, _, h, w = fs.shape
        fs_norm = F.normalize(fs, p=2)
        ft_norm = F.normalize(ft, p=2)
        f_loss = 0.5 * (ft_norm - fs_norm) ** 2
        f_loss = f_loss.sum() / (h * w)
        t_loss += f_loss

    return t_loss / N


def cosine_loss(data1, data2,isDefect):
    data_s_norm = F.normalize(data1, p=2)
    data_t_norm = F.normalize(data2, p=2)
    data_s_norm = data_s_norm.view(data_s_norm.size(0), data_s_norm.size(1), -1)
    data_t_norm = data_t_norm.view(data_t_norm.size(0), data_t_norm.size(1), -1)
    if (not isDefect):
        loss = 1 - cosine_similarity(data_s_norm, data_t_norm,dim=2)
    else:
        loss = cosine_similarity(data_s_norm, data_t_norm,dim=2)
    return loss.mean()


def cosine_loss_batch(data1, data2,isDefects):
    loss=0
    for feature1,feature2,isDefect in zip(data1,data2,isDefects):
        
        feature1 = feature1.view(feature1.size(0), feature1.size(1), -1)
        feature2 = feature2.view(feature2.size(0), feature2.size(1), -1)
        if (not isDefect):
            loss += 1 - cosine_similarity(feature1, feature2,dim=2)
        else:
            loss += (1 + cosine_similarity(feature1, feature2,dim=2))/2
    return loss.mean()


def anomaly_localisation(X_train, X_test, k,out_size):
    scores = []
    scoresMaps=[]
    
    for test_point in X_test:
        distances=[]
        anoMaps=[]
        test_pointCosine=test_point
       
        test_pointCosine = test_pointCosine.view(test_pointCosine.size(0), -1)
        for train_point in X_train:
            train_pointCosine=train_point

            
            train_pointCosine = train_pointCosine.view(train_pointCosine.size(0), train_pointCosine.size(1), -1) #! GOOD
            
            distances.append(torch.mean(1 - cosine_similarity(test_pointCosine, train_pointCosine),dim=1)) #! GOOD
            
            h,w=test_point.shape[1],test_point.shape[2]
            a_map = (0.5 * (train_point - test_point) ** 2) / (h * w)
            a_map = a_map.sum(1, keepdim=True)
            anoMaps.append(a_map)
            
        k_nearest_cos_indices = torch.topk(torch.tensor(distances), k, largest=True).indices
        
        k_nearest_distances = torch.tensor([distances[i] for i in k_nearest_cos_indices])
        
        k_nearestMaps = torch.cat([anoMaps[i] for i in k_nearest_cos_indices],dim=0)
          
        score=torch.mean(k_nearest_distances)
        anoMap=torch.mean(k_nearestMaps,dim=0)
        
        scores.append(score)
        scoresMaps.append(anoMap)

    scoresMapsNp=[]
    for anoMap in scoresMaps:
        for i in range(anoMap.shape[0]):
            anoMap = F.interpolate(
                a_map, size=out_size, mode="bilinear", align_corners=False
                )   
            anoMap = anoMap.cpu().numpy().squeeze() #
            anoMap = gaussian_filter(anoMap, sigma=4)
            scoresMapsNp.append(anoMap)
            
    return torch.tensor(scores), scoresMapsNp   
