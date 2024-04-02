import time
import random

import yaml
from yaml.loader import BaseLoader
import numpy as np
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def time_string():
    ISOTIMEFORMAT = '%Y-%m-%d %X'
    string = '[{}]'.format(time.strftime(ISOTIMEFORMAT, time.localtime()))
    return string


def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600 * need_hour) / 60)
    need_secs = int(epoch_time - 3600 * need_hour - 60 * need_mins)
    return need_hour, need_mins, need_secs


def itr_merge(*itrs):
    for itr in itrs:
        for v in itr:
            yield v
            
            
def readYamlConfig(configFileName):
    with open(configFileName) as f:
        data=yaml.safe_load(f)
        return data
    
import torch
from sklearn.cluster import KMeans


def extractClustersKmeans(correctFeats,k=4): # k was 4
    correctFeats=torch.cat(correctFeats)
    correctFeats=correctFeats.cpu().numpy()
    correctFeats_flat = correctFeats.reshape(correctFeats.shape[0], -1)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(correctFeats_flat)
    clustersFeatures=[]
    for cluster in kmeans.cluster_centers_:
        clusterFeatAct=torch.tensor(cluster.reshape(correctFeats.shape[1],correctFeats.shape[2],correctFeats.shape[3])).unsqueeze(0).cuda()
        clustersFeatures.append(clusterFeatAct)
    return clustersFeatures

def extractElementsNearClusters(correctFeats,k=4): # k was 4
    correctFeats=torch.cat(correctFeats)
    correctFeats=correctFeats.cpu().numpy()
    correctFeats_flat = correctFeats.reshape(correctFeats.shape[0], -1)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(correctFeats_flat)
    representativeFeats=[]
    for cluster in kmeans.cluster_centers_:
        clusterFeatAct=cluster.reshape(correctFeats.shape[1],correctFeats.shape[2],correctFeats.shape[3])
        distMin=100000
        for correctFeat in correctFeats:

            dist=np.linalg.norm(correctFeat-clusterFeatAct)
            if dist<distMin:
                distMin=dist
                reprFeat=torch.tensor(correctFeat).unsqueeze(0).cuda()
        representativeFeats.append(reprFeat)
    return representativeFeats

def extractMeanOfCorrects(correctFeats,kMeans=True): 
    if (kMeans):
        correctFeats=torch.cat(correctFeats)
        correctFeats=correctFeats.cpu().numpy()
        correctFeats_flat = correctFeats.reshape(correctFeats.shape[0], -1)
        kmeans = KMeans(n_clusters=1, random_state=0).fit(correctFeats_flat)
        meanFeature=torch.tensor(kmeans.cluster_centers_[0].reshape(correctFeats.shape[1],correctFeats.shape[2],correctFeats.shape[3])).unsqueeze(0).cuda()

    else:
        correctFeats=torch.cat(correctFeats)
        meanFeature=torch.mean(correctFeats, dim=(0), keepdim=True)
    
    return [meanFeature]


def extractActiveFeaturesCluster(cluster):
    arrayIndex=np.zeros(cluster.shape[1])
    for i,feature in enumerate(cluster.squeeze(0)):
        if (torch.sum(feature)<0):     
            arrayIndex[i]=1     
    return arrayIndex

def purgeFeatures(features,arrayIndex):
    for i,feature in enumerate(features[0]):
        if (not arrayIndex[i]):
            features[0][i]=torch.zeros_like(feature)
    return features