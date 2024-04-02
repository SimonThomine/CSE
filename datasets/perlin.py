import torch
import math
import numpy as np
import cv2
import imgaug.augmenters as iaa
import random
import torchvision.transforms as T
import glob

def lerp_np(x,y,w):
    fin_out = (y-x)*w + x
    return fin_out

def generate_fractal_noise_2d(shape, res, octaves=1, persistence=0.5):
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(shape, (frequency*res[0], frequency*res[1]))
        frequency *= 2
        amplitude *= persistence
    return noise


def generate_perlin_noise_2d(shape, res):
    def f(t):
        return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)


def rand_perlin_2d_np(shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1

    angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    tt = np.repeat(np.repeat(gradients,d[0],axis=0),d[1],axis=1)

    tile_grads = lambda slice1, slice2: np.repeat(np.repeat(gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]],d[0],axis=0),d[1],axis=1)
    dot = lambda grad, shift: (
                np.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]),
                            axis=-1) * grad[:shape[0], :shape[1]]).sum(axis=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * lerp_np(lerp_np(n00, n10, t[..., 0]), lerp_np(n01, n11, t[..., 0]), t[..., 1])


def rand_perlin_2d(shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    grid = torch.stack(torch.meshgrid(torch.arange(0, res[0], delta[0]), torch.arange(0, res[1], delta[1])), dim=-1) % 1
    angles = 2 * math.pi * torch.rand(res[0] + 1, res[1] + 1)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    tile_grads = lambda slice1, slice2: gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]].repeat_interleave(d[0],
                                                                                                              0).repeat_interleave(
        d[1], 1)
    dot = lambda grad, shift: (
                torch.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]),
                            dim=-1) * grad[:shape[0], :shape[1]]).sum(dim=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])

    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1])


def rand_perlin_2d_octaves(shape, res, octaves=1, persistence=0.5):
    noise = torch.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * rand_perlin_2d(shape, (frequency * res[0], frequency * res[1]))
        frequency *= 2
        amplitude *= persistence
    return noise


class TexturalAnomalyGenerator():
    def __init__(self, resize_shape=None,dtd_path="/home/aquilae/Thèse/datasets/dtd/images"):
        
        
        self.resize_shape=resize_shape
        self.anomaly_source_paths = sorted(glob.glob(dtd_path+"/*/*.jpg"))
        self.augmenters = [iaa.GammaContrast((0.5,2.0),per_channel=True),
                      iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
                      iaa.pillike.EnhanceSharpness(),
                      iaa.AddToHueAndSaturation((-10,10),per_channel=True),
                      iaa.Solarize(0.5, threshold=(32,128)),
                      iaa.Posterize(),
                      iaa.Invert(),
                      iaa.pillike.Autocontrast(),
                      iaa.pillike.Equalize(),
                      ]
    
    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug
    def getDtdImage(self):
        randIndex=random.randint(0, len(self.anomaly_source_paths)-1)
        
        image=cv2.imread(self.anomaly_source_paths[randIndex])
        image=cv2.resize(image, dsize=(self.resize_shape[0], self.resize_shape[1]))
        aug=self.randAugmenter()
        image=aug(image=image)
        return image 
    
class StructuralAnomalyGenerator():
    def __init__(self,resize_shape=None):
        
        self.resize_shape=resize_shape
        self.augmenters = [iaa.Fliplr(0.5),  
                            iaa.Affine(rotate=(-45, 45)),  
                            iaa.Multiply((0.8, 1.2)),  
                            iaa.MultiplySaturation((0.5, 1.5)), 
                            iaa.MultiplyHue((0.5, 1.5))
                      ]
    
    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug
    
    def generateStructuralDefect(self,image):
        aug=self.randAugmenter()
        image_array=(image.permute(1,2,0).numpy()*255).astype(np.uint8)# # *


        image_array=aug(image=image_array)

        height, width, _ = image_array.shape
        grid_size = 8
        cell_height = height // grid_size
        cell_width = width // grid_size
        
        grid = []
        for i in range(grid_size):
            for j in range(grid_size):
                cell = image_array[i * cell_height: (i + 1) * cell_height,
                                j * cell_width: (j + 1) * cell_width, :]
                grid.append(cell)
        
        np.random.shuffle(grid)
        
        reconstructed_image = np.zeros_like(image_array)
        for i in range(grid_size):
            for j in range(grid_size):
                reconstructed_image[i * cell_height: (i + 1) * cell_height,
                                    j * cell_width: (j + 1) * cell_width, :] = grid[i * grid_size + j]
        return reconstructed_image

class DefectGenerator():

    def __init__(self, resize_shape=None,dtd_path="/home/aquilae/Thèse/datasets/dtd/images"):


        self.texturalAnomalyGenerator=TexturalAnomalyGenerator(resize_shape,dtd_path)
        self.structuralAnomalyGenerator=StructuralAnomalyGenerator(resize_shape)
        
        self.resize_shape=resize_shape
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
        
    def generateMask(self,bMask):
        perlin_scale = 6
        min_perlin_scale = 0
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

        perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image=perlin_noise)
        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_thr = np.expand_dims(perlin_thr, axis=2)
        msk = (perlin_thr).astype(np.float32) #*255.0 # à voir le 255.0
        msk=torch.from_numpy(msk).permute(2,0,1)
        if (len(bMask)>0):
            msk=bMask*msk
        return msk
    
    
    def generateTexturalDefect(self, image,bMask=[]):
        msk=torch.zeros((self.resize_shape[0], self.resize_shape[1]))
        while (torch.count_nonzero(msk)<100):
            msk=self.generateMask(bMask)
        texturalImg=self.texturalAnomalyGenerator.getDtdImage()
        texturalImg=torch.from_numpy(texturalImg).permute(2,0,1)
        mskDtd=texturalImg*(msk)
        image = image * (1 - msk)+  (mskDtd/255)
        return image ,msk
    
    def generateStructuralDefect(self, image,bMask=[]):
        msk=torch.zeros((self.resize_shape[0], self.resize_shape[1]))
        while (torch.count_nonzero(msk)<100):
            msk=self.generateMask(bMask)
        structuralImg=self.structuralAnomalyGenerator.generateStructuralDefect(image)
        structuralImg=torch.from_numpy(structuralImg).permute(2,0,1)
        mskDtd=structuralImg*(msk)
        image = image * (1 - msk)+  (mskDtd/255)
        
        return image ,msk
    
    
    def generateBlurredDefectiveImage(self, image,bMask=[]):
        msk=torch.zeros((self.resize_shape[0], self.resize_shape[1]))
        while (torch.count_nonzero(msk)<100):
            msk=self.generateMask(bMask)
        randGaussianValue = random.randint(0, 5)*2+5
        transform = T.GaussianBlur(kernel_size=(randGaussianValue, randGaussianValue), sigma=6.0)
        imageBlurred = transform(image)

        imageBlurred=imageBlurred*(msk)
        image=image*(1-msk)

        image=image+imageBlurred
        return image,msk
    
    def generateDefectRandomlyAndMskTesting(self, image):
        isDefect=random.randint(0, 1)
        if (isDefect==1):
            randDefectType = random.randint(0, 1) 
            if randDefectType==0:
                result,msk=self.generateBlurredDefectiveImage(image,[])
            else:
                result,msk=self.generateTexturalDefect(image,[])
            return result,True,msk
        else:
            return image,False,torch.ones((1,self.resize_shape[0], self.resize_shape[1]))
    
    def generateDefectRandomlyAndMsk(self, image):
        isDefect=random.randint(0, 1)
        if (isDefect==1):
            randDefectType = random.randint(0, 2) 
            if randDefectType==0:
                result,msk=self.generateStructuralDefect(image,[])

            elif randDefectType==1:
                result,msk=self.generateTexturalDefect(image,[])
            else:
                result,msk=self.generateBlurredDefectiveImage(image,[])
            return result,True,msk
        else:
            return image,False,torch.ones((1,self.resize_shape[0], self.resize_shape[1]))

    def generateDefectRandomlyAndMskObjects(self, image,bMask):
        isDefect=random.randint(0, 1)
        if (isDefect==1):
            randDefectType = random.randint(0, 2)  
            if randDefectType==0:
                result,msk=self.generateStructuralDefect(image,bMask)
            elif randDefectType==1:
                result,msk=self.generateTexturalDefect(image,bMask)
            else:
                result,msk=self.generateBlurredDefectiveImage(image,bMask)
            return result,True,msk
        else:
            return image,False,torch.ones((1,self.resize_shape[0], self.resize_shape[1]))
    

