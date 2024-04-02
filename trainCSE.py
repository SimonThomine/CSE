import os
import time
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from datasets.mvtec import MVTecDataset,MVTecDefectDataset
from utils.util import  AverageMeter,readYamlConfig,extractElementsNearClusters
from utils.functions import (
    cal_loss,
    anomaly_localisation,
    cosine_loss_batch)
from utils.visualization import plt_fig
from models.extractor import teacherTimm
from models.fusionEmbedder import fusionEmbedder,fusionDecoder
from torchvision import transforms as T


class AnomalyDistillation:          
    def __init__(self, data,device): 
        self.device = device
        self.validation_ratio = 0.2
        self.data_path = data['data_path']
        self.obj = data['obj']
        self.img_resize = data['TrainingData']['img_size']
        self.img_cropsize = data['TrainingData']['crop_size']
        self.num_epochs = data['TrainingData']['epochs']
        self.lr = data['TrainingData']['lr']
        self.batch_size = data['TrainingData']['batch_size']
        self.vis = data['vis']     
        self.model_dir = data['save_path'] + "/models" + "/" + self.obj
        self.img_dir = data['save_path'] + "/imgs" + "/" + self.obj
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.img_dir, exist_ok=True)
        self.modelName = data['backbone']
        self.outIndices = data['out_indice']
        self.inputDim=data['embed_input_dim']
        self.embedDim=data['embed_dim']
            
        self.load_model()
        self.load_dataset()


        self.optimizer = torch.optim.Adam(
            self.embedder.parameters(), lr=self.lr, betas=(0.5, 0.999),weight_decay=1e-5
        ) 
        
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.lr*10,
            epochs=self.num_epochs,
            steps_per_epoch=len(self.train_loader),
        )

    def load_dataset(self):
        kwargs = (
            {"num_workers": 8, "pin_memory": True} if torch.cuda.is_available() else {}
        )
        train_dataset = MVTecDataset(
            self.data_path,
            class_name=self.obj,
            is_train=True,
            resize=self.img_resize,
            cropsize=self.img_cropsize,
        )
        img_nums = len(train_dataset)
        valid_num = int(img_nums * self.validation_ratio)
        train_num = img_nums - valid_num
        train_data, val_data = torch.utils.data.random_split(
            train_dataset, [train_num, valid_num]
        )
        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True, **kwargs)
        self.val_loader = torch.utils.data.DataLoader(val_data, batch_size=8, shuffle=False, **kwargs)

        train_defect_dataset = MVTecDefectDataset(
            self.data_path,
            class_name=self.obj,
            is_train=True,
            resize=self.img_resize,
            cropsize=self.img_cropsize,
        )
        train_dataDefect, val_dataDefect = torch.utils.data.random_split(
            train_defect_dataset, [train_num, valid_num]
        )
        self.train_loaderDefect = torch.utils.data.DataLoader(train_dataDefect, batch_size=self.batch_size, shuffle=True, **kwargs)
        self.val_loaderDefect = torch.utils.data.DataLoader(val_dataDefect, batch_size=8, shuffle=True, **kwargs)

    def load_model(self):
        print("loading and training " + self.modelName)
        
        self.extractor = teacherTimm(backbone_name=self.modelName,out_indices=self.outIndices).to(self.device)
        
        self.embedder=fusionEmbedder(self.inputDim[0]+self.inputDim[1],self.embedDim).to(self.device)
        self.decoder=fusionDecoder(self.inputDim[0],self.inputDim[1],self.embedDim).to(self.device)

        
        self.extractor.eval()
        for param in self.extractor.parameters():
            param.requires_grad = False

    def train(self):
        print("training " + self.obj)
        self.embedder.train()
        self.decoder.train()
                
        best_score = None
        start_time = time.time()
        epoch_time = AverageMeter()
        epoch_bar = tqdm(
            total=len(self.train_loader) * self.num_epochs, 
            desc="Training",
            unit="batch",
        )
        for _ in range(1, self.num_epochs + 1):
            losses = AverageMeter()

            for (data,_, _), (data2 ,isDefects ,masks) in zip(self.train_loader,self.train_loaderDefect):
                data = data.to(self.device)
                data2 = data2.to(self.device)
                
                self.optimizer.zero_grad()
                with torch.set_grad_enabled(True):          
                    output1Reduced,output2Reduced,output1,outputExtractor = self.infer(data,data2,False)    
                    
                    lossIntra=cosine_loss_batch(output1Reduced,output2Reduced,isDefects)
                        
                    outputDecoder=self.decoder(output1)  
                    lossExtra=cal_loss(outputDecoder, outputExtractor)
                       
                    loss=10*lossIntra+lossExtra
                    
                    losses.update(loss.sum().item(), data.size(0))
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                epoch_bar.set_postfix({"Loss": loss.item()})
                epoch_bar.update()

            val_loss = self.val(epoch_bar)
            if best_score is None:
                best_score = val_loss
                self.save_checkpoint()
            elif val_loss < best_score:
                best_score = val_loss
                self.save_checkpoint()
                print("model saved")

            epoch_time.update(time.time() - start_time)
            start_time = time.time()
        epoch_bar.close()
        print("Training end.")

    def val(self, epoch_bar):
        self.embedder.eval()
        
        losses = AverageMeter()
        for (data,_, _), (data2 ,isDefects ,masks) in zip(self.val_loader,self.val_loaderDefect): 
            data = data.to(self.device)
            data2 = data2.to(self.device)
            with torch.set_grad_enabled(False):
                
                output1Reduced,output2Reduced,output1,outputExtractor = self.infer(data,data2,False)    
                lossIntra=cosine_loss_batch(output1Reduced,output2Reduced,isDefects)
                
                outputDecoder=self.decoder(output1)
                lossExtra=cal_loss(outputDecoder, outputExtractor)
                
                loss=10*lossIntra+lossExtra
                
                losses.update(loss.item(), data.size(0))
        epoch_bar.set_postfix({"Loss": loss.item()})

        return losses.avg

    def save_checkpoint(self):
        state = {"model": self.embedder.state_dict()}
        torch.save(state, os.path.join(self.model_dir, "embedder.pth"))
        
        state = {"model": self.decoder.state_dict()}
        torch.save(state, os.path.join(self.model_dir, "decoder.pth"))

    def test(self):
        try:
            checkpoint = torch.load(os.path.join(self.model_dir, "embedder.pth"))
        except:
            raise Exception("Check saved model path.")
        self.embedder.load_state_dict(checkpoint["model"])
        self.embedder.eval()

        kwargs = (
            {"num_workers": 4, "pin_memory": True} if torch.cuda.is_available() else {}
        )
        test_dataset = MVTecDataset(
            self.data_path,
            class_name=self.obj,
            is_train=False,
            resize=self.img_resize,
            cropsize=self.img_cropsize,
        )
        batch_size_test = 1
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size_test, shuffle=False, **kwargs
        )
        mem_bank_scores = []
        map_scores = []
        test_imgs = []
        gt_list = []
        gt_mask_list = []
        progressBar = tqdm(test_loader)
        correctFeatures=self.extractCorrectFeatures() 

        for data, label, mask in test_loader:
            test_imgs.extend(data.cpu().numpy())
            gt_list.extend(label.cpu().numpy())
            gt_mask_list.append(mask.squeeze().cpu().numpy())
            
            data = data.to(self.device)
            with torch.set_grad_enabled(False):
                timeBefore = time.perf_counter()
                
                outputTest=self.extractor(data)
                _,outputTest=self.embedder(outputTest)  
                
                mem_bank_score, map_score=anomaly_localisation(correctFeatures, outputTest, 1,out_size=self.img_cropsize) # was 2
                
                timeAfterFeatures = time.perf_counter()
                print("inference : " + str(timeAfterFeatures - timeBefore))
                
                progressBar.update()
                
            if batch_size_test == 1:
                mem_bank_scores.append(mem_bank_score)
                map_scores.append(map_score)
            else:
                mem_bank_scores.extend(mem_bank_score)
                map_scores.extend(map_score)
        progressBar.close()
        gt_list = np.asarray(gt_list)
        img_roc_aucCosine = roc_auc_score(gt_list, mem_bank_scores)
        print(self.obj + " image ROCAUC memBank: %.3f" % (img_roc_aucCosine))

        map_scores = np.asarray(map_scores)

        max_anomaly_score = map_scores.max()
        min_anomaly_score = map_scores.min()
        map_scores = (map_scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)

        img_scores = map_scores.reshape(map_scores.shape[0], -1).max(axis=1)
        gt_list = np.asarray(gt_list)
        img_roc_auc = roc_auc_score(gt_list, img_scores)
        
        print(self.obj + " image ROCAUC localisation: %.3f" % (img_roc_auc))
        if self.vis:
            map_scores=map_scores.squeeze()
            precision, recall, thresholds = precision_recall_curve(
                gt_list.flatten(), img_scores.flatten()
            )
            a = 2 * precision * recall
            b = precision + recall
            f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
            cls_threshold = thresholds[np.argmax(f1)]

            gt_mask = np.asarray(gt_mask_list)
            precision, recall, thresholds = precision_recall_curve(
                gt_mask.flatten(), map_scores.flatten()
            )
            a = 2 * precision * recall
            b = precision + recall
            f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
            seg_threshold = thresholds[np.argmax(f1)]
            per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), map_scores.flatten())
            print("pixel ROCAUC: %.3f" % (per_pixel_rocauc))
            plt_fig(
                test_imgs,
                map_scores,
                img_scores,
                gt_mask_list,
                seg_threshold,
                cls_threshold,
                self.img_dir,
                self.obj,
            )
            
        return img_roc_aucCosine
    
    def infer(self, data,data2,measureTime=False):
        if measureTime:
            timeBefore = time.perf_counter()
        output1temp = self.extractor(data)
        output2temp = self.extractor(data2)
        output1,output1Reduced=self.embedder(output1temp)
        _,output2Reduced=self.embedder(output2temp)
        if measureTime:
            torch.cuda.synchronize()
            timeAfterFeatures = time.perf_counter()
            print("inference : " + str(timeAfterFeatures - timeBefore))
        return output1Reduced, output2Reduced, output1, output1temp

    def extractCorrectFeatures(self):
        self.extractor.eval()
        self.embedder.eval()
        kwargs = ({"num_workers": 4, "pin_memory": True} if torch.cuda.is_available() else {})
        extract_dataset = MVTecDataset(self.data_path,class_name=self.obj,is_train=True,resize=self.img_resize,cropsize=self.img_cropsize)
        extract_loader = torch.utils.data.DataLoader(extract_dataset, batch_size=1, shuffle=False, **kwargs)
        correctFeats=[]
        for data, _, _ in extract_loader:
            data = data.to(self.device)
            with torch.set_grad_enabled(False):
                correctFeat=self.extractor(data)
                _,correctFeat=self.embedder(correctFeat)
                
            correctFeats.append(correctFeat)
            
        clustersFeatures=extractElementsNearClusters(correctFeats,k=1)
    
        return clustersFeatures
    
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data=readYamlConfig("config.yaml")
    distill = AnomalyDistillation(data,device)
     
    if data['phase'] == "train":
        distill.train()
        distill.test()
    elif data['phase'] == "test":
        distill.test()
    else:
        print("Phase argument must be train or test.")

