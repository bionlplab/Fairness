import pickle
import os
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import json
import numpy as np
import time
import copy
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler
import cv2
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from dataset_mimic import Dataset

image_path_train = '/prj0129/mil4012/glaucoma/NIH-chest-x-ray/CXR8/train_cxr11.csv'
image_path_val = '/prj0129/mil4012/glaucoma/NIH-chest-x-ray/CXR8/val_cxr11.csv'
image_path_test = '/prj0129/mil4012/glaucoma/NIH-chest-x-ray/CXR8/test_cxr11.csv'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Counter(object):
    """Computes and stores the average and current value"""
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

def compute_AUCs(gt, pred):
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    return roc_auc_score(gt_np[:,0], pred_np[:,0])

def cross_auc(R_a_0, R_b_1): 
    scores = np.array(list(R_a_0.cpu().numpy()) + list(R_b_1.cpu().numpy()))
    y_true = np.zeros(len(R_a_0)+len(R_b_1))
    y_true[0:len(R_a_0)] = 1 # Pr[ LEFT > RIGHT]; Y = 1 is the left (A0)
    return roc_auc_score(y_true, scores)


def group_auc(labels, outputs, groups):
    
    group0p = []
    group0n = []
    group1p = []
    group1n = []
    group2p = []
    group2n = []
    group3p = []
    group3n = []

    for i in range(len(labels)):
        if groups[i] == 0:
            if labels[i][0] == 1:
                group0p.append(i)
            if labels[i][0] == 0:
                group0n.append(i)
        if groups[i] == 1:
            if labels[i][0] == 1:
                group1p.append(i)
            if labels[i][0] == 0:
                group1n.append(i)
        if groups[i] == 2:
            if labels[i][0] == 1:
                group2p.append(i)
            if labels[i][0] == 0:
                group2n.append(i)       
        if groups[i] == 3:
            if labels[i][0] == 1:
                group3p.append(i)
            if labels[i][0] == 0:
                group3n.append(i)   
                
    groupp = group0p+group1p+group2p+group3p  
    groupn = group0n+group1n+group2n+group3n
    outputs_ = outputs.clone().detach().cpu()
    
    try:
        AUC = cross_auc(torch.index_select(outputs_,0,torch.tensor(groupp)), torch.index_select(outputs_,0,torch.tensor(groupn)))
    except:
        AUC = 1
    try:
        A00 = cross_auc(torch.index_select(outputs_,0,torch.tensor(group0p)), torch.index_select(outputs_,0,torch.tensor(group0n)))
    except:
        A00 = 1
    try:
        A11 = cross_auc(torch.index_select(outputs_,0,torch.tensor(group1p)), torch.index_select(outputs_,0,torch.tensor(group1n)))
    except:
        A11 = 1
    try:
        A22 = cross_auc(torch.index_select(outputs_,0,torch.tensor(group2p)), torch.index_select(outputs_,0,torch.tensor(group2n)))
    except:
        A22 = 1
    try:
        A33 = cross_auc(torch.index_select(outputs_,0,torch.tensor(group3p)), torch.index_select(outputs_,0,torch.tensor(group3n)))
    except:
        A33 = 1
    try:
        A0a = cross_auc(torch.index_select(outputs_,0,torch.tensor(group0p)), torch.index_select(outputs_,0,torch.tensor(groupn)))
    except:
        A0a = 1
    try:
        A1a = cross_auc(torch.index_select(outputs_,0,torch.tensor(group1p)), torch.index_select(outputs_,0,torch.tensor(groupn)))
    except:
        A1a = 1
    try:
        A2a = cross_auc(torch.index_select(outputs_,0,torch.tensor(group2p)), torch.index_select(outputs_,0,torch.tensor(groupn)))
    except:
        A2a = 1
    try:
        A3a = cross_auc(torch.index_select(outputs_,0,torch.tensor(group3p)), torch.index_select(outputs_,0,torch.tensor(groupn)))
    except:
        A3a = 1
    try:
        Aa0 = cross_auc(torch.index_select(outputs_,0,torch.tensor(groupp)), torch.index_select(outputs_,0,torch.tensor(group0n)))
    except:
        Aa0 = 1
    try:
        Aa1 = cross_auc(torch.index_select(outputs_,0,torch.tensor(groupp)), torch.index_select(outputs_,0,torch.tensor(group1n)))
    except:
        Aa1 = 1
    try:
        Aa2 = cross_auc(torch.index_select(outputs_,0,torch.tensor(groupp)), torch.index_select(outputs_,0,torch.tensor(group2n)))
    except:
        Aa2 = 1
    try:
        Aa3 = cross_auc(torch.index_select(outputs_,0,torch.tensor(groupp)), torch.index_select(outputs_,0,torch.tensor(group3n)))
    except:
        Aa3 = 1
                

    group_num = [len(group0p),len(group0n),len(group1p),len(group1n),len(group2p),len(group2n),len(group3p),len(group3n)]
    
    return AUC, A00, A11, A22, A33, A0a, A1a, A2a, A3a, Aa0, Aa1, Aa2, Aa3, group_num



def criterion(outputs, labels, groups):
    
    group0p = []
    group0n = []
    group1p = []
    group1n = []
    group2p = []
    group2n = []
    group3p = []
    group3n = []

    for i in range(len(labels)):
        if groups[i] == 0:
            if labels[i][0] == 1:
                group0p.append(i)
            if labels[i][0] == 0:
                group0n.append(i)
        if groups[i] == 1:
            if labels[i][0] == 1:
                group1p.append(i)
            if labels[i][0] == 0:
                group1n.append(i)
        if groups[i] == 2:
            if labels[i][0] == 1:
                group2p.append(i)
            if labels[i][0] == 0:
                group2n.append(i)       
        if groups[i] == 3:
            if labels[i][0] == 1:
                group3p.append(i)
            if labels[i][0] == 0:
                group3n.append(i)   
                
    groupp = group0p+group1p+group2p+group3p  
    groupn = group0n+group1n+group2n+group3n
    outputs_ = outputs.clone().detach().cpu()
    
    try:
        AUC = cross_auc(torch.index_select(outputs_,0,torch.tensor(groupp)), torch.index_select(outputs_,0,torch.tensor(groupn)))
    except:
        AUC = 1
    try:
        AUC0a = cross_auc(torch.index_select(outputs_,0,torch.tensor(group0p)), torch.index_select(outputs_,0,torch.tensor(groupn)))
    except:
        AUC0a = 1.1
    try:
        AUC1a = cross_auc(torch.index_select(outputs_,0,torch.tensor(group1p)), torch.index_select(outputs_,0,torch.tensor(groupn)))
    except:
        AUC1a = 1.1
    try:
        AUC2a = cross_auc(torch.index_select(outputs_,0,torch.tensor(group2p)), torch.index_select(outputs_,0,torch.tensor(groupn)))
    except:
        AUC2a = 1.1
    try:
        AUC3a = cross_auc(torch.index_select(outputs_,0,torch.tensor(group3p)), torch.index_select(outputs_,0,torch.tensor(groupn)))
    except:
        AUC3a = 1.1

#     if AUC ==1 and AUC0a == 1 and AUC1a == 1:
#         print('three auc are', AUC)
    loss = nn.MarginRankingLoss(margin=0.05)
    
    minimum = np.argsort(np.array([AUC0a, AUC1a, AUC2a, AUC3a, AUC]))[0]
    # print(minimum)
    
    if minimum == 0:
        index0p = []
        for i in group0p:
            index0p.extend([i]*len(groupn))
        index0an = (groupn)*len(group0p)
        # print (index0p, index0an)
        return loss(torch.index_select(outputs,0,torch.tensor(index0p).to(device)), torch.index_select(outputs,0,torch.tensor(index0an).to(device)), (torch.ones(len(index0p),1)*1).to(device))
    
    elif minimum == 1:
        index1p = []
        for i in group1p:
            index1p.extend([i]*len(groupn))
        index1an = (groupn)*len(group1p)
        
        return loss(torch.index_select(outputs,0,torch.tensor(index1p).to(device)), torch.index_select(outputs,0,torch.tensor(index1an).to(device)), (torch.ones(len(index1p),1)*1).to(device))


    elif minimum == 2:
        index2p = []
        for i in group2p:
            index2p.extend([i]*len(groupn))
        index2an = (groupn)*len(group2p)
        
        return loss(torch.index_select(outputs,0,torch.tensor(index2p).to(device)), torch.index_select(outputs,0,torch.tensor(index2an).to(device)), (torch.ones(len(index2p),1)*2).to(device))
    
    elif minimum == 3:
        index3p = []
        for i in group3p:
            index3p.extend([i]*len(groupn))
        index3an = (groupn)*len(group3p)
        
        return loss(torch.index_select(outputs,0,torch.tensor(index3p).to(device)), torch.index_select(outputs,0,torch.tensor(index3an).to(device)), (torch.ones(len(index3p),1)*1).to(device))    
    
    else:
        indexp = []
        for i in groupp:
            indexp.extend([i]*len(groupn))
        indexn = (groupn)*len(groupp)
        
        return loss(torch.index_select(outputs,0,torch.tensor(indexp).to(device)), torch.index_select(outputs,0,torch.tensor(indexn).to(device)), (torch.ones(len(indexp),1)*1).to(device))


# def train_model(dataloaders,model, criterion, optimizer, scheduler, num_epochs=25):
def train_model(dataloaders,model, criterion, optimizer, num_epochs=25):
    since = time.time()
    fopen = open("/prj0129/mil4012/AREDS/accuracy_pfm_gender.txt", "w")
    best_model_wts = copy.deepcopy(model.state_dict())
    best_AUROC_avg = 0.0
    losses = Counter()
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 100)
        
        
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            gt = torch.FloatTensor().to(device)
            pred = torch.FloatTensor().to(device)
            losses.reset()
            groups = []
            
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            # Iterate over data.
            t = tqdm(enumerate(dataloaders[phase]),  desc='Loss: **** ', total=len(dataloaders[phase]), bar_format='{desc}{bar}{r_bar}')
            for batch_idx, (inputs, labels, group) in t:
                # if batch_idx == 0:
                #     continue
                # print(torch.isnan(inputs).sum())
                inputs = inputs.to(device)
                labels = labels.to(device)
                #print(inputs.shape, labels.shape)
                # print('the lables is',torch.unique(labels))
                if len(torch.unique(labels)) !=1 and len(np.unique(group) != 1):
                    # print(len(torch.unique(labels)))
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)

                        gt = torch.cat((gt, labels), 0)
                        pred = torch.cat((pred, outputs.data), 0)
                        groups += group

                        # print('outputs shape',outputs.shape)
                        # print('labels shape', labels.shape)
                        # print('groups shape', group.shape)
                        loss = criterion(outputs, labels, group)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    losses.update(loss.data.item(), inputs.size(0))
                    t.set_description('Loss: %.3f ' % (losses.avg))
            
            AUCs = compute_AUCs(gt, pred)
            AUROC_avg = AUCs
            AUC, A00, A11, A22, A33, A0a, A1a, A2a, A3a, Aa0, Aa1, Aa2, Aa3, group_num = group_auc(gt, pred, groups)
            
            if phase == "val":
                
                # scheduler.step(losses.avg)
                
                if best_AUROC_avg < AUROC_avg:
                    best_AUROC_avg = AUROC_avg
                    torch.save(model.state_dict(), "/prj0129/mil4012/glaucoma/NIH-chest-x-ray/CXR8/weights/densenet201_mimic_gender_age_nofinding4.pth")
                fopen.write('\nEpoch {} \t [{}] : \t {AUROC_avg:.3f}\n'.format(epoch, phase, AUROC_avg=AUROC_avg))
                fopen.write('{} \t {}\n'.format(CLASS_NAMES, AUCs))
                fopen.write('-' * 100)
                    
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   epoch, batch_idx + 1, len(dataloaders[phase]), loss=losses))
            print('{} : \t {AUROC_avg:.3f}'.format(phase, AUROC_avg=AUROC_avg))
            print('AUC',AUC)
            print('A00',A00)
            print('A11',A11)
            print('A22',A22)
            print('A33',A33)
            print('A0a',A0a)
            print('A1a',A1a)
            print('A2a',A2a)
            print('A3a',A3a)
            print('Aa0',Aa0)
            print('Aa1',Aa1)
            print('Aa2',Aa2)
            print('Aa3',Aa3)
            print('Group Num',group_num)
            
            fopen.flush()
    fopen.close()
    return model


def test_model(test_loader,model):
    model.eval()
    gt = torch.FloatTensor().to(device)
    pred = torch.FloatTensor().to(device)
    groups = []
    with torch.no_grad():
        for batch_idx, (inputs, labels, group) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            gt = torch.cat((gt, labels), 0)
            pred = torch.cat((pred, outputs.data), 0)
            groups += group
    AUCs = compute_AUCs(gt, pred)
    AUC, A00, A11, A22, A33, A0a, A1a, A2a, A3a, Aa0, Aa1, Aa2, Aa3, group_num = group_auc(gt, pred, groups)
    print('AUCs',AUCs)
    print('AUC',AUC)
    print('A00',A00)
    print('A11',A11)
    print('A22',A22)
    print('A33',A33)
    print('A0a',A0a)
    print('A1a',A1a)
    print('A2a',A2a)
    print('A3a',A3a)
    print('Aa0',Aa0)
    print('Aa1',Aa1)
    print('Aa2',Aa2)
    print('Aa3',Aa3)
    print('Group Num',group_num)
    pred1 = pred.cpu()
    pred2 = pred1.numpy()
    gt1 = gt.cpu()
    gt2 = gt1.numpy()
#     groups1 = groups.cpu()
#     groups2 = groups1.numpy
    np.savez('/prj0129/mil4012/glaucoma/NIH-chest-x-ray/CXR8/Result/densenet201_mimic_gender_age_nofinding4.npz', prediction=pred2, label=gt2, group=groups) 
    np.savetxt('/prj0129/mil4012/glaucoma/NIH-chest-x-ray/CXR8/Result/densenet201_mimic_gender_age_nofinding4.txt', pred2)
            
        

if __name__ == '__main__':
    
    train_sampler = None
    batch_size = 96
    workers = 4
    N_CLASSES = 1
    CLASS_NAMES = 'MIMIC'
    

    #get data and label for training, validate, and testing dataset.
    
    #training dataset
    tmp = np.loadtxt(image_path_train, dtype=np.str, delimiter=",")
    train_path = tmp[:,0]
    train_path = train_path[1:len(train_path)]
    # train_path = train_path[1:2000]
    #7-> lung lesion, 12->pneumonia, 13->pneumothorax, 9->no finding, 16->gender, 17->age, 18->race
    labels = tmp[:,9]
    print('the disease is',labels[0])
    labels = labels[1:len(labels)]  
    # labels = labels[1:2000]
    gender = tmp[:,16]
    gender = gender[1:len(gender)]
    
    age = tmp[:,17]
    age = age[1:len(age)]
    
    race = tmp[:,18]
    race = race[1:len(race)]
    
    
    train_label = copy.deepcopy(labels)
    ind = np.argwhere(labels=='1.0')
    train_label[ind] = 1
    ind = np.argwhere(labels!='1.0')
    train_label[ind] = 0
    train_label = np.asarray(train_label, dtype=int)
    
    ##gender and age
    train_groups = copy.deepcopy(gender)
    age = np.asarray(age, dtype=float)
    ind = np.argwhere(gender=='M')
    ind1 = np.argwhere(age < 60)
    ind2 = np.intersect1d(ind, ind1)
    train_groups[ind2] = 0
    
    ind = np.argwhere(gender=='M')
    ind1 = np.argwhere(age >= 60)
    ind2 = np.intersect1d(ind, ind1)
    train_groups[ind2] = 1
    
    ind = np.argwhere(gender=='F')
    ind1 = np.argwhere(age < 60)
    ind2 = np.intersect1d(ind, ind1)
    train_groups[ind2] = 2
    
    ind = np.argwhere(gender=='F')
    ind1 = np.argwhere(age >= 60)
    ind2 = np.intersect1d(ind, ind1)
    train_groups[ind2] = 3
    
    
    
#     ##gender and race
#     train_groups = copy.deepcopy(gender)
#     ind = np.argwhere(gender=='M')
#     ind1 = np.argwhere(race=='BLACK/AFRICAN AMERICAN')
#     ind2 = np.intersect1d(ind, ind1)
#     train_groups[ind2] = 0
    
#     ind = np.argwhere(gender=='M')
#     ind1 = np.argwhere(race!='BLACK/AFRICAN AMERICAN')
#     ind2 = np.intersect1d(ind, ind1)
#     train_groups[ind2] = 1
    
#     ind = np.argwhere(gender=='F')
#     ind1 = np.argwhere(race=='BLACK/AFRICAN AMERICAN')
#     ind2 = np.intersect1d(ind, ind1)
#     train_groups[ind2] = 2
    
#     ind = np.argwhere(gender=='F')
#     ind1 = np.argwhere(race!='BLACK/AFRICAN AMERICAN')
#     ind2 = np.intersect1d(ind, ind1)
#     train_groups[ind2] = 3
    
    
    
#     ##age and race
#     train_groups = copy.deepcopy(race)
#     age = np.asarray(age, dtype=float)
#     ind = np.argwhere(age < 60)
#     ind1 = np.argwhere(race=='BLACK/AFRICAN AMERICAN')
#     ind2 = np.intersect1d(ind, ind1)
#     train_groups[ind2] = 0
    
#     ind = np.argwhere(age < 60)
#     ind1 = np.argwhere(race !='BLACK/AFRICAN AMERICAN')
#     ind2 = np.intersect1d(ind, ind1)
#     train_groups[ind2] = 1
    
#     ind = np.argwhere(age >= 60)
#     ind1 = np.argwhere(race =='BLACK/AFRICAN AMERICAN')
#     ind2 = np.intersect1d(ind, ind1)
#     train_groups[ind2] = 2
    
#     ind = np.argwhere(age >= 60)
#     ind1 = np.argwhere(race!='BLACK/AFRICAN AMERICAN')
#     ind2 = np.intersect1d(ind, ind1)
#     train_groups[ind2] = 3
    
    train_groups = np.asarray(train_groups, dtype=int)
    
    
    #val dataset
    tmp = np.loadtxt(image_path_val, dtype=np.str, delimiter=",")
    val_path = tmp[:,0]
    val_path = val_path[1:len(val_path)]
    #7-> lung lesion, 12->pneumonia, 13->pneumothorax, 9->no finding, 16->gender, 17->age, 18->race
    labels = tmp[:,9]
    print('the disease is',labels[0])
    labels = labels[1:len(labels)]  
    
    gender = tmp[:,16]
    gender = gender[1:len(gender)]
    
    age = tmp[:,17]
    age = age[1:len(age)]
    
    race = tmp[:,18]
    race = race[1:len(race)]
    
    
    val_label = copy.deepcopy(labels)
    ind = np.argwhere(labels=='1.0')
    val_label[ind] = 1
    ind = np.argwhere(labels!='1.0')
    val_label[ind] = 0
    val_label = np.asarray(val_label, dtype=int)
    
    
    ##gender and age
    val_groups = copy.deepcopy(gender)
    age = np.asarray(age, dtype=float)
    ind = np.argwhere(gender=='M')
    ind1 = np.argwhere(age < 60)
    ind2 = np.intersect1d(ind, ind1)
    val_groups[ind2] = 0
    
    ind = np.argwhere(gender=='M')
    ind1 = np.argwhere(age >= 60)
    ind2 = np.intersect1d(ind, ind1)
    val_groups[ind2] = 1
    
    ind = np.argwhere(gender=='F')
    ind1 = np.argwhere(age < 60)
    ind2 = np.intersect1d(ind, ind1)
    val_groups[ind2] = 2
    
    ind = np.argwhere(gender=='F')
    ind1 = np.argwhere(age >= 60)
    ind2 = np.intersect1d(ind, ind1)
    val_groups[ind2] = 3
    
    
    
#     ##gender and race
#     val_groups = copy.deepcopy(gender)
#     ind = np.argwhere(gender=='M')
#     ind1 = np.argwhere(race=='BLACK/AFRICAN AMERICAN')
#     ind2 = np.intersect1d(ind, ind1)
#     val_groups[ind2] = 0
    
#     ind = np.argwhere(gender=='M')
#     ind1 = np.argwhere(race!='BLACK/AFRICAN AMERICAN')
#     ind2 = np.intersect1d(ind, ind1)
#     val_groups[ind2] = 1
    
#     ind = np.argwhere(gender=='F')
#     ind1 = np.argwhere(race=='BLACK/AFRICAN AMERICAN')
#     ind2 = np.intersect1d(ind, ind1)
#     val_groups[ind2] = 2
    
#     ind = np.argwhere(gender=='F')
#     ind1 = np.argwhere(race!='BLACK/AFRICAN AMERICAN')
#     ind2 = np.intersect1d(ind, ind1)
#     val_groups[ind2] = 3
    
    
    
#     ##age and race
#     val_groups = copy.deepcopy(race)
#     age = np.asarray(age, dtype=float)
#     ind = np.argwhere(age < 60)
#     ind1 = np.argwhere(race=='BLACK/AFRICAN AMERICAN')
#     ind2 = np.intersect1d(ind, ind1)
#     val_groups[ind2] = 0
    
#     ind = np.argwhere(age < 60)
#     ind1 = np.argwhere(race !='BLACK/AFRICAN AMERICAN')
#     ind2 = np.intersect1d(ind, ind1)
#     val_groups[ind2] = 1
    
#     ind = np.argwhere(age >= 60)
#     ind1 = np.argwhere(race =='BLACK/AFRICAN AMERICAN')
#     ind2 = np.intersect1d(ind, ind1)
#     val_groups[ind2] = 2
    
#     ind = np.argwhere(age >= 60)
#     ind1 = np.argwhere(race!='BLACK/AFRICAN AMERICAN')
#     ind2 = np.intersect1d(ind, ind1)
#     val_groups[ind2] = 3
    

    
    val_groups = np.asarray(val_groups, dtype=int)
        
    #test dataset
    tmp = np.loadtxt(image_path_test, dtype=np.str, delimiter=",")
    test_path = tmp[:,0]
    test_path = test_path[1:len(test_path)]
    #7-> lung lesion, 12->pneumonia, 13->pneumothorax, 9->no finding, 16->gender, 17->age, 18->race
    labels = tmp[:,9]
    print('the disease is',labels[0])
    labels = labels[1:len(labels)]  
    
    gender = tmp[:,16]
    gender = gender[1:len(gender)]
    
    age = tmp[:,17]
    age = age[1:len(age)]
    
    race = tmp[:,18]
    race = race[1:len(race)]
    

    test_label = copy.deepcopy(labels)
    ind = np.argwhere(labels=='1.0')
    test_label[ind] = 1
    ind = np.argwhere(labels!='1.0')
    test_label[ind] = 0
    test_label = np.asarray(test_label, dtype=int)
    
    
    ##gender and age
    test_groups = copy.deepcopy(gender)
    age = np.asarray(age, dtype=float)
    ind = np.argwhere(gender=='M')
    ind1 = np.argwhere(age < 60)
    ind2 = np.intersect1d(ind, ind1)
    test_groups[ind2] = 0
    
    ind = np.argwhere(gender=='M')
    ind1 = np.argwhere(age >= 60)
    ind2 = np.intersect1d(ind, ind1)
    test_groups[ind2] = 1
    
    ind = np.argwhere(gender=='F')
    ind1 = np.argwhere(age < 60)
    ind2 = np.intersect1d(ind, ind1)
    test_groups[ind2] = 2
    
    ind = np.argwhere(gender=='F')
    ind1 = np.argwhere(age >= 60)
    ind2 = np.intersect1d(ind, ind1)
    test_groups[ind2] = 3
    
    
    
#     ##gender and race
#     test_groups = copy.deepcopy(gender)
#     ind = np.argwhere(gender=='M')
#     ind1 = np.argwhere(race=='BLACK/AFRICAN AMERICAN')
#     ind2 = np.intersect1d(ind, ind1)
#     test_groups[ind2] = 0
    
#     ind = np.argwhere(gender=='M')
#     ind1 = np.argwhere(race!='BLACK/AFRICAN AMERICAN')
#     ind2 = np.intersect1d(ind, ind1)
#     test_groups[ind2] = 1
    
#     ind = np.argwhere(gender=='F')
#     ind1 = np.argwhere(race=='BLACK/AFRICAN AMERICAN')
#     ind2 = np.intersect1d(ind, ind1)
#     test_groups[ind2] = 2
    
#     ind = np.argwhere(gender=='F')
#     ind1 = np.argwhere(race!='BLACK/AFRICAN AMERICAN')
#     ind2 = np.intersect1d(ind, ind1)
#     test_groups[ind2] = 3
    
    
    
#     ##age and race
#     test_groups = copy.deepcopy(race)
#     age = np.asarray(age, dtype=float)
#     ind = np.argwhere(age < 60)
#     ind1 = np.argwhere(race =='BLACK/AFRICAN AMERICAN')
#     ind2 = np.intersect1d(ind, ind1)
#     test_groups[ind2] = 0
    
#     ind = np.argwhere(age < 60)
#     ind1 = np.argwhere(race !='BLACK/AFRICAN AMERICAN')
#     ind2 = np.intersect1d(ind, ind1)
#     test_groups[ind2] = 1
    
#     ind = np.argwhere(age >= 60)
#     ind1 = np.argwhere(race =='BLACK/AFRICAN AMERICAN')
#     ind2 = np.intersect1d(ind, ind1)
#     test_groups[ind2] = 2
    
#     ind = np.argwhere(age >= 60)
#     ind1 = np.argwhere(race !='BLACK/AFRICAN AMERICAN')
#     ind2 = np.intersect1d(ind, ind1)
#     test_groups[ind2] = 3
    
    
    test_groups = np.asarray(test_groups, dtype=int)
                        
    train_label = train_label.astype(np.float)
    val_label = val_label.astype(np.float) 
    test_label = test_label.astype(np.float) 
    
    train_label = np.reshape(train_label,(len(train_label),1)) 
    val_label = np.reshape(val_label,(len(val_label),1)) 
    test_label = np.reshape(test_label,(len(test_label),1)) 
    
    

    
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(10),
            # transforms.ToPILImage(),
            transforms.Resize(224),
            # transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(224),
            # transforms.CenterCrop(224),
            # transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    train_dataset = Dataset(train_path,train_label,groups=train_groups,transform = data_transforms["train"])
    val_dataset = Dataset(val_path,val_label,groups=val_groups,transform = data_transforms["val"])
    test_dataset = Dataset(test_path,test_label,groups=test_groups,transform = data_transforms["val"])
    
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(train_sampler is None), 
                                           num_workers=workers, pin_memory=True, sampler=train_sampler)
        
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, 
#                                            num_workers=workers, pin_memory=True, sampler=train_sampler)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=(train_sampler is None), 
                                           num_workers=workers, pin_memory=True, sampler=train_sampler)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                                           num_workers=workers, pin_memory=True, sampler=train_sampler)

    dataloaders = {"train": train_loader, "val": val_loader}
    
    model_ft = models.densenet201(pretrained=True)
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Sequential(
                nn.Linear(num_ftrs, N_CLASSES),
                nn.Sigmoid()
            )
    model_ft = model_ft.to(device)
    
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.00005)
    
    # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    
    # Decay LR by a factor of 0.1 every 7 epochs
#     exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, 'min', patience=2, eps=1e-08, verbose=True)

    # model_ft = train_model(dataloaders, model_ft, criterion, optimizer_ft, exp_lr_scheduler,
    #                        num_epochs=20)
    model_ft = train_model(dataloaders, model_ft, criterion, optimizer_ft,num_epochs=20)
    model_ft.load_state_dict(torch.load("/prj0129/mil4012/glaucoma/NIH-chest-x-ray/CXR8/weights/densenet201_mimic_gender_age_nofinding4.pth"))
    test_model(test_loader,model_ft)