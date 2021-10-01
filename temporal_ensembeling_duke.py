
from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image
import time
import os
from model_mt import ft_net, ft_net_test
from random_erasing import RandomErasing
import json

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--train_dir',default='./dataset/modified_dataset',type=str, help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data' )
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--PCB', action='store_true', help='use PCB+ResNet50' )
parser.add_argument('--model_name', default='ft_ResNet50', help='use PCB+ResNet50' )
parser.add_argument('--source', default='duke', help='duke,market' )
opt = parser.parse_args()

train_dir = opt.train_dir+"/"+opt.source
str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >=0:
        gpu_ids.append(gid)

if opt.PCB:
    model_name = opt.model_name+"_"+"pcb"+"_"+opt.source+"_e"
else:
    model_name = opt.model_name+"_"+opt.source+"_e"

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])



#print(gpu_ids[0])
if not os.path.exists("./model_/"):
    os.makedirs("./model_/")

######################################################################
# Load Data
# ---------
#

transform_train_list = [
        #transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        transforms.Resize((288,144), interpolation=3),
        transforms.RandomCrop((256,128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

transform_val_list = [
        transforms.Resize(size=(256,128),interpolation=3), #Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

if opt.PCB:
    transform_train_list = [
        transforms.Resize((384,192), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    transform_val_list = [
        transforms.Resize(size=(384,192),interpolation=3), #Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

if opt.erasing_p>0:
    transform_train_list = transform_train_list +  [RandomErasing(probability = opt.erasing_p, mean=[0.0, 0.0, 0.0])]

if opt.color_jitter:
    transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_train_list

print(transform_train_list)
data_transforms = {
    'train': transforms.Compose( transform_train_list ),
    'val': transforms.Compose(transform_val_list),
}


train_all = ''
if opt.train_all:
     train_all = '_all'

image_datasets = {}

image_datasets['train1'] = datasets.ImageFolder(os.path.join(train_dir, 'sudo_label' , "1"),
                                          data_transforms['train'])

image_datasets['train2'] = datasets.ImageFolder(os.path.join(train_dir, 'sudo_label' , "2"),
                                          data_transforms['train'])

image_datasets['train3'] = datasets.ImageFolder(os.path.join(train_dir, 'sudo_label' , "3"),
                                          data_transforms['train'])

image_datasets['train4'] = datasets.ImageFolder(os.path.join(train_dir, 'sudo_label' , "4"),
                                          data_transforms['train'])

image_datasets['train5'] = datasets.ImageFolder(os.path.join(train_dir, 'sudo_label' , "5"),
                                          data_transforms['train'])

image_datasets['train6'] = datasets.ImageFolder(os.path.join(train_dir, 'sudo_label' , "6"),
                                          data_transforms['train'])

image_datasets['train7'] = datasets.ImageFolder(os.path.join(train_dir, 'sudo_label' , "7"),
                                          data_transforms['train'])

image_datasets['train8'] = datasets.ImageFolder(os.path.join(train_dir, 'sudo_label' , "8"),
                                          data_transforms['train'])
############???????????????????????????????????

# image_datasets['val'] = datasets.ImageFolder(os.path.join(train_dir, 'val'),
                                          # data_transforms['val'])





dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=True, num_workers=16)
              for x in ['train1','train2','train3','train4','train5','train6','train7','train8']}  #########????????????????
dataset_sizes = {x: len(image_datasets[x]) for x in ['train1','train2','train3','train4','train5','train6','train7','train8']}  ##########???????????????
class_names1 = len(image_datasets['train1'].classes)
class_names2 = len(image_datasets['train2'].classes)
class_names3 = len(image_datasets['train3'].classes)
class_names4 = len(image_datasets['train4'].classes)
class_names5 = len(image_datasets['train5'].classes)
class_names6 = len(image_datasets['train6'].classes)
class_names7 = len(image_datasets['train7'].classes)
class_names8 = len(image_datasets['train8'].classes)
######??????????????????????????????????????


# print([len(class_names1),len(class_names2),len(class_names3, class_names4, class_names5, class_names6])

use_gpu = torch.cuda.is_available()

def save_network(network, epoch_label):
    save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join('./model_mt',model_name,save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available:
        network.cuda(gpu_ids[0])




save_path = './model_mt'
# model_name = 'ft_ResNet50_market_e'
model_name = 'ft_ResNet50_duke_e'  ##########???????????????????????????????

# net_last_t.pth'

model = ft_net_test([class_names1,class_names2,class_names3, class_names4, class_names5, class_names6,class_names7, class_names8]) ####???????????????????????


n = 0
for i in range(n,40):
    net = save_path+'/'+model_name+'/'+'net_'+str(i)+'.pth'
    
    if i == 0+n:
        student = torch.load(net)
    
    if i==1+n:
        teacher = torch.load(net)
        
    if i>1+n:
        for key in teacher:
            teacher[key] = 0.7*teacher[key] + 0.3*student[key]
        
        student = torch.load(net)

# for key in teacher:
#     teacher[key] = teacher[key]/47


model.load_state_dict(teacher)

save_network(model, 'last_t')



