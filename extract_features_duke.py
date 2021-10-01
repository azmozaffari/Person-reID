# -*- coding: utf-8 -*-

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
import time
import os
import scipy.io
from model_mt import ft_net, ft_net_test

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='last_t', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir',default='./dataset/modified_dataset',type=str, help='./test_data')
parser.add_argument('--model_name', default='ft_ResNet50', type=str, help='save model path')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--PCB', action='store_true', help='use PCB' )
parser.add_argument('--source', default ='duke' , help='training domain' )
parser.add_argument('--target', default ='duke' , help='test domain' )
parser.add_argument('--query_type', default ='query' , help='query, multi_query' )



opt = parser.parse_args()
dataset = opt.target

if opt.PCB:
    model_name = opt.model_name+"_"+"pcb"+"_"+opt.source+"_e"
else:
    model_name = opt.model_name+"_"+opt.source+"_e"


str_ids = opt.gpu_ids.split(',')
#which_epoch = opt.which_epoch
# name = opt.name
test_dir = opt.test_dir+"/"+opt.target


gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
data_transforms = transforms.Compose([
        transforms.Resize((288,144), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
############### Ten Crop        
        #transforms.TenCrop(224),
        #transforms.Lambda(lambda crops: torch.stack(
         #   [transforms.ToTensor()(crop) 
          #      for crop in crops]
           # )),
        #transforms.Lambda(lambda crops: torch.stack(
         #   [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop)
          #       for crop in crops]
          # ))
])

transform_train_list = [
        #transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        transforms.Resize((288,144), interpolation=3),
        transforms.RandomCrop((256,128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

train_dir = "./dataset/modified_dataset/"+opt.source
image_datasets = {}

image_datasets['train1'] = datasets.ImageFolder(os.path.join(train_dir, 'sudo_label' , "1"),
                                          transform_train_list)

image_datasets['train2'] = datasets.ImageFolder(os.path.join(train_dir, 'sudo_label' , "2"),
                                          transform_train_list)

image_datasets['train3'] = datasets.ImageFolder(os.path.join(train_dir, 'sudo_label' , "3"),
                                          transform_train_list)

image_datasets['train4'] = datasets.ImageFolder(os.path.join(train_dir, 'sudo_label' , "4"),
                                          transform_train_list)

image_datasets['train5'] = datasets.ImageFolder(os.path.join(train_dir, 'sudo_label' , "5"),
                                          transform_train_list)

image_datasets['train6'] = datasets.ImageFolder(os.path.join(train_dir, 'sudo_label' , "6"),
                                          transform_train_list)

image_datasets['train7'] = datasets.ImageFolder(os.path.join(train_dir, 'sudo_label' , "7"),
                                          transform_train_list)

image_datasets['train8'] = datasets.ImageFolder(os.path.join(train_dir, 'sudo_label' , "8"),
                                          transform_train_list)
#################????????????????????????????????????????

# image_datasets['val'] = datasets.ImageFolder(os.path.join(train_dir, 'val'),
                                          # data_transforms['val'])





dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=True, num_workers=16)
              for x in ['train1','train2','train3','train4','train5','train6','train7','train8']}   #############?????????????????????????????????????????
dataset_sizes = {x: len(image_datasets[x]) for x in ['train1','train2','train3','train4','train5','train6','train7','train8']}   ###########?????????????
class_names1 = len(image_datasets['train1'].classes)
class_names2 = len(image_datasets['train2'].classes)
class_names3 = len(image_datasets['train3'].classes)
class_names4 = len(image_datasets['train4'].classes)
class_names5 = len(image_datasets['train5'].classes)
class_names6 = len(image_datasets['train6'].classes)
class_names7 = len(image_datasets['train7'].classes)
class_names8 = len(image_datasets['train8'].classes)

##############?????????????????????????????















if opt.PCB:
    data_transforms = transforms.Compose([
        transforms.Resize((384,192), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ])

query_type = opt.query_type
data_dir = test_dir

if query_type == 'query':
    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=16) for x in ['gallery','query']}

else:
    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','multi_query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=16) for x in ['gallery','multi_query']}


# class_names = image_datasets['query'].classes
use_gpu = torch.cuda.is_available()

######################################################################
# Load model
#---------------------------
def load_network(network):
    save_path = os.path.join('./model_mt',model_name,'net_%s.pth'%opt.which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network


######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model,dataloaders):
    model.train(False)
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        print(count)
        if opt.use_dense:
            ff = torch.FloatTensor(n,1024).zero_()
        else:
            ff = torch.FloatTensor(n,2048).zero_()
        if opt.PCB:
            ff = torch.FloatTensor(n,2048,6).zero_() # we have four parts
        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            if opt.PCB:
                outputs = model(input_img) 
            
                f = outputs.data.cpu()
            else:
                outputs,f = model(input_img)
                f = f.data.cpu()
                # outputs = outputs.data.cpu()
            
            ff = ff+f
        # norm feature
        if opt.PCB:
            # feature size (n,2048,4)
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features,ff), 0)
    return features

def get_id(img_path,dataset):
    camera_id = []
    labels = []
    frames = []
    for path, v in img_path:
        filename = path.split('/')[-1]
        label = filename[0:4]
        camera = filename.split('c')[1]
        if dataset == "duke":
            frame = filename[9:16]
        if dataset == "market":
            frame = filename[10:16]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
        frames.append(int(frame))
    return camera_id, labels, frames

gallery_path = image_datasets['gallery'].imgs

if opt.query_type == 'query':
    query_path = image_datasets['query'].imgs
else:
    query_path = image_datasets['multi_query'].imgs

gallery_cam,gallery_label, gallery_frames = get_id(gallery_path,dataset)
query_cam,query_label, query_frames = get_id(query_path,dataset)

######################################################################


# class_num is the number of classes that the pre-trained model is trained with
#Duke



# if opt.source == "duke":
#     class_num=702
#     # print(class_num)
# if opt.source == "market":
#     # market
#     class_num=752

print('-------test-----------')


model_structure = ft_net_test([class_names1,class_names2,class_names3, class_names4, class_names5, class_names6,class_names7, class_names8])
###########################?????????????????????????????????????????????????????????????????????????




model = load_network(model_structure)
# Remove the final fc layer and classifier layer
# if not opt.PCB:
#     model.model.fc = nn.Sequential()
#     model.classifier = nn.Sequential()


# Change to test mode
model = model.eval()
if use_gpu:
    model = model.cuda()

# Extract feature
gallery_feature = extract_feature(model,dataloaders['gallery'])
if opt.query_type == 'query':
    query_feature = extract_feature(model,dataloaders['query'])
else:
    query_feature = extract_feature(model,dataloaders['multi_query'])

if not os.path.isdir('./rep_'):
    os.mkdir('./rep_')

if not os.path.isdir('./rep_/'+model_name):
    os.mkdir('./rep_/'+model_name)


# Save to Matlab for check
result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'gallery_cam':gallery_cam,'gallery_frames':gallery_frames,'query_f':query_feature.numpy(),'query_label':query_label,'query_cam':query_cam,'query_frames':query_frames }
scipy.io.savemat('./rep_/'+model_name+'/'+opt.source+"_"+opt.target+'_result_'+opt.query_type+'.mat',result)


#######################################   results with avg for tracklets  ###########################
result = scipy.io.loadmat('./rep_/'+model_name+'/'+opt.source+"_"+opt.target+'_result_'+opt.query_type+'.mat')

query_feature = result['query_f']
query_cam = result['query_cam'][0]
query_label = result['query_label'][0]
query_frames = result['query_frames'][0]

gallery_feature = result['gallery_f']
gallery_cam = result['gallery_cam'][0]
gallery_label = result['gallery_label'][0]
gallery_frames = result['gallery_frames'][0]



print(query_feature.shape)
print(gallery_feature.shape)




label_list =  os.listdir(test_dir+"/"+opt.query_type)


query_ft = []
query_c = []
query_l = []
query_fr = []

for label in label_list:
    query_label = np.array(query_label).astype(str)
    query_label  = np.char.zfill(query_label,4)
    index1 = np.argwhere(query_label==label)
    # print(index1)
    for c in range(1,9):
        index2 = np.argwhere(query_cam == c)
        # print(index2)
        mask = np.intersect1d(index1, index2)
        if len(mask) > 0:
            # print(query_frames[mask].shape,(np.mean(query_frames[mask],axis = 0)))
            query_fr.append(np.mean(query_frames[mask],axis = 0))
            query_ft.append(np.mean(query_feature[mask],axis=0))
            query_c.append(c)
            query_l.append(label)

            

label_list =  os.listdir(test_dir+"/"+"gallery")
# gallery_feature = gallery_feature.numpy()

gallery_ft = []
gallery_c = []
gallery_l = []
gallery_fr = []

for label in label_list:
    gallery_label = np.array(gallery_label).astype(str)
    gallery_label  = np.char.zfill(gallery_label,4)
    index1 = np.argwhere(gallery_label==label)
    for c in range(1,9):
        index2 = np.argwhere(gallery_cam == c)
        mask = np.intersect1d(index1, index2)
        if len(mask) > 0:
            gallery_fr.append(np.mean(gallery_frames[mask],axis = 0))
            gallery_ft.append(np.mean(gallery_feature[mask],axis=0))
            gallery_c.append(c)
            gallery_l.append(label)



result = {'gallery_f':np.array(gallery_ft),'gallery_label':np.array(gallery_l),'gallery_cam':np.array(gallery_c),'gallery_frames':np.array(gallery_fr),'query_f':np.array(query_ft),'query_label':np.array(query_l),'query_cam':np.array(query_c),'query_frames':np.array(query_fr)}
scipy.io.savemat('./rep_/'+model_name+'/'+opt.source+"_"+opt.target+'_result_'+opt.query_type+'_s.mat',result)

