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
if not os.path.exists("./model_mt/"):
    os.makedirs("./model_mt/")

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

# image_datasets['train7'] = datasets.ImageFolder(os.path.join(train_dir, 'sudo_label' , "7"),   #####???????????????????????
#                                           data_transforms['train'])

# image_datasets['train8'] = datasets.ImageFolder(os.path.join(train_dir, 'sudo_label' , "8"),
#                                           data_transforms['train'])


# image_datasets['val'] = datasets.ImageFolder(os.path.join(train_dir, 'val'),
                                          # data_transforms['val'])





dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=True, num_workers=16)
              for x in ['train1','train2','train3','train4','train5','train6']}#,'train7','train8']}  ########???????????????????????????????
dataset_sizes = {x: len(image_datasets[x]) for x in ['train1','train2','train3','train4','train5','train6']}#,'train7','train8']}
class_names1 = len(image_datasets['train1'].classes)
class_names2 = len(image_datasets['train2'].classes)
class_names3 = len(image_datasets['train3'].classes)
class_names4 = len(image_datasets['train4'].classes)
class_names5 = len(image_datasets['train5'].classes)
class_names6 = len(image_datasets['train6'].classes)
# class_names7 = len(image_datasets['train7'].classes) ###########????????????????????????????????
# class_names8 = len(image_datasets['train8'].classes)

# print([len(class_names1),len(class_names2),len(class_names3, class_names4, class_names5, class_names6])

use_gpu = torch.cuda.is_available()

# inputs, classes = next(iter(dataloaders['train']))

######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.

y_loss = {} # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []

def train_model(model,model_teacher, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    teacher = model.state_dict()

    save_path = os.path.join('./model_mt/ft_ResNet50_duke_e/net_last_teacher.pth')
    # model_teacher.load_state_dict(torch.load(save_path))
     
    

    S = nn.Softmax(dim=1)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        if epoch>2:
            model_teacher.load_state_dict(teacher)
            model_teacher.eval()



        # Each epoch has a training and validation phase
        for phase in ['train1', 'train2', 'train3', 'train4', 'train5', 'train6']:#,'train7','train8']:  #??????????????????????????????
           
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data
                #print(inputs.shape)
                # wrap them in Variable

                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                
                # outputs = model(inputs)
            
                if phase =="train1":
                    k = 0
                if phase =="train2":
                    k = 1
                if phase =="train3":
                    k = 2
                if phase =="train4":
                    k = 3
                if phase =="train5":
                    k = 4
                if phase =="train6":
                    k = 5
                # if phase =="train7":
                #     k = 6
                # if phase =="train8":
                #     k = 7
                    ###########????????????????????????????????????
                
                if inputs.size()[0] == 1:
                    inputs = torch.cat((inputs, inputs), 0)
                    labels = torch.cat((labels, labels), 0)
        

                outputs,_ = model(inputs)

                loss_teacher = torch.zeros(8)

                noise = torch.randn(inputs.size())
                noise = noise*0.1
                noise = Variable(noise.cuda())
                
                if epoch >2:    
                    teacher_prob,teacher_feature = model_teacher(inputs+noise)
                    student_prob, student_feature= model(inputs)
                
                    loss_teacher = nll(teacher_prob,student_prob)
                # = 0.2*criterion(outputs[k], labels)+
                    # print(loss_teacher, criterion(outputs[k], labels))
                    loss  = 0.8*criterion(outputs[k], labels)#+10*loss_teacher
                # print(loss_teacher.item(), loss.item(),criterion(outputs[k], labels).item())
                else:
                    loss  = 0.8*criterion(outputs[k], labels)

                loss.backward()
                optimizer.step()
                # statistics
                running_loss += loss.item()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            
            # 

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))

        if epoch%10 == 9:
            save_network(model, epoch)
           
        # teacher = model_teacher.state_dict()
        # student = model.state_dict()
        

        if epoch == 0:
            student = model.state_dict()
    
        if epoch==1:
            teacher = model_teacher.state_dict()
        
        if epoch>1:
            for key in teacher:
                teacher[key] = 0.7*teacher[key] + 0.3*student[key]
        
                student = model.state_dict()


        
        
        print()
        save_network(model, epoch)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    save_network(model, 'last')
    # model.load_state_dict(spare_t)

    save_network(model, 'last_t')
    return model


######################################################################
# Draw Curve
#---------------------------
x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")
def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig( os.path.join('./model',model_name,'train.jpg'))

######################################################################
# Save model
#---------------------------
def save_network(network, epoch_label):
    save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join('./model_mt',model_name,save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available:
        network.cuda(gpu_ids[0])

def nll(x,y):
    S = nn.Softmax(dim=1)
    mse = nn.MSELoss()
    loss = 0
    for i in range(6):    ################???????????????????????????for i in range(6):#
        xs = S(x[i])
        ys = S(y[i])
        for j in range(xs.size(0)):
            # loss = loss-1*torch.round(xs[j,torch.argmax(xs[j])]-0.2)*torch.log(ys[j,(torch.argmax(xs[j]))])#-1*xs[j,torch.argmax(xs[j])]*torch.log(ys[j,(torch.argmax(xs[j]))])
            # loss = loss-1*xs[j,torch.argmax(xs[j])]*torch.log(ys[j,(torch.argmax(xs[j]))])
            loss = loss +mse(xs,ys)
        # loss = loss/xs.size(0)



    
    # print(loss,"*************")



    return loss 

######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrainied model and reset final fully connected layer.
#

if opt.use_dense:
    model = ft_net_dense(len(class_names))
else:
    model = ft_net_test([class_names1,class_names2,class_names3, class_names4, class_names5, class_names6])#,class_names7, class_names8])  #########??????????????????????
    model_teacher = ft_net_test([class_names1,class_names2,class_names3, class_names4, class_names5, class_names6])#,class_names7, class_names8])
    # model_teacher = ft_net([class_names1,class_names2,class_names3, class_names4, class_names5, class_names6,class_names7, class_names8])

if opt.PCB:
    model = PCB(len(class_names))

print(model)

if use_gpu:
    model = model.cuda()
    model_teacher = model_teacher.cuda()

criterion = nn.CrossEntropyLoss()

if not opt.PCB:
    # ignored_params = list(map(id, model.model.fc.parameters() )) + list(map(id, model.classifier.parameters() ))
    # base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    optimizer_ft = optim.SGD([
             {'params': model.parameters(), 'lr': 0.01},
             # {'params': model.model.fc.parameters(), 'lr': 0.1},
             # {'params': model.classifier0.parameters(), 'lr': 0.1},
             # {'params': model.classifier1.parameters(), 'lr': 0.1},
             # {'params': model.classifier2.parameters(), 'lr': 0.1},
             # {'params': model.classifier3.parameters(), 'lr': 0.1},
             # {'params': model.classifier4.parameters(), 'lr': 0.1},
             # {'params': model.classifier5.parameters(), 'lr': 0.1},
         ], weight_decay=5e-4, momentum=0.9, nesterov=True)
else:
    ignored_params = list(map(id, model.model.fc.parameters() ))
    ignored_params += (list(map(id, model.classifier0.parameters() )) 
                     +list(map(id, model.classifier1.parameters() ))
                     +list(map(id, model.classifier2.parameters() ))
                     +list(map(id, model.classifier3.parameters() ))
                     +list(map(id, model.classifier4.parameters() ))
                     +list(map(id, model.classifier5.parameters() ))
                     #+list(map(id, model.classifier6.parameters() ))
                     #+list(map(id, model.classifier7.parameters() ))
                      )
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    optimizer_ft = optim.SGD([
             {'params': base_params, 'lr': 0.01},
             {'params': model.model.fc.parameters(), 'lr': 0.1},
             {'params': model.classifier0.parameters(), 'lr': 0.1},
             {'params': model.classifier1.parameters(), 'lr': 0.1},
             {'params': model.classifier2.parameters(), 'lr': 0.1},
             {'params': model.classifier3.parameters(), 'lr': 0.1},
             {'params': model.classifier4.parameters(), 'lr': 0.1},
             {'params': model.classifier5.parameters(), 'lr': 0.1},
             #{'params': model.classifier6.parameters(), 'lr': 0.01},
             #{'params': model.classifier7.parameters(), 'lr': 0.01}
         ], weight_decay=5e-4, momentum=0.9, nesterov=True)

# Decay LR by a factor of 0.1 every 40 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)

######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# It should take around 1-2 hours on GPU. 
#
dir_name = os.path.join('./model_mt',model_name)
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)

# save opts
with open('%s/opts.json'%dir_name,'w') as fp:
    json.dump(vars(opt), fp, indent=1)

model = train_model(model,model_teacher, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=50)
