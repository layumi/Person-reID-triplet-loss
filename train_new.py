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
import copy
from PIL import Image
import time
import os
#from reid_sampler import StratifiedSampler
from model import ft_net, ft_net_dense, PCB
from random_erasing import RandomErasing
from tripletfolder import TripletFolder
import json
from shutil import copyfile

version =  torch.__version__

#fp16
try:
    from apex.fp16_utils import *
    from apex import amp, optimizers
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')


######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name',default='ft_ResNet50', type=str, help='output model name')
parser.add_argument('--data_dir',default='../Market/pytorch',type=str, help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data' )
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--poolsize', default=128, type=int, help='poolsize')
parser.add_argument('--margin', default=0.3, type=float, help='margin')
parser.add_argument('--lr', default=0.01, type=float, help='margin')
parser.add_argument('--alpha', default=0.0, type=float, help='regularization, push to -1')
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--PCB', action='store_true', help='use PCB+ResNet50' )
parser.add_argument('--fp16', action='store_true', help='use float16 instead of float32, which will save about 50% memory' )
opt = parser.parse_args()

data_dir = opt.data_dir
name = opt.name
fp16 = opt.fp16
str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >=0:
        gpu_ids.append(gid)

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
#print(gpu_ids[0])


######################################################################
# Load Data
# ---------
#

transform_train_list = [
        #transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        transforms.Resize((256,128), interpolation=3),
        #transforms.RandomCrop((256,128)),
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
image_datasets['train'] = TripletFolder(os.path.join(data_dir, 'train_all'),
                                          data_transforms['train'])
image_datasets['val'] = TripletFolder(os.path.join(data_dir, 'val'),
                                          data_transforms['val'])

batch = {}

class_names = image_datasets['train'].classes
class_vector = [s[1] for s in image_datasets['train'].samples]
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize, 
                                             shuffle=True, num_workers=8)
              for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

use_gpu = torch.cuda.is_available()

since = time.time()
#inputs, classes, pos, pos_classes = next(iter(dataloaders['train']))
print(time.time()-since)


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

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0
    last_margin = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            running_margin = 0.0
            running_reg = 0.0
            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels, pos, pos_labels = data
                now_batch_size,c,h,w = inputs.shape
                
                if now_batch_size<opt.batchsize: # next epoch
                    continue
                pos = pos.view(4*opt.batchsize,c,h,w)
                #copy pos 4times
                pos_labels = pos_labels.repeat(4).reshape(4,opt.batchsize)
                pos_labels = pos_labels.transpose(0,1).reshape(4*opt.batchsize)

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    pos = Variable(pos.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                #model_eval = copy.deepcopy(model)
                #model_eval = model_eval.eval()
                outputs, f = model(inputs)
                _, pf = model(pos)
                #pf = Variable( pf, requires_grad=True)
                neg_labels = pos_labels
                # hard-neg
                # ----------------------------------
                nf_data = pf # 128*512
                # 128 is too much, we use pool size = 64
                rand = np.random.permutation(4*opt.batchsize)[0:opt.poolsize]
                nf_data = nf_data[rand,:]
                neg_labels = neg_labels[rand]
                nf_t = nf_data.transpose(0,1) # 512*128
                score = torch.mm(f.data, nf_t) # cosine 32*128 
                score, rank = score.sort(dim=1, descending = True) # score high == hard
                labels_cpu = labels.cpu()
                nf_hard = torch.zeros(f.shape).cuda()
                for k in range(now_batch_size):
                    hard = rank[k,:]
                    for kk in hard:
                        now_label = neg_labels[kk] 
                        anchor_label = labels_cpu[k]
                        if now_label != anchor_label:
                            nf_hard[k,:] = nf_data[kk,:]
                            break

                # hard-pos
                # ----------------------------------
                pf_hard = torch.zeros(f.shape).cuda() # 32*512
                for k in range(now_batch_size):
                    pf_data = pf[4*k:4*k+4,:]
                    pf_t = pf_data.transpose(0,1) # 512*4
                    ff = f.data[k,:].reshape(1,-1) # 1*512
                    score = torch.mm(ff, pf_t) #cosine
                    score, rank = score.sort(dim=1, descending = False) #score low == hard
                    pf_hard[k,:] = pf_data[rank[0][0],:]

                # loss
                # ---------------------------------
                criterion_triplet = nn.MarginRankingLoss(margin=opt.margin)                
                pscore = torch.sum( f * pf_hard, dim=1) 
                nscore = torch.sum( f * nf_hard, dim=1)
                y = torch.ones(now_batch_size)
                y = Variable(y.cuda())

                if not opt.PCB:
                    _, preds = torch.max(outputs.data, 1)
                    #loss = criterion(outputs, labels)
                    #loss_triplet = criterion_triplet(f, pf, nf)
                    reg = torch.sum((1+nscore)**2) + torch.sum((-1+pscore)**2)
                    loss = torch.sum(torch.nn.functional.relu(nscore + opt.margin - pscore))  #Here I use sum
                    loss_triplet = loss + opt.alpha*reg
                else:
                    part = {}
                    sm = nn.Softmax(dim=1)
                    num_part = 6
                    for i in range(num_part):
                        part[i] = outputs[i]

                    score = sm(part[0]) + sm(part[1]) +sm(part[2]) + sm(part[3]) +sm(part[4]) +sm(part[5])
                    _, preds = torch.max(score.data, 1)

                    loss = criterion(part[0], labels)
                    for i in range(num_part-1):
                        loss += criterion(part[i+1], labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    if fp16: # we use optimier to backward loss
                        with amp.scale_loss(loss_triplet, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss_triplet.backward()
                    
                    optimizer.step()
                # statistics
                if int(version[0])>0 or int(version[2]) > 3: # for the new version like 0.4.0 and 0.5.0
                    running_loss += loss_triplet.item() #* opt.batchsize
                else :  # for the old version like 0.3.0 and 0.3.1
                    running_loss += loss_triplet.data[0] #*opt.batchsize
                #print( loss_triplet.item())
                running_corrects += float(torch.sum(pscore>nscore+opt.margin))
                running_margin +=float(torch.sum(pscore-nscore))
                running_reg += reg

            datasize = dataset_sizes['train']//opt.batchsize * opt.batchsize
            epoch_loss = running_loss / datasize
            epoch_reg = opt.alpha*running_reg/ datasize
            epoch_acc = running_corrects / datasize
            epoch_margin = running_margin / datasize

            #if epoch_acc>0.75:
            #    opt.margin = min(opt.margin+0.02, 1.0)
            print('now_margin: %.4f'%opt.margin)           
            print('{} Loss: {:.4f} Reg: {:.4f} Acc: {:.4f} MeanMargin: {:.4f}'.format(
                phase, epoch_loss, epoch_reg, epoch_acc, epoch_margin))
            if phase == 'train':
                scheduler.step()
            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0-epoch_acc)
            # deep copy the model
            if epoch_margin>last_margin:
                last_margin = epoch_margin            
                last_model_wts = model.state_dict()

            if epoch%10 == 9:
                save_network(model, epoch)
            draw_curve(epoch)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(last_model_wts)
    save_network(model, 'last')
    return model


######################################################################
# Draw Curve
#---------------------------
x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="triplet_loss")
ax1 = fig.add_subplot(122, title="top1err")
def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
#    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
#    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig( os.path.join('./model',name,'train.jpg'))

######################################################################
# Save model
#---------------------------
def save_network(network, epoch_label):
    save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join('./model',name,save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available:
        network.cuda(gpu_ids[0])


######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrainied model and reset final fully connected layer.
#

if opt.use_dense:
    model = ft_net_dense(len(class_names))
else:
    model = ft_net(len(class_names))

if opt.PCB:
    model = PCB(len(class_names))

print(model)

if use_gpu:
    model = model.cuda()

criterion = nn.CrossEntropyLoss()

if not opt.PCB:
    ignored_params = list(map(id, model.model.fc.parameters() )) + list(map(id, model.classifier.parameters() ))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    optimizer_ft = optim.SGD([
             {'params': base_params, 'lr': 0.1*opt.lr},
             {'params': model.model.fc.parameters(), 'lr': opt.lr},
             {'params': model.classifier.parameters(), 'lr': opt.lr}
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
             {'params': base_params, 'lr': 0.001},
             {'params': model.model.fc.parameters(), 'lr': 0.01},
             {'params': model.classifier0.parameters(), 'lr': 0.01},
             {'params': model.classifier1.parameters(), 'lr': 0.01},
             {'params': model.classifier2.parameters(), 'lr': 0.01},
             {'params': model.classifier3.parameters(), 'lr': 0.01},
             {'params': model.classifier4.parameters(), 'lr': 0.01},
             {'params': model.classifier5.parameters(), 'lr': 0.01},
             #{'params': model.classifier6.parameters(), 'lr': 0.01},
             #{'params': model.classifier7.parameters(), 'lr': 0.01}
         ], weight_decay=5e-4, momentum=0.9, nesterov=True)

exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[40,60], gamma=0.1)

######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# It should take around 1-2 hours on GPU. 
#
dir_name = os.path.join('./model',name)
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)
    copyfile('./train_new.py', dir_name+'/train_new.py')
    copyfile('./model.py', dir_name+'/model.py')
    copyfile('./tripletfolder.py', dir_name+'/tripletfolder.py')

# save opts
with open('%s/opts.json'%dir_name,'w') as fp:
    json.dump(vars(opt), fp, indent=1)
    
if fp16:
    model, optimizer_ft = amp.initialize(model, optimizer_ft, opt_level = "O1")

model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=70)

