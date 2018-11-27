# this project looks at a dataset of flowers and uses PyTorch to accurately predict the proper class and name of a particular flowers. 
##########
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler
from collections import OrderedDict
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

import json
from workspace_utils import active_session

from PIL import Image
from __future__ import print_function, division
  
import argparse

#######################

def load_data(data_dir = "/.flowers"):
    data_dir = data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),

    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ]),
}


#data_dir = 'flowers'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'valid', 'test']}


    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=4,
                                                  shuffle=True,
                                                  num_workers=4)
                   for x in ['train', 'valid','test']}

    dataset_sizes = {x: len(image_datasets[x])
                 for x in ['train', 'valid', 'test']}

    class_names = image_datasets['train'].classes

    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    return dataloaders

#######################

structures = {"vgg16":25088,
             "densenet161": 2208,
             "SqueezeNet":512}


def my_model(structure='vgg16', dropout=0.2,
             hidden_layer=1024,lr = 0.001):
    
    
    if structure == "vgg16":
        model = models.vgg16(pretrained=True)
    elif structure == "densenet161":
        model = models.densenet161(pretrained=True)
    elif structure == "SqueezeNet":
        model = models.SqueezNet1_0(pretrained=True)
    else:
        print("wrong model, try vgg16, densenet161, or SqueezeNet")
        
    model = models.__dict__[structure](pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
    
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(structures[structure], hidden_layer)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(dropout)),
                          ('fc2', nn.Linear(hidden_layer, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        model.classifier = classifier
        
        criterion = nn.NLLLoss()
        
        optimizer = optim.Adam(model.classifier.parameters(), lr)
        
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        model = model.to(device)
        
        return model, optimizer, criterion, exp_lr_scheduler
#model,optimizer,criterion,exp_lr_scheduler = my_model('vgg16')
#######################

def train_model(model, criterion, optimizer, scheduler,
                num_epochs=15, device = 'cuda'):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
#######################

def save_checkpoint(checkpoint_path = 'classifier.pth ', structure = 'vgg16'):
    model.class_to_idx = image_datasets['train'].class_to_idx
    model.cpu()
    torch.save({'structure': structure,
            'hidden_layer': 1024,
            'state_dict': model.state_dict(), 
            'class_to_idx': model.class_to_idx}, 
            'classifier.pth')

def load_model(checkpoint_path):
    chkpt = torch.load(checkpoint_path)
    structure = chkpt['structure']
    hidden_layer = chkpt['hidden_layer']
    model,_,_,_ = my_model(structure, 0.2, hidden_layer, 0.001)
    model.class_to_idx = chkpt['class_to_idx']
    model.load_state_dict(chkpt['state_dict'])
    
    
    return model

#######################

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img_pil = Image.open(image)
   
    adjustments = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = adjustments(img_pil)
    
    return img_tensor

#######################

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    model.cpu()
    
    image = process_image(image_path)
    image = image.unsqueeze(0)
    output = model.forward(Variable(image, volatile=True))
    top_prob, top_labels = torch.topk(output, topk)
    top_prob = top_prob.exp()
    top_prob_array = top_prob.data.numpy()[0]
    
    inv_class_to_idx = {v: k for k, v in model.class_to_idx.items()}
    
    top_labels_data = top_labels.data.numpy()
    top_labels_list = top_labels_data[0].tolist()  
    
    top_classes = [inv_class_to_idx[x] for x in top_labels_list]
    
    return top_prob_array, top_classes
#######################  