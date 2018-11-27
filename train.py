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

import model_structure

AP = argeparse.ArgumentParser(description='train.py')

AP.add_argument('data_dir', nargs='*', action="store", default="./flowers")
AP.add_argument('--gpu', dest="gpu", action="store", default="gpu")
AP.add_argument('--save_dir', dest="save_dir", action="store", default="classifier.pth")
AP.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
AP.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
AP.add_argument('--structure', dest="structure", action="store", default="vgg16", type = str)
AP.add_argument('--hidden_layer', dest="hidden_layer", action]"store", type = int, default = 1024)

parse= AP.parse_args()
data_dir = parse.data_dir
power = parse.gpu
checkpoint_path = parse.save_dir
lr = parse.learning_rate
epochs= parse.epochs
structure = parse.structure
hidden_layer = parse.hidden_layer

dataloaders = model_structure.load_data(data_dir)

structure, hidden_layer = model_structure.my_model(model, classifier,
                                                        criterion,optimizer)
#model, criterion, optimizer = model_structure.my_model((model, classifier, criterion,optimizer)

model_structure.train_model( model, criterion, optimizer, lr,
                            epochs, power)
model_structure.save_checkpoint(checkpoint_path, structure)

print("Model is now trained, you may proceed to rule the world")                                                       
                                                      
                               
                                                       


    
    