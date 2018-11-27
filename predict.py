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

AP = argeparse.ArgumentParser(description='predict.py')
AP.add_argument('img', nargs='*', action="store", type = str, default="aind-project/flowers/test/15/image_06351.jps")
AP.add_argument('--gpu', dest="gpu", action="store", default="gpu")
AP.add_argument('checkpoint', nargs='*', action="store", type = str, default="aind-project/classifier.pth")
AP.add_argument('--top_k', default = 5, dest="top_k", action="store", type = int)
Ap.add_argument('--category_names', dest= "category_names", action="store", default='cat_to_name.json')

parse = AP.parse_args()
#path_image = parse.img
number_of_outputs = parse.top_k
power = parse.gpu
input_img = parse.img
checkpoint_path = parse.checkpoint

dataloaders= model_structure.load_data()

model_structure.load_checkpoint(checkpoint_path)

 # CPU
 device = torch.device("cpu")

# GPU
if parse.gpu:
 device = torch.device("cuda:0")

with open(category_names) as json_file:
    cat_to_name = json.load(json_file)
    
top_prob, top_classes = model_structure.predict(input_image, model, number_of_outputs)

 for i in range(len(top_prob)):
        print(json_file"{cat_to_name[top_classes[i]]:<25} {top_prob[i]*100:.2f}%")



    
    
    
    
    
    
    
    #probabilities = model_structure.predict(path_image, model, number_of_outputs)

#flowers_names =  [cat_to_name[str(index)] 
                     #for index in np.array(probabilities[1][0])]
#probability = np.array(probabilities[0][0])

#i=0
#while i < number_of_outputs:
#    print("{} with a probability of {}".format(flowers_names[i],
#                                                probability[i]))
 
#    i +=1

#print("Voila!")