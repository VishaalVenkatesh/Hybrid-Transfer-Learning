#!/usr/bin/env python
# coding: utf-8

# In[5]:


'''

Generic Module Imports 

'''
import time
import os
import copy
import matplotlib.pyplot as plt


# In[3]:


'''
    Pennylane is an open source quantum machine learning package by Xanadu. Pennylane also allows to use IBMs Qisklit as an
    installed plugin. We may or may not evntually use that.
    
    Citation: 
    Ville Bergholm, Josh Izaac, Maria Schuld, Christian Gogolin, Carsten Blank, Keri McKiernan and Nathan Killoran. 
    PennyLane. arXiv, 2018. arXiv:1811.04968
'''
import pennylane as qml
from pennylane import numpy as np # Same as standard numpy, but also does automatic differentiation


# In[8]:


'''
    PyTorch is an open-source machine learning library by Facebook. It is similar to Tensorflow by Google. It's pretty 
    cool to use this to build models and import pre-trained neural nets like ResNet18. 
    
    PyTorch uses torch tensors and not numpy arrays. We may have to convert as and when necesary. 

'''

import torch
import torch.nn as nn                    #This serves as a base class for neural networks (NNs)#
import torch.optim as optim              #Contains built-in optimizer functions - like Stochastic Gradient Descent(SGD) & Adam Optimizer'''
from torch.optim import lr_scheduler     #Helps modulate learning rate based on number of epochs
import torchvision                       #PyTorchs imgage datasets and other related stuff
from torchvision import datasets, transforms, models


# In[11]:


'''

    Initialize Some Parameters - For now we will use an integrated Pennylane Device. This is sort of a simulator for 
    a quantum computerI will later attempt to use a real Quantum Computer using IBMs Q Experience but no guarantees. 
    This should not technically affect our results. A real quantum computer will be realistic in the sense that it 
    incorporates an element of uncertainty. A quantum simulator will be ideal and will have no uncertainty. 
    
'''
n_qubits = 4                # Number of qubits
step = 0.0004               # Learning rate
batch_size = 4              # Number of samples for each training step. This is the number of features taken in from ResNet18.
num_epochs = 1              # Number of training epochs. Set to 1 to train quickly. We will later change this to 30
q_depth = 6                 # Depth of the quantum circuit (number of variational layers)
gamma_lr_scheduler = 0.1    # Learning rate reduction applied every 10 epochs.
q_delta = 0.01              # Initial spread of random quantum weights
rng_seed = 0                # Seed for random number generator
start_time = time.time()    # Start of the computation timer


# In[12]:


'''
    Initialize the quantum device. Two options for names are Default & Gaussian. Shots is the number of times the circuit 
    will be run. 1024 or 2^10 is generally a good number. Wires is the number of modes to intialise the qubits in. 
    Remember qubits have a probabilistic nature and can take up various states. 
'''
dev = qml.device(name = 'default.qubit', shots = 1024, wires = 4)


# In[ ]:


#Read in data, read in ResNet18, Variational Quantum Cirucuit, Dressed Quantum Circuit, Train model Hymenoptera Dataset, Test


# In[ ]:




