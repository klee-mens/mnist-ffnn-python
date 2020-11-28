#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 23:22:29 2020

@author: mens
"""

import numpy as np
import matplotlib.pyplot as plt
plt.close("all")

from mnist import MNIST #third party, need to be installed 
from myNetwork import Network

# =============================================================================
# loading Mnist-Data Set, converting into proper npArrays and splitting into 
# Test and Training Data
# =============================================================================
mndata = MNIST('./')
images, labels = mndata.load_training()

def int2array(lable):
    a = np.zeros(10)
    a[lable] = 1.0
    return a

images = [np.reshape(np.array(x), (784,1))  for x in images]
labels = [np.reshape(int2array(lable), (10,1)) for lable in labels]

training_data_size = 55000
training_data_reduce = 10000

training_set = [paerschen for paerschen in zip(images[0:training_data_size], labels[0:training_data_size])]
# training_set = [paerschen for paerschen in zip(images[0:training_data_size-training_data_reduce], labels[0:training_data_size-training_data_reduce])]

test_data = [pair for pair in zip(images[training_data_size::], labels[training_data_size::])]


# =============================================================================
# creating and trining my Network
# =============================================================================
    
myNet = Network([28*28, 20, 10])
myNet.SGD(training_set, 5, 20, 0.2, test_data=test_data)


# =============================================================================
# comparison and plotting
# =============================================================================

def show_sample(x,y):
    img = np.reshape(x, (28,28))
    plt.figure()
    plt.imshow(img)
    plt.title(str(np.argmax(y)))
    
    plt.figure()
    plt.plot(y)
    plt.title(str(np.argmax(y)))
    

def compare_network_sample(x,y, net):
    img = np.reshape(x, (28,28))
    plt.figure()
    plt.imshow(img)
    plt.title(str(np.argmax(y)))
    
    y = net.feedforward(x)
    plt.figure()
    plt.plot(y)
    plt.title(str(np.argmax(y)))

for i in range(20):
    x,y = test_data[i]
    compare_network_sample(x, y, myNet)