# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 13:58:44 2015

@author: test
"""

import sys
sys.path.append("../scripts/")
import helperFuncs
import numpy as np
from os import listdir
from random import shuffle
import cPickle

#from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad, RMSprop
sys.setrecursionlimit(10000)
np.random.seed(0)
"""Require an argument specifying whether this is an update or a new model, parse input"""
#update, size, lay1size, run     = handleArgs(sys.argv,size=300)



"""Define parameters of the run"""

size            = 300           #EDIT ME!   #how large the images are
outType         = "justbonds" #EDIT ME!   #what the CNN is predicting

#imdim           = size - 20                 #strip 10 pixels buffer from each size
direct          = "../data/SDF/"            #directory containing the SD files
ld              = listdir(direct)                   #contents of that directory
shuffle(ld)                                 #shuffle the image list for randomness
numEx           = len(ld)                   #number of images in the directory
DUMP_WEIGHTS    = True                      #will we dump the weights of conv layers for visualization
trainTestSplit  = 0.90                      #percentage of data to use as training data
batch_size      = 32                        #how many training examples per batch
chunkSize       = 50000                     #how much data to ever load at once      
testChunkSize   = 6000                      #how many examples to evaluate per iteration
run             = "1"





"""Define the folder where the model will be stored based on the input arguments"""
folder          = helperFuncs.defineFolder(False,outType,size,run)
print folder
trainDirect     = folder+"tempTrain/"
testDirect      = folder+"tempTest/"

#if update:     
#    stop = raw_input("Loading from folder "+folder+" : Hit enter to proceed or ctrl+C to cancel")
#else:
#    print "Initializing in folder "+folder





"""Load the train/test split information if update, else split and write out which images are in which dataset"""
trainFs, testFs     = helperFuncs.getTrainTestSplit(False,folder,numEx,trainTestSplit,ld)
trainL  = len(trainFs)
testL   = len(testFs)


print "number of examples: ", numEx
print "training examples : ", trainL
print "test examples : ", testL

features,labels  = helperFuncs.getTargets("justbonds")            #get the target vector for each CID
outsize         = helperFuncs.getOutSize(features)

"""DEFINE THE MODEL HERE"""  

model = Sequential()

model.add(Convolution2D(8,8, 8, input_shape=(1, size, size))) 
model.add(Activation('relu'))

model.add(Convolution2D(8, 5, 5)) 
model.add(Activation('relu'))


model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))

model.add(Convolution2D(8, 5, 5))
model.add(Activation('relu'))

model.add(Convolution2D(16, 5, 5)) 
model.add(Activation('relu'))    

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
#
model.add(Convolution2D(64, 5, 5)) 
model.add(Activation('relu'))
#
#model.add(MaxPooling2D(pool_size=(2,2)))
#
#model.add(Convolution2D(128, 4, 4)) 
#model.add(Activation('relu'))
#
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(outsize, init='normal'))

lr  = 0.000001
optimizer   = Adadelta()
model.compile(loss='mean_squared_error', optimizer=optimizer)

#    model.set_weights(getWeights("../OCRfeatures/200_5_3/bestModel.pickle"))

helperFuncs.saveModel(model,folder+"wholeModel")



