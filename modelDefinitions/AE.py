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
#import cPickle

#from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, TimeDistributedDense
#from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adadelta #,SGD, Adagrad, RMSprop
from keras.layers.recurrent import  GRU
from keras.layers.extra import TimeDistributedFlatten, TimeDistributedConvolution2D, TimeDistributedMaxPooling2D
sys.setrecursionlimit(10000)
np.random.seed(0)


def im2window(image,wSize,stride):
    #create a padded image
    canvas  = np.zeros((image.shape[0]+stride-1,image.shape[1]+stride-1))
    #lay the image over the canvas
    canvas[0:image.shape[0],0:image.shape[1]] = image
    #compute max value for x, y window start positions
    ydim    = image.shape[0] - wSize + 1
    xdim    = image.shape[1] - wSize + 1
    output  = []
    outVec  = []
    xran    = np.linspace(0,xdim,(xdim/stride))
    yran    = np.linspace(0,ydim,(ydim/stride))
    
    for y in yran:
        for x in xran:
            c   = canvas[y:y+wSize,x:x+wSize]
            output.append(c)
            outVec.append(np.ndarray.flatten(c))
            
    output  = np.reshape(np.array(output),(len(output),1,wSize,wSize))
    outVec  = np.reshape(np.array(outVec),(len(output),wSize**2))
    
    return output, outVec




"""Define parameters of the run"""
size            = 300           #EDIT ME!   #how large the images are
outType         = "ae" #EDIT ME!   #what the CNN is predicting

direct          = "../data/SDF/"            #directory containing the SD files
ld              = listdir(direct)                   #contents of that directory
shuffle(ld)                                 #shuffle the image list for randomness
numEx           = len(ld)                   #number of images in the directory
trainTestSplit  = 0.90                      #percentage of data to use as training data
batch_size      = 32                        #how many training examples per batch
chunkSize       = 50000                     #how much data to ever load at once      
testChunkSize   = 6000                      #how many examples to evaluate per iteration
run             = "1"

wSize           = 60
stride          = wSize/3



"""Define the folder where the model will be stored based on the input arguments"""
folder          = helperFuncs.defineFolder(False,outType,size,run)
print folder
trainDirect     = folder+"tempTrain/"
testDirect      = folder+"tempTest/"


"""Load the train/test split information if update, else split and write out which images are in which dataset"""
trainFs, testFs     = helperFuncs.getTrainTestSplit(False,folder,numEx,trainTestSplit,ld)
trainL  = len(trainFs)
testL   = len(testFs)


print "number of examples: ", numEx
print "training examples : ", trainL
print "test examples : ", testL

#features,labels = helperFuncs.getTargets("justbonds")            #get the target vector for each CID


fakeImage   = np.ones((size,size))
test,oVecs  = im2window(fakeImage,wSize,stride)
timeSteps   = test.shape[0]
outsize     = wSize**2 
print timeSteps, outsize


"""DEFINE THE MODEL HERE"""  

model = Sequential()
model.add(TimeDistributedConvolution2D(8, 4, 4, border_mode='valid', input_shape=(timeSteps,1,wSize,wSize)))
model.add(TimeDistributedMaxPooling2D(pool_size=(2, 2),border_mode='valid'))
model.add(Activation('relu'))
model.add(TimeDistributedConvolution2D(16, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
#model.add(TimeDistributedMaxPooling2D(pool_size=(2, 2),border_mode='valid'))
model.add(TimeDistributedConvolution2D(16, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(TimeDistributedFlatten())
model.add(Activation('relu'))
model.add(GRU(output_dim=100,return_sequences=True))
#model.add(GRU(output_dim=50,return_sequences=False))
model.add(Dropout(.2))
model.add(TimeDistributedDense(outsize))

#model.add(Dense(outsize, init='normal'))

#lr  = 0.000001
optimizer   = Adadelta()
model.compile(loss='mean_squared_error', optimizer=optimizer)

test    = np.reshape(test,(1,)+test.shape)
preds   = model.predict(test)
print preds.shape
#    model.set_weights(getWeights("../OCRfeatures/200_5_3/bestModel.pickle"))

helperFuncs.saveModel(model,folder+"wholeModel")



