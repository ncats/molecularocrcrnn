# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 13:58:44 2015

@author: test
"""


import helperFuncs 

#import matplotlib.pyplot as plt
import skimage.io as io
#from skimage.transform import resize 
import numpy as np

from os import listdir
#from os.path import isdir
#from os import mkdir
from os.path import isfile
from random import shuffle
import cPickle
import h5py
import sys
import subprocess
import time

from sklearn.metrics import mean_squared_error

#from keras.preprocessing.image import ImageDataGenerator
#from keras.models import Sequential
#from keras.layers.core import Dense, Dropout, Activation, Flatten
#from keras.layers.convolutional import Convolution2D, MaxPooling2D
#from keras.optimizers import SGD, Adadelta, Adagrad


sys.setrecursionlimit(10000)
np.random.seed(0)

def dumpWeights(model):     
    layercount  = 0
    for layer in model.layers:
        try:
            weights     = model.layers[layercount].get_weights()[0]
            size        = len(weights)
            if size < 100:
                with open(folder+"layer"+str(layercount)+".pickle",'wb') as f:
                    cp = cPickle.Pickler(f)
                    cp.dump(weights)
            else:
                pass
                
        except IndexError:
            pass
        layercount  +=1


def testAverages(direct,OCRfeatures):
    means = np.mean(OCRfeatures.values(),axis=0)    
    s   = len(means)
    ld  = listdir(direct)
    shuffle(ld)
    num     = 20000
    preds   = np.zeros((num,s),dtype=np.float)
    y       = np.zeros((num,s),dtype=np.float)
    count   = 0
    for x in ld[:num]:
        CID     = x[:x.find(".png")]
        y[count,:]  = OCRfeatures[CID]
        preds[count,:] = means
        count+=1
   
    print "RMSE of guessing: ", np.sqrt(mean_squared_error(y, preds))


"""Require an argument specifying whether this is an update or a new model, parse input"""
size, run, outType     = helperFuncs.handleArgs(sys.argv)


"""Define parameters of the run"""
DUMP_WEIGHTS    = True                     #will we dump the weights of conv layers for visualization
batch_size      = 64                        #how many training examples per batch


"""Define the folder where the model will be stored based on the input arguments"""
folder          = helperFuncs.defineFolder(True,outType,size,run)
print folder
trainDirect     = folder+"tempTrain/"
trainNP         = folder+"tempTrainNP/"
testDirect      = folder+"tempTest/"
testNP          = folder+"tempTestNP/"


"""Load the train/test split information"""
trainFs, testFs     = helperFuncs.getTrainTestSplit(True,folder)
trainL  = len(trainFs)
testL   = len(testFs)


features,labels     = helperFuncs.getTargets(outType) #get the OCR vector for each CID
outsize             = helperFuncs.getOutSize(features)
print("output dimensionality: ", outsize)

#    model.set_weights(getWeights("../OCRfeatures/200_5_3/bestModel.pickle"))

"""load model"""
#with open(folder+"wholeModel.pickle",'rb') as f:
#    model     = cPickle.load(f)

model   = helperFuncs.loadModel(folder+"wholeModel")
print np.sum([np.sum(a) for a in model.get_weights()])


""" TRAINING """
superEpochs     = 100000
RMSE            = 1000000
oldRMSE         = 1000000
for sup in range(0,superEpochs):
     
    oldRMSE     = min(oldRMSE,RMSE)
    print "*"*80
    print "TRUE EPOCH ", sup
    print "*"*80    

    count   = 0
    added   = 0

    #Wait for the other processes to dump a pickle file
    while not isfile(trainNP+"Xtrain.h5"):   
        print "sleeping because Train folder empty             \r",
        time.sleep(1.)
    print ""

    
    #Load the training data   
    print "Loading np  training arrays"
    loadedUp    = False
    while not loadedUp:
        try:
           
            trainImages  = helperFuncs.loadData(trainNP+"Xtrain",'h5')
            trainTargets = helperFuncs.loadData(trainNP+"ytrain",'h5')            
            
            loadedUp    = True
        except Exception as e:
            err     = e
            print err, "                              \r",
            time.sleep(2)
    print ""

    subprocess.call("rm "+trainNP+"Xtrain.h5",shell=True)
    subprocess.call("rm "+trainNP+"ytrain.h5",shell=True)

    #train the model on it
    print trainImages.shape
    model.fit(trainImages, trainTargets, batch_size=batch_size, nb_epoch=1)

   
    del trainImages, trainTargets


    """TESTING"""
    while not isfile(testNP+"Xtest.h5"):
        print "sleeping because Test folder empty             \r",
        time.sleep(1.)
    print ""

    print "Loading np test arrays" 

    loadedUp    = False
    while not loadedUp:       
        try:
 
            testImages  = helperFuncs.loadData(testNP+"Xtest",'h5')
            testTargets = helperFuncs.loadData(testNP+"ytest",'h5')

            loadedUp = True
        except Exception as e:
            err     = e
            print err, "                              \r",
            time.sleep(2)
            
    print ""

    subprocess.call("rm "+testNP+"Xtest.h5",shell=True)
    subprocess.call("rm "+testNP+"ytest.h5",shell=True)

    preds   = model.predict(testImages)
    RMSE    = np.sqrt(mean_squared_error(testTargets, preds))         
    print "RMSE of epoch: ", RMSE
    testMean= np.ones(testTargets.shape)*np.mean(testTargets,axis=0)
    RMSE2   = np.sqrt(mean_squared_error(testTargets, testMean))
    print "RMSE of guessing the mean: ", RMSE2
    #print testMean

    del testImages, testTargets    


    """Dumping the model"""
    
    if oldRMSE > RMSE:
        if DUMP_WEIGHTS:
            dumpWeights(model)

#        with open(folder+"bestModel.pickle", 'wb') as f:
#            cp     = cPickle.Pickler(f)
#            cp.dump(model)    
        helperFuncs.saveModel(model,folder+"bestModel")    

    else:
#        with open(folder+"wholeModel.pickle", 'wb') as f:
#            cp     = cPickle.Pickler(f)
#            cp.dump(model) 
        helperFuncs.saveModel(model,folder+"wholeModel")



