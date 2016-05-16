# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 13:58:44 2015

@author: test
"""


import helperFuncs 
import skimage.io as io
import numpy as np
from os import listdir
from os.path import isfile
from random import shuffle
import cPickle
import h5py
import sys
import subprocess
import time
from sklearn.metrics import mean_squared_error

sys.setrecursionlimit(10000)
np.random.seed(0)

def dumpWeights(model):     
    """Pickles the weights of each layer in the network and dumps them in 
     the input folder"""
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


"""Require an argument specifying whether this is an update or a new model, parse input

inputs:
    -
"""
size, run, outType     = helperFuncs.handleArgs(sys.argv)


"""Define parameters of the run"""
DUMP_WEIGHTS    = False     #will we dump the weights of conv layers for visualization
batch_size      = 64        #how many training examples per batch


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

"""if we want to grab weights from a model with the same architecture"""
#    model.set_weights(getWeights("../OCRfeatures/200_5_3/bestModel.pickle"))

"""load model"""
model   = helperFuncs.loadModel(folder+"wholeModel")

""" TRAINING """
iterations		= 100000 	#number of iterations, not really epochs
RMSE            	= 1000000	#arbitrarily large number for starting RMSE
oldRMSE         	= 1000000	#arbitrarily large number for starting RMSE
for sup in range(0,superEpochs):
    print "*"*80
    print "EPOCH ", sup
    print "*"*80    
     
    #set new best RMSE achieved
    oldRMSE     = min(oldRMSE,RMSE)
    
    #Wait for the other processes to dump an hdf5 file
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

    #remove the training data
    subprocess.call("rm "+trainNP+"Xtrain.h5",shell=True)
    subprocess.call("rm "+trainNP+"ytrain.h5",shell=True)

    #train the model on it
    model.fit(trainImages, trainTargets, batch_size=batch_size, nb_epoch=1)

   
    del trainImages, trainTargets


    """TESTING"""
    
    """same loading procedure as with training data above"""
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


    """Predict on test data, calculate RMSE, print it"""
    preds   = model.predict(testImages)
    RMSE    = np.sqrt(mean_squared_error(testTargets, preds))         
    print "RMSE of epoch: ", RMSE
    testMean= np.ones(testTargets.shape)*np.mean(testTargets,axis=0)
    RMSE2   = np.sqrt(mean_squared_error(testTargets, testMean))
    print "RMSE of guessing the mean: ", RMSE2
    #print testMean

    del testImages, testTargets    


    """Dumping the model"""
    
    """If this is the lowest RMSE to date, dump the model as 'best model' 
        else dump it as wholeModel"""
    if oldRMSE > RMSE:
        if DUMP_WEIGHTS:
            dumpWeights(model)
        helperFuncs.saveModel(model,folder+"bestModel")    

    else:
        helperFuncs.saveModel(model,folder+"wholeModel")



