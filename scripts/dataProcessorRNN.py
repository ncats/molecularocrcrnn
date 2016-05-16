# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 13:35:52 2016

@author: frickjm

Converts the images from dataGenerator into NP arrays
"""
from os import listdir
from os.path import isdir
from os import mkdir
from os.path import isfile
from random import shuffle
import sys
import time
import numpy as np
import subprocess
import cPickle
import helperFuncs
import h5py
import scipy.misc as mi


wSize           = 60      #window size
stride          = wSize/3 #stride size (lower means more overlap and more computation)

"""function for taking an image and returning sequential windows from that image
   at an interval of stride"""
def im2window(image,wSize,stride):
    #create a padded image
    canvas  = np.zeros((image.shape[0]+stride-1,image.shape[1]+stride-1))
    #lay the image over the canvas
    canvas[0:image.shape[0],0:image.shape[1]] = image
    #compute max value for x, y window start positions
    ydim    = image.shape[0] - wSize + 1
    xdim    = image.shape[1] - wSize + 1
    output  = []
    xran    = np.linspace(0,xdim,(xdim/stride))
    yran    = np.linspace(0,ydim,(ydim/stride))
    
    for y in yran:
        for x in xran:
            output.append(canvas[y:y+wSize,x:x+wSize])
    
    return np.reshape(np.array(output),(len(output),1,wSize,wSize))


#Process the arguments given (see helperFuncs.py for details)
size, indir, binarize, blur, padding, targetType = helperFuncs.dataProcessorArgs(sys.argv[1:])
targets, labels     = helperFuncs.getTargets(targetType)
outsize             = helperFuncs.getOutSize(targets)


#wait until the folder exists
while not isdir(indir+"tempTrain/"):
    time.sleep(10.)
    print "I'm sleeping", isdir(indir), indir, "                     \r",

if not isdir(indir+"tempTrainNP/"):
    mkdir(indir+"tempTrainNP/")
    mkdir(indir+"tempTestNP/")
    
#define folders
trainFolder     = indir+"tempTrain/"
trainNPfolder   = indir+"tempTrainNP/"
testFolder      = indir+"tempTest/"
testNPfolder    = indir+"tempTestNP/"


#get the number of 'time' steps in the sequence
fakeImage   = np.ones((size,size))
test        = im2window(fakeImage,wSize,stride)
timeSteps   = test.shape[0]

while True:

    #wait until there are images in the training folder
    if isfile(trainNPfolder+ "Xtrain.h5"):
        time.sleep(1)
        print "sleeping because Train folder full      \r",
    else:
        ld  = listdir(trainFolder)
        shuffle(ld)
        numTrainEx      = len(listdir(indir+"tempTrain/"))
        trainImages     = np.zeros((numTrainEx,timeSteps,1,wSize,wSize),dtype=np.float)
        trainTargets    = np.zeros((numTrainEx,outsize),dtype=np.float)
        trainCIDs            = []

        added   = 0 #keep track of how many images have been added
        count   = 0
        while added < numTrainEx:
            x   = ld[count]
            if x.find(".sdf") > -1:
                try:
                    try:
                        CID     = x[:x.find(".sdf")]
                        
                        image   = helperFuncs.processImage(CID,trainFolder,binarize,blur,padding,size,noise=True)                        
                        #mi.imsave("../evaluation/"+str(CID)+".jpg",image)                        
                        subprocess.call("rm "+trainFolder+x,shell=True)
                        image   = im2window(image,wSize,stride)                        
                        trainImages[added,:,:,:,:]  = image
                        trainTargets[added]         = targets[CID] 
                        trainCIDs.append(CID)
                        
                        added+=1
                        print added
                    except (IOError,ValueError) as e:
                        print e
                except (KeyError, ValueError) as e:
                    subprocess.call("rm "+trainFolder+x,shell=True) #This means this molecule was too big
            count+=1
            if count > len(ld)-1:
                count = 0
                ld = listdir(trainFolder)
            while len(ld) == 0:
                ld = listdir(trainFolder)


        #dump the data
        helperFuncs.saveData(trainImages,trainNPfolder+"Xtrain",'h5')
        helperFuncs.saveData(trainTargets,trainNPfolder+"ytrain",'h5')

        with open(trainNPfolder+"trainCIDs.pickle",'wb') as f:
            cp  = cPickle.Pickler(f)
            cp.dump(trainCIDs)
            

    #do the same thing for the test folder
    if isfile(testNPfolder+ "Xtest.h5"):
        time.sleep(1)
        print "sleeping because test folder full      \r",
    else:
        ld  = listdir(testFolder)
        shuffle(ld)
        numTestEx      = len(listdir(indir+"tempTest/"))
        testImages     = np.zeros((numTestEx,timeSteps,1,wSize,wSize),dtype=np.float)
        testTargets    = np.zeros((numTestEx,outsize),dtype=np.float)
        testCIDs       = []

        added   = 0
        count   = 0
        while added < numTestEx:
            x   = ld[count]
            if x.find(".sdf") > -1:
                try:
                    try:
                        CID     = x[:x.find(".sdf")]
                        image   = helperFuncs.processImage(CID,testFolder,binarize,blur,padding,size,noise=True)
                        subprocess.call("rm "+testFolder+x,shell=True)
                        image   = im2window(image,wSize,stride) 
                        testImages[added,:,:,:,:]  = image
                        testTargets[added]         = targets[CID]
                        testCIDs.append(CID)
                        added+=1

                    except (IOError,ValueError) as e:
                        pass
                except (KeyError, ValueError) as e:
                    subprocess.call("rm "+testFolder+x,shell=True) #This means this molecule was too big
            count+=1
            if count > len(ld)-1:
                count = 0
                ld = listdir(testFolder)
            while len(ld) == 0:
                ld = listdir(testFolder)
        

        helperFuncs.saveData(testImages,testNPfolder+"Xtest",'h5')
        helperFuncs.saveData(testTargets,testNPfolder+"ytest",'h5')


        with open(testNPfolder+"testCIDs.pickle",'wb') as f:
            cp  = cPickle.Pickler(f)
            cp.dump(testCIDs)
        
    #repeat process for the training folder, so as to be ready to dump as soon as Xtrain is consumed
    if isfile(trainNPfolder+ "Xtrain.h5") and isfile(testNPfolder+ "Xtest.h5"):
        ld  = listdir(trainFolder)
        shuffle(ld)
        numTrainEx      = len(listdir(indir+"tempTrain/"))
        trainImages     = np.zeros((numTrainEx,timeSteps,1,wSize,wSize),dtype=np.float)
        trainTargets    = np.zeros((numTrainEx,outsize),dtype=np.float)
        trainCIDs            = []

        added   = 0
        count   = 0
        while added < numTrainEx:
            x   = ld[count]
            print x, added
            if x.find(".sdf") > -1:
                try:
                    try:
                        CID     = x[:x.find(".sdf")]
                        
                        image   = helperFuncs.processImage(CID,trainFolder,binarize,blur,padding,size,noise=True)                        
                        subprocess.call("rm "+trainFolder+x,shell=True)
                        image   = im2window(image,wSize,stride)
                        trainImages[added,:,:,:,:]  = image
                        trainTargets[added]         = targets[CID]
                        trainCIDs.append(CID)
                        added+=1

                    except (IOError,ValueError) as e:
                        print e
                except (KeyError, ValueError) as e:
                    subprocess.call("rm "+trainFolder+x,shell=True) #This means this molecule was too big
            count+=1
            if count > len(ld)-1:
                count = 0
                ld = listdir(trainFolder)
            while len(ld) == 0:
                ld = listdir(trainFolder)


        while isfile(trainNPfolder+ "Xtrain.h5"):
            time.sleep(1)
            print "sleeping until Train file used             \r",
        print ""
        print "dumping Train data"


       
        helperFuncs.saveData(trainImages,trainNPfolder+"Xtrain",'h5')
        helperFuncs.saveData(trainTargets,trainNPfolder+"ytrain",'h5')
        


        with open(trainNPfolder+"trainCIDs.pickle",'wb') as f:
            cp  = cPickle.Pickler(f)
            cp.dump(trainCIDs)

