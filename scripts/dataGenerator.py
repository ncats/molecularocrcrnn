# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 14:57:12 2015

@author: frickjm
"""



"""create a service that keeps a folder full of randomly generated images"""
from os import listdir
from os.path import isdir
from os import mkdir
from os.path import isfile
from random import shuffle
import sys
import time
import helperFuncs


#handle the input and output the options for this run
size, indir, maximum, modifyAntiAlias, resize, targetType = helperFuncs.dataGenArgs(sys.argv[1:])
targets, labels     = helperFuncs.getTargets(targetType)

#wait until a model has been created and the folder exists
while not isfile(indir+"traindata.csv"):
    time.sleep(0.1)
    print "I'm sleeping", isdir(indir), indir, "                     \r",

if not isdir(indir+"tempTrain/"):
    mkdir(indir+"tempTrain/")
    mkdir(indir+"tempTest/")
       

print "reading Train/Test files"   

#read the train and test files
train   = [x for x in file(indir+"traindata.csv").read().split("\n") if x.replace('.sdf','') in targets]    
test    = [x for x in file(indir+"testdata.csv").read().split("\n") if x.replace('.sdf','') in targets]    

#indefinitely create new images using the renderer, waiting until they're consumed
while True:
    ld  = listdir(indir+"tempTrain/")

    if len(ld) > maximum:
        time.sleep(1)
        print "sleeping because Train folder full      \r",
    else:
        shuffle(train)
        helperFuncs.makeSDFtrain(train,indir)
        parameters  = helperFuncs.getParameters(indir,size=size,modifyAntiAlias=modifyAntiAlias,modifySize=resize)
        helperFuncs.callToRenderer(parameters,indir,indir+"tempTrain")
        
    ld2 = listdir(indir+"tempTest/")
    #ld2     = listdir("temp/")
    if len(ld2) > maximum/10:
        time.sleep(1)
	print "sleeping because Test folder full       \r",
    else:
	print len(ld2)
        shuffle(test)
        helperFuncs.makeSDFtest(test,indir)
        parameters  = helperFuncs.getParameters(indir,size=size,modifyAntiAlias=modifyAntiAlias,modifySize=resize)
        helperFuncs.callToRenderer(parameters,indir,indir+"tempTest")
        
    
