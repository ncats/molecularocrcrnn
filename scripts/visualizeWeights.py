# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 16:42:29 2015

@author: frickjm
"""

import numpy as np
from os import mkdir
from os.path import isdir
import scipy.misc as mi
from helperFuncs import loadModel,loadData
import sys
import theano

model   = loadModel(sys.argv[1]+"wholeModel")
testImages  = loadData(sys.argv[1]+"tempTestNP/Xtest",'h5')
testImages  = np.reshape(testImages[2],(1,testImages[0].shape[0],testImages[0].shape[1],testImages[0].shape[2]))
mi.imsave("../evaluation/visualize/BEFORECONVOLUTION.jpg",testImages[0][0][:,:])

for num,layer in enumerate(model.layers):
    
    outFromLayerTwo = theano.function([model.get_input(train=False)],layer.get_output(train=False),allow_input_downcast=True)
    afterOneConvolution     = outFromLayerTwo(testImages)
    
    #print afterOneConvolution[0][0]
    if not isdir("../evaluation/visualize/"+str(num)+"/"):
        mkdir("../evaluation/visualize/"+str(num)+"/")
    try:
        print num, len(afterOneConvolution), len(afterOneConvolution[0]), len(afterOneConvolution[0][0])
        for filt in range(0,len(afterOneConvolution[0])):    
            mi.imsave("../evaluation/visualize/"+str(num)+"/"+str(filt)+".jpg",afterOneConvolution[0][filt])
    except TypeError:
        if len(afterOneConvolution[0] < 100):
            print "final output: ", afterOneConvolution

