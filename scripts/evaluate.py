# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 10:41:01 2016

@author: frickjm

Evaluates a model using the data in its tempTestNP folder
"""


import helperFuncs 
import scipy.misc as mi
import numpy as np
from tabulate import tabulate
import cPickle
import sys
import time
from scipy.spatial import distance
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
from os.path import isdir
from os import mkdir
from os.path import isfile

np.random.seed(0)

"""below are functions for outputing predictions v. ground truth"""
def printFormula(p,t,cid,atomlist,means):
    print '\t',cid
    headers     = ["FEATURE","ACTUAL","PREDICTED","FLOAT","MEAN"]
    tab         = []
    for ind in range(0,len(atomlist)):
        if t[ind] > .1:
            tab.append([atomlist[ind],int(t[ind]),int(np.round(p[ind])),p[ind],means[ind]])
            #print atomlist[ind],'\t\t',int(t[ind]),'\t\t',int(np.round(p[ind])),'\t\t', p[ind],'\t',ind
        elif np.round(p[ind]) > 0:
            #print atomlist[ind],'\t\t',int(t[ind]),'\t',int(np.round(p[ind])),'\t', p[ind],'\t',ind
            tab.append([atomlist[ind],int(t[ind]),int(np.round(p[ind])),p[ind],means[ind]])
    print tabulate(tab,headers=headers)

def printFormula3(p,t,cid,atomlist,means,stds):
    print '\t',cid
    headers     = ["FEATURE","ACTUAL","PREDICTED","FLOAT","MEAN"]
    tab         = []
    for ind in range(0,len(atomlist)):
        if t[ind] > .1:
            printIt = True
            #tab.append([atomlist[ind],int(t[ind]),int(np.round(p[ind])),p[ind],means[ind]])
            
        elif np.round(p[ind]) > 0:
            printIt = True
            #tab.append([atomlist[ind],int(t[ind]),int(np.round(p[ind])),p[ind],means[ind]])
        else:
            printIt = False
            
        if printIt:
            name    = atomlist[ind]
            real    = int(t[ind])
            fl      = p[ind]*stds[ind]+means[ind]
            ro      = np.round(fl)
            tab.append([name,real,fl,ro,means[ind]])
    print tabulate(tab,headers=headers)


def printFormulaFromCDF(p,t,cid,atomlist,means,stds):
    print '\t',cid
    headers     = ["FEATURE","ACTUAL","PREDICTED","FLOAT","MEAN","REAL FLOAT??","",""]
    tab         = []
    for ind in range(0,len(atomlist)):
        if norm.ppf(t[ind])*stds[ind]+means[ind] > .1:
            printIt = True
            #tab.append([atomlist[ind],int(t[ind]),int(np.round(p[ind])),p[ind],means[ind]])
            
        elif np.round(norm.ppf(p[ind])*stds[ind]+means[ind]) > 0:
            printIt = True
            #tab.append([atomlist[ind],int(t[ind]),int(np.round(p[ind])),p[ind],means[ind]])
        else:
            printIt = False
            
        if printIt:
            name    = atomlist[ind]
            real    = np.round(norm.ppf(t[ind])*stds[ind]+means[ind])
            realFl  = norm.ppf(t[ind])*stds[ind]+means[ind]
            realFl2 = t[ind]
            realFl3 = norm.ppf(t[ind])
            fl      = norm.ppf(p[ind])*stds[ind]+means[ind]
            ro      = np.round(fl)
            tab.append([name,real,ro,fl,means[ind],realFl,realFl2,realFl3])
    print tabulate(tab,headers=headers)

def printFormula2(p,atomlist):
    headers     = ["FEATURE","PREDICTED","FLOAT"]
    tab         = []
    for ind in range(0,len(atomlist)):
        tab.append([atomlist[ind],int(np.round(p[ind])),p[ind]])
    print tabulate(tab,headers=headers)



"""Function for comparing which vec in allVec is closest to the predicted vec"""
def getRank(cid,vec,allVec):

    tosort  = []    
    for k,vec2 in allVec.iteritems():
        #cos     = distance.cosine(vec, vec2)
        euc     = distance.euclidean(vec,vec2)
        #row     = [k, cos]        
        row     = [k,euc]
        tosort.append(row)
           
    data    = np.array(tosort)    
    #sCos    = data[np.argsort(data[:,1])]   
    sEuc    = data[np.argsort(data[:,1])]
    #c   = list(sCos[:,0]).index(cid)
    c   = list(sEuc[:,0]).index(cid)
    return c, sEuc[:10]


"""Require an argument specifying whether this is an update or a new model, parse input"""
size, run, outType     = helperFuncs.handleArgs(sys.argv)


"""Define parameters of the run"""
batch_size      = 32                        #how many training examples per batch


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
outsize             = len(features[features.keys()[0]]) #this it the size of the target (# of OCRfeatures)
means,stds          = helperFuncs.getMeansStds(features)


"""load model"""
model   = helperFuncs.loadModel(folder+"wholeModel")


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
            
        with open(testNP+"testCIDs.pickle",'rb') as f:
            testCIDs       = cPickle.load(f)

        loadedUp = True
    except Exception as e:
        err     = e
        print err, "                              \r",
        time.sleep(2)
        
print ""

preds   = model.predict(testImages)
RMSE    = np.sqrt(mean_squared_error(testTargets, preds))         
print "RMSE of epoch: ", RMSE

for i in range(0,len(preds)):
    print("*"*80)
    #printFormula(preds[i],testTargets[i],testCIDs[i],labels,means)
    print testCIDs[i]
    print preds[i]
    print testTargets[i]
    for number, image in enumerate(testImages[i]):
        if not isdir("../evaluation/series/"+testCIDs[i]+"/"):
            mkdir("../evaluation/series/"+testCIDs[i]+"/")
        mi.imsave("../evaluation/series/"+testCIDs[i]+"/"+str(number)+".jpg",image[0][:,:])
    
    #rank,mostSim     = getRank(testCIDs[i],preds[i],features)
    #print "correct rank: ", rank
    #print "most similar: ", mostSim
    
    
    stop=raw_input("*"*80)


del testImages, testTargets    
