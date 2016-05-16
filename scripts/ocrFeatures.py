# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 14:17:45 2015

@author: frickjm


Generates feature vectors from the output of fromSDFs.py
"""
import cPickle
import numpy as np
from operator import itemgetter
from scipy.stats import norm

MAX_SSSR_FEATURE_SIZE = 7   #Only include features for SSSRs of size < X
MAX_ATOM_COUNT_STDDEV = 2   #Only include molecules w/ (atom count) < (mean) + MAX_ATOM*(std_dev)
FEATURE_FREQ_CUTOFF   = 100 #Only include features that occur more than N times in the data

#load the dictionary pickled by 'fromSDFs.py'
with open("../data/cidsFEATURES.pickle",'rb') as f:  
    features    = cPickle.load(f)
  
atomCounts  = []    #size of the molecule in atoms
SSSRnums    = []  
pairskeys   = set()
bondskeys   = set()
atomskeys   = set()
SSSRkeys    = set()
#get the set of values each feature can take on    
for k,v in features.iteritems():
    pairs   = v[0]
    bonds   = v[1]
    atoms   = v[2]
    SSSR    = v[3]
    SSSRnum = v[4]

    
    pairskeys   = set(pairskeys)|set(pairs.keys())
    bondskeys   = set(bondskeys)|set(bonds.keys())
    atomskeys   = set(atomskeys)|set(atoms.keys())
    SSSRkeys    = set(SSSRkeys)|set(SSSR.keys())
    atomCounts.append(np.sum(atoms.values()))
    
print "Mean of atomcount", np.mean(atomCounts)
print "STD of atomcount", np.std(atomCounts)
print len(atomCounts)

    
pairskeys   = list(pairskeys)
pairsize    = len(pairskeys)

bondskeys   = list(bondskeys)
bondsize    = len(bondskeys)

atomskeys   = list(atomskeys)
atomsize    = len(atomskeys)

SSSRkeys    = list(SSSRkeys)   
SSSRkeys    = [k for k in SSSRkeys if int(k.replace('SSSR_','')) <= MAX_SSSR_FEATURE_SIZE] 
SSSRsize    = len(SSSRkeys)

pairvec     = np.zeros((1,pairsize))
atomvec     = np.zeros((1,atomsize))
bondvec     = np.zeros((1,bondsize))
SSSRvec     = np.zeros((1,SSSRsize))

print pairskeys
print bondskeys
print atomskeys
print SSSRkeys

atomCountThreshold  = np.mean(atomCounts)+MAX_ATOM_COUNT_STDDEV*np.std(atomCounts)
vectors     = {}
for k,v in features.iteritems():
    pairvec     = np.zeros((1,pairsize))
    atomvec     = np.zeros((1,atomsize))
    bondvec     = np.zeros((1,bondsize))
    SSSRvec     = np.zeros((1,SSSRsize))
    pairs   = v[0]
    bonds   = v[1]
    atoms   = v[2]
    SSSR    = v[3]
    SSSRnum = v[4]
    
    molAtomCount   = np.sum(atoms.values())
    if molAtomCount < atomCountThreshold:
      
        
        for k2, v2 in pairs.iteritems():
            ind     = pairskeys.index(k2)
            pairvec[0,ind]  = v2
            
        for k2, v2 in bonds.iteritems():
            ind     = bondskeys.index(k2)
            bondvec[0,ind]  = v2
            
        for k2, v2 in atoms.iteritems():
            ind     = atomskeys.index(k2)
            atomvec[0,ind]  = v2
            
        for k2, v2 in SSSR.iteritems():
            k3  = int(k2.replace('SSSR_',''))
            if k3 <= MAX_SSSR_FEATURE_SIZE:
                ind     = SSSRkeys.index(k2)
                SSSRvec[0,ind]  = v2
            
        wholevec    = np.append(pairvec,bondvec)
        #wholevec    = np.append(wholevec,bondvec)
        wholevec    = np.append(wholevec,atomvec)
        wholevec    = np.append(wholevec,SSSRvec)
        wholevec    = np.append(wholevec,SSSRnum)
        
        vectors[k]  = wholevec

#featureVec will contain the names of the columns
featureVec  = [p for p in pairskeys]
[featureVec.append(bond) for bond in bondskeys]
[featureVec.append(atom) for atom in atomskeys]
[featureVec.append(SSSR) for SSSR in SSSRkeys]
featureVec.append("SSSRnum")

#see how many times each feature occurs in the dataset - sort by this number
sums    = np.sum(vectors.values(),axis=0)
sumDict = [ [featureVec[i],sums[i]] for i in range(0,len(featureVec))]
feature2= sorted(sumDict,key=itemgetter(1),reverse=True)
print feature2
#
#for k,v in vectors.iteritems():
#    pass    
print len(vectors.keys())

truncFeatures   = [feat[0] for feat in feature2 if feat[1] > FEATURE_FREQ_CUTOFF]
featuresOut     = {}
for k,v in vectors.iteritems():
    newvector = [v[i] for i in range(0,len(v)) if featureVec[i] in truncFeatures]
    featuresOut[k] = newvector

finalFeatures   =  [featureVec[i] for i in range(0,len(featureVec)) if featureVec[i] in truncFeatures]
print "*"*80 
print finalFeatures
print featuresOut[featuresOut.keys()[0]]
print featuresOut[featuresOut.keys()[1]]





means 	= np.mean(featuresOut.values(),axis=0)
stds 	= np.mean(featuresOut.values(),axis=0)

#dump the means and stds of each feature
with open("../data/cidsMeansStds.pickle",'wb') as f:
    cp  = cPickle.Pickler(f)
    cp.dump([means, stds])

#dump the features as a dictionary {cid : feature_vec}
with open("../data/cidsFeatureVectors.pickle",'wb') as f:
    cp  = cPickle.Pickler(f)
    cp.dump(featuresOut)


#dump the feature scaled to be z-scores
featuresScaled = {}
for k,v in featuresOut.iteritems():
    v		   = np.subtract(v,means)
    featuresScaled[k] = np.divide(v,stds) 

with open("../data/cidsFeaturesScaled.pickle",'wb') as f:
    cp = cPickle.Pickler(f)
    cp.dump(featuresScaled)


    

#this is a transform of the features calculated represented by the CDF of their norm (to bound error appropriately)    
featuresCDF     = {}
for k,v in featuresScaled.iteritems():
    v		   = norm.cdf(v)
    featuresCDF[k] = v
 
with open("../data/cidsFeaturesCDF.pickle",'wb') as f:
    cp  = cPickle.Pickler(f)
    cp.dump(featuresCDF)
   
   
justRings   = {}    
for k,v in featuresOut.iteritems():
    justRings[k]    = [v[33],v[36],v[38]]   
with open("../data/justRings.pickle",'wb') as f:
    cp  = cPickle.Pickler(f)
    cp.dump(justRings)   
    

#this is a subset of atoms of interest - may differ if using fewer/more molecules
basicAtoms   = {}    
for k,v in featuresOut.iteritems():
    #basicAtoms[k]    = v[19:33]
    basicAtoms[k]   = [v[20],v[23],v[24],v[26],v[27],v[28],v[30]]

with open("../data/basicAtoms.pickle",'wb') as f:
    cp  = cPickle.Pickler(f)
    cp.dump(basicAtoms)

#this is just the counts of bonds of each type  
justBonds   = {}    
for k,v in featuresOut.iteritems():
    #basicAtoms[k]    = v[19:33]
    justBonds[k]   = v[16:19]
    #print basicAtoms[k]
with open("../data/justBonds.pickle",'wb') as f:
    cp  = cPickle.Pickler(f)
    cp.dump(justBonds)

#this is the name of each feature in cidsFeatures
with open("../data/cidsFeatureKeys.txt",'wb') as f:
    f.write(','.join(finalFeatures))
