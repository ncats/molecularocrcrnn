This repository has been archived and is no longer maintained.The code is provided for historical reference and may contain unpatched or unknown vulnerabilities. It should not be used in production systems.

##MolecularOCRcrnn
Molecular OCR using convolutional recurrent neural networks

#Built on
This workflow is built on keras (using Theano). It also makes use of scikit-image (skimage) and the scientific python stack. Features are calculated based on SD files using RDkit.

#Acquiring data
As input data, place a dump of SD files (named "unique_id_XXXX.sdf") in data/SDF/. 

Compute features using `python fromSDFs.py`

Process features using `python ocrfeatures.py`