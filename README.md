# Scene Classification
Side coursework for COMP6223 (computer vision). This repository contains 3 different runs on scene classification. Every branch
(```tiny_image```, ```bag_of_visual_words``` and ```transfer_learning```) contains a run.

## Tiny image:
Contains a simple k-nearest-neighbour classifier using the “tiny image” feature. 
The “tiny image” feature is one of the simplest possible image representations. 
Each image is cropped to a square about the centre based on the smallest dimension, and then is resized to a small,
fixed resolution (e.g 16x16). Afterwards, the "tiny image" is packed into a vector by concatenating each image row and gets 
normalised (zero mean and unit length). K-fold validation gives an estimate for the accuracy.
You can choose the optimal k-value for the classifier.

- __Run1.m__: Runs the "tiny image" classification and produces a text file with the predictions of the testing data
- __nearest_neighbor_classify.m__: KNN classification
- __training.zip__: Contains 15 scene categories each with 100 labelled samples
- __testing.zip__: Contains 2988 unlabelled testing samples

#### Instructions:
Unzip ```training.zip``` , ```testing.zip``` and keep them in the same directory as ```Run1.m```. Modify the hyperparameters 
in ```Run1.m``` and run it.

## Bag of visual words:
<p align="center">
  <img width="600" height="160" src="https://uk.mathworks.com/help/vision/ug/bagoffeatures_visualwordsoverview.png">
</p>
<p align="center">
  <img width="600" height="160" src="https://uk.mathworks.com/help/vision/ug/bagoffeatures_encodeoverview.png">
</p>

For every training sample, ```m``` fixed dimension image patches are extracted resulting in ```m x N``` features (```N```
images). These features are being clustered into ```K``` centroids building a vocabulary of ```K``` visual words. The visual words
are being flattened then as in run 1. \
After building the vocabulary, ```p``` visual words are extracted per training image, and regarding their distance 
from the vocabulary words, a word histogram is assigned to every image sample. Afterwards, the word histograms are used to
train an ensemble of 15 one-vs-all classifiers. \
Overall time depends on:
- vocabulary size
- visual word patch dimension
- number of samples to be clustered
- number of features to get extracted in order to build the word histograms\

As a method is highly consuming.
- __Run2.m__: Runs the "bag of visual words" classification and produces a text file with the predictions of the testing data
- __bag_of_words.m__: Builds a vocabulary of ```K``` words using fixed dimension image patches
- __word_mining.m__: Extracts ```p``` features per training image and builds a word histogram
- __vocabulary500.m__: Prebuilt vocabulary to avoid clustering time
- __svm_classify.m__: This function will train a linear SVM for every category (i.e. one vs all) and then use the 
learned linear classifiers to predict the category of every test image.

#### Instructions:
Unzip ```training.zip``` , ```testing.zip``` and keep them in the same directory as ```Run2.m```. Modify the hyperparameters
in ```Run2.m``` and run it.

## Transfer learning:
Run 3, makes use of a pretrained CNN to classify training samples. Specifically, each sample is fed to the CNN and a feature vector
is extracted before the softmax layer. Finally, feature vectors are used to train an enseble of 15 one-vs-all classifiers.\

- __Run3.m__: Runs the "transfer learning" classification and produces a text file with the predictions of the testing data
- __imagenet-vgg-m-1024.mat__: Contains the pretrained CNN. More pretrained networks can be downloaded from
[here](http://www.vlfeat.org/matconvnet/pretrained/).

#### Instructions:
Unzip ```training.zip``` , ```testing.zip``` and keep them in the same directory as ```Run3.m```. Modify the hyperparameters
in ```Run3.m``` and run it. Beware in feature vector dimensions and feature extraction layer if you change the pretrained net.

Example output:
<p align="center">
  <img width="700" height="600" src="https://github.com/nikostsagk/scene_classification/blob/transfer_learning/output_fig.png">
</p>

#### Requirements:
To run the repository, you need to have [VLFeat](http://www.vlfeat.org/install-matlab.html) and 
[MatConvNet](http://www.vlfeat.org/matconvnet/) installed. MatConvNet does not need GPU but make sure that it is compiled
after installing it.\
You can type:
```
>> vl_version %for vl_feat
>> vl_root    %for matconvnet
```
to check if each package is installed respectively.
