# Material Notes

## Face Recognition Pipeline Clearly Explained [link](https://medium.com/backprop-labs/face-recognition-pipeline-clearly-explained-f57fc0082750)

### Face Alignment

Important phase in the modern pipeline for recognition.
Increases accuracy by 1-2 %. In a face recognition pipeline we can use the Haar-cascade detector to find and align the eyes.

### Feature Extraction (Deep Learning Approach)

Consider the following models:
1. VGGFace
2. FaceNet (introduces a triplet loss, that allows images to be encoded as featzre vectors that allow rapid similarity calculation and matching via distance mesures)


### Feature Classification
1. distance measures (euclidiean, cosine, ...)
2. SVMs (find optimal hyper-plane to classifiy the classes based on the feature vectors)
3. KNN (Use majority voting to determine the identity of the person on the image, NOTE: use PCA or LDA before using KNN to reduce the dimensionality, since the KNN has a curse of dimensionality problem)


## Implementing Face Recognition Using Deep Learning and Support Vector Machines [link](https://www.codemag.com/Article/2205081/Implementing-Face-Recognition-Using-Deep-Learning-and-Support-Vector-Machines)

### Deep Learning based recogniton

Network:
1. Conv & Pooling layers
2. Fully connected layers
3. Softmax layer over all identities

#### VGGFace

Model is was trained on a dataset of celebrities, public figures and actors. Dataset consisted of multiple images per identity. The original VGGFace uses teh VGG16 model while VGGFace2 uses the ResNet-50, we can also use the implementation which uses SENet50 (smaller network). 

Sidenote: rescale images to 224 x 224 pixels.


#### Using Transfer Learning to recognize custom faces.

Since VGGFace is trained on the task of face identification or recognition we can use its lower (convolutional) layers to extract the features and just train the classifier on top of those features.

Steps:

1. detect faces, crop faces, rescale images to 224 x 224.
2. augment the training images (i.e transform input images)
3. load pre-trained model
4. freeze all layers but the fully connected layer


### Support Vector Machine based recognition

SVM finds the boundary that separates classes by as wide a margin as possible. In this article the SVM is trained on raw pixels.

## Towards automated multiscale imaging and analysis in TEM: Glomerulus detection by fusion of CNN and LBP maps [link](https://openaccess.thecvf.com/content_ECCVW_2018/papers/11134/Wetzer_Towards_automated_multiscale_imaging_and_analysis_in_TEM_Glomerulus_detection_ECCVW_2018_paper.pdf)

### Previous work

A number of papers show that it is often beneficial to combine hand crafted and learned features. Adding hand-crafted features can be seen as a variation of transfer learning, where the network is helped by additional views of the data.

Typical approaches that combine machine learning and LBPs:
1. extract histograms of LBPs & train a SVM on top of extracted histograms.
2. use LBP histograms in combination with CNN features.

### Their approach:

Idea: Use dense LBP maps in combinataion with the raw image data as input for a CNN.
Detail: It turns out that performing numerical operations such as averaging (as done with convolution operation) on top of LBP codes is not a good idea. To solve this we have to map LBP codes into Euclidean space. 

They use Multidimensional Scaling (MDS) to map the data from an unordered set into a metric space, i.e we map LBP code to a LBP Map (of shape (H X W X 3), this is just an RGB image).

### Network:

They compare two CNN models: VGG16 and ResNet50. Both networks are trained from scratch on either the raw image data or the LBP maps. They perform fusion on raw and LBP data at three different depths of the networks. 

#### Early Fusion:

The raw image layer is staked with the three layers of 3D LBP Maps and feed into the input layer of the CNN. In the multiscale experimental setup, the raw images layer is stacked with in total 9 layeres of LBP Maps corresponding to the varying radii (R€{1,2,3}) in LBP extraction.


#### Mid Fusion:

Uses a two-stream architecture. We train two CNNs:

1. Train one CNN on normalized intesity images.
2. Train one CNN on the single scale 3-layer LBP Maps

Once networks are trained, the outputs of the second dully connected layers of both architectures are concatenated. A linear SVM is then trained on top of the feature vectors.


#### Late Fusion:

Uses a two-stream architecture. We train two CNNs:

1. Train one CNN on normalized intesity images.
2. Train one CNN on the single scale 3-layer LBP Maps

Once networks are trained, the output probabilites of the sfotmax layers of the two networks are concatenated and 1 linear SVM is trained to classifiy. 


## Understanding U-Net [link](https://towardsdatascience.com/understanding-u-net-61276b10f360)

## U-Net Paper overview [link](https://towardsdatascience.com/understanding-u-net-61276b10f360)

## U-Net Implementation in Pytorch [link](https://towardsdatascience.com/understanding-u-net-61276b10f360)