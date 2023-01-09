# Ear-Recognition-Pipeline

In this work, we implement a ear recognition pipeline. Our
pipeline includes training a custom DeepLabV3 segmentation
model to isolate ears in images, extracting both Local Binary
Patterns (LBPs) and ResNet50 (pre-trained on ImageNet) fea-
tures from the segmented ears, and implementing a matching
stage to identify individuals based on extracted features. The
performance of our system is evaluated at each stage of the
pipeline, including the segmentation, recognition, and overall
pipeline stage. For further detail please read [Biometric Pipeline](./biometric_pipeline.pdf)


## Project structure

### [Segmentation](./segmentation.ipynb) 

Notebook trains and evaluates a custom  DeepLabV3 segmentation model, the model is fine-tuned for the task of ear segmentation.

### [Feature extraction](./feature_extraction.ipynb)

Notebook extracts two types of feature vectors. We extract Local Binary Patterns (LBP) and ResNet50 (pretrained on ImageNet) features.

### [Matching](./matching.ipynb)

Notebook implements and evaluates the recognition pipeline.
