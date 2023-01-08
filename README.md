# Ear-Recognition-Pipeline

## Project structure

### [Segmentation](./segmentation.ipynb) 

Notebook trains and evaluates a custom  DeepLabV3 segmentation model, the model is fine-tuned for the task of ear segmentation.

### [Feature extraction](./feature_extraction.ipynb)

Notebook extracts two types of feature vectors. We extract Local Binary Patterns (LBP) and ResNet50 (pretrained on ImageNet) features.

### [Matching](./matching.ipynb)

Notebook implements and evaluates the recognition pipeline.