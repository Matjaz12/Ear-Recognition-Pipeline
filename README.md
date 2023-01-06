# Ear-Recognition-Pipeline

The project is structured as follows:

1. [segmentation](./segmentation.ipynb) trains a custom (fine-tuned for the task of ear segmentation) DeepLabV3 segmentation model.
2. [feature_extraction](./feature_extraction.ipynb) implement a Multiscale Local Binary Patterns feature extractor and ResNet50 (pretrained on ImageNet) feature extractor
