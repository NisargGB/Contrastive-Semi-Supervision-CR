# Contrastive Semi-Supervised Learning for Medical Image Segmentation

## This repository implements the componenets required for our submission to MICCAI'21

### Some filemaps - 
- attn_unet.py: Code for the backbone/baseline Attention unet
- data_loader.py: Contains data generator for the celiac dataset
- generate_pseudolabels.py: A utility file for generating and saving pseudolabels in case of very slow cpus that cant run on-the-fly
- inference.py: Run inference on a test set of images
- metrics.py: Contains loss functions and metrics to be monitored
- model.py: Defines encoder, decoder, composite model
- train.py: Running semi-supervised CL+CR training of the network. Change config variables at the beginning code and run it
- utils.py: Some functional utilities
- dataloadrs/data_generator.py: A comppletely modular implementation of celiac dataet generator. Easilty modifiable to other datasets

### Requirements
- python 3.6+
- Tensorflow-GPU 2.0+
- Keras (compatible with the TF backend)