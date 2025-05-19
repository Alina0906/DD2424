<h1 align="center"> Project for DD2424: Transfer Learning </h1>

<p align="center">
    <img src="imgs/CMU-NV-logo-crop-png.png" height=50"> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
</p>

## TODO
- [x] Release basic ViT code
- [ ] Implement Lora
- [ ] Further Improvement for ViT baseline


# Dataset

The Oxford-IIIT Pet Dataset [Dataset]([https://www.robots.ox.ac.uk/~vgg/data/pets/])  a 37 category pet dataset with roughly 200 images for each class. \
The images have a large variations in scale, pose and lighting. All images have an associated ground truth annotation of breed, head ROI, and pixel level trimap segmentation. 

## Basic Classification

For binary classification, please use the command below:

```bash
python train.py --task species
```

For multi-class classification:
```bash
python train.py --task class
```


## Lora Configuration

If you hope to add Lora to the training process:

```bash
python train.py --task class --use_lora
```


