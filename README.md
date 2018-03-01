# Deep Gray Matter (DGM) Segmentation using 3D Convolutional Neural Network: application to QSM

This work is based on:
* Jose Dolz, Christian Desrosiers, Ismail Ben Ayed, *3D fully convolutional networks for subcortical segmentation in MRI: A large-scale study*, In NeuroImage, 2017
* joseabernal's solution for iSeg2017. [Github](https://github.com/joseabernal/iSeg2017-nic_vicorob)

## Current outcome

Accepted by ISMRM Workshop on Machine Learning 2018.

Some preliminary reports can be found at Medium ([Part 1](https://medium.com/@zheliu/deep-gray-matter-dgm-segmentation-using-neural-network-application-to-qsm-a0183cb3e3ae)) ([Part 2](https://medium.com/@zheliu/deep-gray-matter-dgm-segmentation-using-3d-convolutional-neural-network-application-to-qsm-part-83c247416389))

## Highlight

* Update 2018-02-04:

Larger kernel size (7, 7, 3), add Batch Normalization and auxiliary feature input of spatial coordinates information.

## How to use it
1. Put QSM images in **datasets/QSM/**
2. Put spatial coordinates maps in **datasets/X/**, **datasets/Y/**, **datasets/Z/** 
3. Put segmented ROI labels in **datasets/label/**
4. Run **segDGM_3DCNN.ipynb**
