## Deformation of images using Generative Adversarial Networks: a study to improve their performance on neural activity data of the worm C. elegans
#### *Roberto Ceraolo - roberto.ceraolo@epfl.ch*
#### *Roberto Minini - roberto.minini@epfl.ch*
#### *Joel Rudsberg - joel.rudsberg@epfl.ch*

#### *Machine Learning CS-433, EPFL, Switzerland* - *Project hosted by Prof. Sahand Rahi, LPBS, Institute of Physics, EPFL*

This repository contains the code to reproduce the experiments discussed in the report of the project. What follows is a brief description of the files present. 

The experiments were run on Izar, which is an Intel Xeon-Gold based cluster, available to the EPFL research community. More specifically, it was used with 1 node, 1 task, 1 gpu and 1 cpu. The node has Xeon-Gold processors running at 2.1 GHz, with 20 cores each, 2 NVIDIA V100 PCIe 32 GB GPUs (2×7TFLOPS), 192 GB of DDR4 RAM, 3.2 TB NVMe local drive.

All the experiments of the paper are done on synthetic data, generated through the scripts available in this repository. We did not upload the dataset with the images of the worm as it is sensible data from the lab, and it is not the main focus of the paper, as it is the "hard case" we are gradually trying to reach, thanks to synthetic data. 


### deformation_2D.py

This .py is the main file used to train the GANs and run the experiments. This file uses many fucntions imported from tools.py and requires the worm dataset if running experiments on worm data. This code was developed by Perrine Worringer and others from LPBS and was partially modified by the authors of this project.

### tools.py

This file contains many functions which are then imported in deformation2D.py . This code was developed by Perrine Worringer and others from LPBS.


### synthetic_images_de_novo.py

This .py was created to generate the synthetic images de novo. The code generates two ellipsoids with random shape and position. 
The code allows for the possibility of adding bezier curves (which simulate neurites) and the possibility of adding noise.

### synthetic_images_non_de_novo.py


This .py was created to generate the non de novo synthetic images. The code generates images by starting from an original image and applying different degrees of a specific transformation to it. 
The possible deformations are:
* Swirl
* Rotation
* Dilation
* Erosion

Swirl and Rotation are created using two functions from the skimage library.
To generate erosion and dilation a kernel-based method was used. Similar to convolution, the kernel slides through the image. For erosion, a pixel in the original image (either 1 or 0, where 1 is non-background) will be considered 1 only if all the pixels under the kernel are 1, otherwise it is eroded (made to zero). This means that all the pixels near the boundaries of the blobs will be discarded depending upon the size of kernel. Hence, the size of the blobs decreases. For dilation, it’s the opposite. Here, a pixel element is ‘1’ if at least one pixel under the kernel is ‘1’. So the size of the blobs increases.
