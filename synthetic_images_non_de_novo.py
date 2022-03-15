#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 16:59:27 2021

@author: rob
"""
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision.transforms as T
from skimage.transform import rotate
from skimage.transform import swirl

from skimage import io
from skimage.viewer import ImageViewer

import os
cwd = os.getcwd()

import shutil

from math import sqrt

from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray

from itertools import product
import cv2


import time

from wand.image import Image
from wand.drawing import Drawing
from wand.color import Color








def find_blobs(img):
   """
   Function that uses methods called Laplacian of Gaussian, Difference of Gaussian and Determinant of Hessian to find blobs in an image.
   Implemented here in order to retrieve the coordinates of the centers of the blobs and apply deformations with center of deformation the
   center of the blobs.

   Input:
   img: image in which to look for blobs

   Output:
   x_s, y_s: lists of coordinates of the centers of the blobs    

   Example of usage
   x_s, y_s = find_blobs(img) where img is an image containing blobs, x_s and y_s are lists of coordinates of the centers of the blobs 
   """
   image = img
#   image = image.astype(np.uint8)
   image_gray = rgb2gray(image)

   blobs_log = blob_log(image_gray, max_sigma=30, num_sigma=10, threshold=.1)

   # Compute radii in the 3rd column.
   blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

   blobs_dog = blob_dog(image_gray, max_sigma=30, threshold=.1)
   blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

   blobs_doh = blob_doh(image_gray, max_sigma=30, threshold=.01)

   blobs_list = [blobs_log, blobs_dog, blobs_doh]
   colors = ['yellow', 'lime', 'red']
   titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
             'Determinant of Hessian']
   sequence = zip(blobs_list, colors, titles)

   fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
   ax = axes.ravel()
   x_s = []
   y_s = []
   for idx, (blobs, color, title) in enumerate(sequence):
       ax[idx].set_title(title)
       ax[idx].imshow(image)
       for ind, blob in enumerate(blobs):
         y, x, r = blob
         if idx==0:
            x_s.append(x)
            y_s.append(y)
         c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
         ax[idx].add_patch(c)
         ax[idx].set_axis_off()

   plt.tight_layout()
   plt.show()
   return x_s, y_s


        

def run():
   """
   Function that runs the script to generate the images
   """
   num_images = 10 #Number of images to be generated default: 500
   deform = "bezier" #Type of deformation
   low_val = 1 #Lower bound for the strenght of deformation (not needed for dilate)
   high_val = 360 #Upper bound for the strenght of deformation (not needed for dilate)
   noise = False #Adding Gaussian noise after the deformation
   
   if deform == "dilate" or deform=="erode" or deform=="bezier":
       folder_name = f"sample_{num_images}_imgs_{deform}"
   else:
       folder_name = f"sample_{num_images}_imgs_{deform}_{low_val}_{high_val}"
      
   if noise:
       folder_name = folder_name+"_with_noise"
       
   path = os.path.join(cwd, "synthetic_images")

   if not os.path.exists(path):
    os.makedirs(path)
   else:
    shutil.rmtree(path) #Removes all the subdirectories
    os.makedirs(path)


   path_to_save = os.path.join(path, folder_name)
   os.mkdir(path_to_save)
   
   choose_the_image = False
   if choose_the_image:
   #if you want to choose the starting image you need to specify the path
       path_base_img = "/Users/rob/Documents/EPFL/second_semester/synthetic_images/sample_500_imgs_sin_and_rotation_-1_4/test_6.png"
       img = io.imread(path_base_img)
   else:
       #center_x, center_y, center_x_2, center_y_2 = generate_base_img("line", "no", False, path)
       generate_base_img("line", "no", False, path)
       img = io.imread(os.path.join(path, "starting_image.png"))
   img = img.astype(np.uint8)
   generate(img, num_images, low_val, high_val, deform, path_to_save, noise)


  
   #path_img = os.path.join(path, "img_orig.png")
   #io.imsave(path_img, img)



   

def generate(img, num_images, low_val, high_val, deform, path_to_save, noise=False, soma_1=None, soma_2=None):
   """
   Generation of images in the Synthetic images folder.
   Inputs:
   num_images: number of images to be generated
   low_val: Lower bound for the strenght of deformation
   high_val: Upper bound for the strenght of deformation
   deform: type of deformation to use. Possible values: rotate (classical rotation), swirl (swirling in the center of the image), swirl_blobs (swirling in the centers of the blobs), dilate 

   Output: None (it saves the images in a folder)
   """
   degs = np.linspace(low_val, high_val, num=num_images)
   if soma_1 is not None:
      center_x, center_y  = soma_1
      center_x_2, center_y_2 = soma_2
   else:
      center_x, center_y, center_x_2, center_y_2 = None, None, None, None

   if deform == "rotate":
      for i in range(num_images):
         img_deformed = rotate(img, angle=degs[i], preserve_range=True)
         if noise:
            img_deformed = add_Gaussian_noise(img_deformed)
         img_deformed = img_deformed.astype(np.uint8)
         path_img = os.path.join(path_to_save, f"img_{i}.png")
         io.imsave(path_img, img_deformed)
         if i%10==0 and i!=0:
            print(f"{i} images generated")
         
   elif deform == "swirl":
      for i in range(num_images):
         img_deformed = swirl(img, strength=degs[i], preserve_range=True)#.astype(np.uint8)
         if noise:
            img_deformed = add_Gaussian_noise(img_deformed)
         img_deformed = img_deformed.astype(np.uint8)
         path_img = os.path.join(path_to_save, f"img_{i}.png")
         io.imsave(path_img, img_deformed)
         if i%10==0 and i!=0:
            print(f"{i} images generated")
         
   elif deform == "swirl_blobs":
      x_s,y_s = find_blobs(img)
      if len(x_s)>2:
         print("Attention: more than 2 blobs detected")
      for i in range(num_images):
         img_deformed = swirl(img, strength=degs[i], center = (x_s[0], y_s[0]), preserve_range=True).astype(np.uint8)
         img_deformed = swirl(img_deformed, strength=degs[i], center = (x_s[1], y_s[1]), preserve_range=True).astype(np.uint8)
         if noise:
            img_deformed = add_Gaussian_noise(img_deformed)
         path_img = os.path.join(path_to_save, f"img_{i}.png")
         io.imsave(path_img, img_deformed)
         if i%10==0 and i!=0:
            print(f"{i} images generated")
         
   elif deform == "dilate":
        max_ind = 20
        inds = np.arange(4, max_ind, dtype=int)
        kernels_dim = list(product(inds, inds))
        if len(kernels_dim)<num_images:
            print("Not sufficient values to create deformations, more than one type of kernel will be used")
        #cv2.imshow('Input', img)
        if len(kernels_dim)*2 < num_images:
            raise Exception("Not sufficient deformations with 2 kernels")
        for i in range(num_images):
            if i>= len(kernels_dim):
                #switch to another shape of kernel
                kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,kernels_dim[i-len(kernels_dim)])
            else:
                kernel = np.ones(kernels_dim[i], np.uint8)
            img_dilation = cv2.dilate(img, kernel, iterations=1)
            cv2.imwrite(os.path.join(path_to_save, f"img_{i}.png"), img_dilation)
            if i%10==0 and i!=0:
               print(f"{i} images generated")
            #cv2.imshow('Dilation', img_dilation)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            
   elif deform == "erode":
        #For erosion, to create a sufficient number of different images, we also need to 
        #use different-shaped kernels (MAX 242)
        inds = np.arange(1, 10, dtype=int)
        kernels_dim = list(product(inds, inds))
        #kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(9,9)) #CROSS-SHAPED KERNEL
        #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10)) #ELLIPTICAL KERNEL
        #cv2.imshow('Input', img)
        cutoff = len(kernels_dim) #num_images//3
        if len(kernels_dim)*3 < num_images:
            raise Exception("Not sufficient deformations with 3 kernels")
            
        for i in range(num_images):
            if i<cutoff:
                kernel = np.ones(kernels_dim[i], np.uint8)
            elif i>=cutoff and i<2*cutoff:
                kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,kernels_dim[i-cutoff]) #CROSS-SHAPED KERNEL
            else:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,kernels_dim[i-cutoff*2]) #ELLIPTICAL KERNEL
            
            img_erosion = cv2.erode(img, kernel, iterations=1)
            cv2.imwrite(os.path.join(path_to_save, f"img_{i}.png"), img_erosion)
            if i%10==0 and i!=0:
               print(f"{i} images generated")
            #cv2.imshow('Dilation', img_dilation)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
   elif deform == "bezier":
      for i in range(num_images):
         add_bezier(img, noise, path_to_save, center_x, center_y, center_x_2, center_y_2, i)
      
   else:
       raise Exception("You need a deformation")

   print("Image generation done")


def add_Gaussian_noise(image, mean=0, var=1):
   """
   Function to add Gaussian noise
   """
   row,col= image.shape
   sigma = var**0.5
   gauss = np.random.normal(mean,sigma,(row,col))
   gauss = gauss.reshape(row,col)
   noisy_img = image + gauss
   return noisy_img



def add_bezier(image, noise, path_to_save, center_x, center_y, center_x_2, center_y_2, ind):
    """
    Function to add bezier curves to the image in order to simulate neurites 
    """
    with Drawing() as draw:
      
        # set stroke color
        draw.stroke_color = Color('white')
      
        # set width for stroke
        draw.stroke_width = 10
        
        #Set number of flexing points
        num_flexing_pts = 4
      
        draw.border_color = Color('none')
        points = [(center_x, center_y)] # Starting point
        for i in range(num_flexing_pts):
           points.append((np.random.randint(20,80),np.random.randint(20,80))) #control points
           
        points.append((center_x_2, center_y_2))# End point 


        # fill white color in arc
        draw.fill_color = Color('none')
      
        # draw bezier curve using bezier function
        draw.bezier(points) 
      

        points_2 = [(center_x, center_y)] # Starting point
        for i in range(num_flexing_pts):
           points_2.append((np.random.randint(0,112),np.random.randint(0,112))) #control points
        points_2.append((center_x_2, center_y_2))# End point 
        
        
        draw.bezier(points)
        
        #image = noisy(noise, image)
        
        with Image.from_array(image) as img:
      
            # draw shape on image using draw() function
            draw.draw(img)
            img.save(filename = os.path.join(path_to_save, f"{ind}.png"))




def generate_base_img(type_, noise, bezier, path_to_save):
   """
   Function to generate the base image
   """
   if type_=="ellipsoid":
        #Creation of first ellipsoid
        center_x = np.random.randint(20, 80) #Implementing stochasticity to the placement of the ellipsoid
        center_y = np.random.randint(20, 35)
        angle = np.random.randint(0,360)
        axis_1 = np.random.randint(7,10) #Implementing stochasticity to the placement of the ellipsoid
        axis_2 = np.random.randint(7,10)
        image = np.ones((112,112,1)) * 0.07
        image = image.reshape((112,112,1))
        #set color of somas
        #color_ = (65,181,84)
        color_ = (255,255,255)
        
        cv2.ellipse(image, (center_x, center_y), angle = angle, startAngle = 0, endAngle =360, axes = (axis_1,axis_2), color = color_, thickness = -1) #col [255,1,1]
        
        
        #Creation of second ellipsoid
        center_x_2 = np.random.randint(20, 80)
        center_y_2 = np.random.randint(65, 80)
        angle = np.random.randint(0,360)
        axis_1 += np.random.randint(-2,2)
        axis_2 += np.random.randint(-2,2)
        cv2.ellipse(image, (center_x_2, center_y_2), angle = angle, startAngle = 0, endAngle =360, axes = (axis_1,axis_2), color = color_, thickness = -1) #color [30,181,84]
     
     
        #image = noisy(noise, image)     
        cv2.imwrite(os.path.join(path_to_save, "starting_image.png"), image)
        print("Base image generation done")
        return center_x, center_y, center_x_2, center_y_2
   elif type_=="line":
        start_point = (28, 56)
          
        # End coordinate, here (250, 250)
        end_point = (84, 56)
          
        # Green color in BGR
        color = (255, 255, 255)
          
        # Line thickness of 9 px
        thickness = 5
          
        image = np.zeros((112,112,1))
        image = image.reshape((112,112,1))

        # Using cv2.line() method
        # Draw a diagonal green line with thickness of 9 px
        cv2.line(image, start_point, end_point, color, thickness)
        cv2.imwrite(os.path.join(path_to_save, "starting_image.png"), 255*image)
        print("Base image generation done")
        





run()





   
   
   
