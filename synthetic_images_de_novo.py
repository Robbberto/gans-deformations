# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 16:16:59 2021

@author: robi
"""
import os
cwd = os.getcwd()
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import torch
import time

from wand.image import Image
from wand.drawing import Drawing
from wand.color import Color

import shutil


def noisy(noise_typ, image):
   """
   Function to add noise to the generated images.
   
   Input:
   noise_typ: type of noise to be added: Gauss, Speckle or Poisson
   image: image onto which noise will be added

   Output:
   image with noise
   """
   if noise_typ == "gauss":
      row,col,ch= image.shape
      mean = 0
      var = 1  # 0.1  ## we need high variance to create visible effect
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return noisy
  
   elif noise_typ =="speckle":
      row,col,ch = image.shape
      gauss = np.random.randn(row,col,ch)
      gauss = gauss.reshape(row,col,ch)        
      noisy = image + image * gauss
      return noisy
     
  
   elif noise_typ == "poisson":
      vals = len(np.unique(image))
      vals = 2 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(image * vals) / float(vals)
      return noisy
    
   else:
      return image





def generate(num_images, noise, bezier, path_to_save):
   """
   Function to run the script. 
   """

   for i in range(num_images):
       #Creation of first ellipsoid
       center_x = np.random.randint(20, 80) #Implementing stochasticity to the placement of the ellipsoid
       center_y = np.random.randint(20, 35)
       angle = np.random.randint(0,360)
       axis_1 = np.random.randint(7,10) #Implementing stochasticity to the placement of the ellipsoid
       axis_2 = np.random.randint(7,10)
       image = np.ones((112,112,1)) * 0.07
       image = image.reshape((112,112,1))
       cv2.ellipse(image, (center_x, center_y), angle = angle, startAngle = 0, endAngle =360, axes = (axis_1,axis_2), color = (65,181,84), thickness = -1) #col [255,1,1]
       
       
       #Creation of second ellipsoid
       center_x_2 = np.random.randint(20, 80)
       center_y_2 = np.random.randint(65, 80)
       angle = np.random.randint(0,360)
       axis_1 += np.random.randint(-2,2)
       axis_2 += np.random.randint(-2,2)
       cv2.ellipse(image, (center_x_2, center_y_2), angle = angle, startAngle = 0, endAngle =360, axes = (axis_1,axis_2), color = (65,181,84), thickness = -1) #color [30,181,84]


       if bezier == True:
         add_bezier(image, noise, path_to_save, center_x, center_y, center_x_2, center_y_2, i)
       else:
          image = noisy(noise, image)     
          cv2.imwrite(os.path.join(path_to_save, f"{i}.png"), image)

   print("Image generation done")
   


def add_bezier(image, noise, path_to_save, center_x, center_y, center_x_2, center_y_2, ind):
    """
    Function to add bezier curves to the image in order to simulate neurites 
    """
    with Drawing() as draw:
      
        # set stroke color
        draw.stroke_color = Color('white')
      
        # set width for stroke
        draw.stroke_width = 1
      

        points = [(center_x, center_y)] # Starting point
        for i in range(8):
           points.append((np.random.randint(20,80),np.random.randint(20,80))) #control points
           
        points.append((center_x_2, center_y_2))# End point 


        # fill white color in arc
        draw.fill_color = Color('none')
      
        # draw bezier curve using bezier function
        draw.bezier(points) 
      

        points_2 = [(center_x, center_y)] # Starting point
        for i in range(8):
           points_2.append((np.random.randint(0,112),np.random.randint(0,112))) #control points
        points_2.append((center_x_2, center_y_2))# End point 
        
        
        draw.bezier(points)
        
        image = noisy(noise, image)
        
        with Image.from_array(image) as img:
      
            # draw shape on image using draw() function
            draw.draw(img)
            img.save(filename = os.path.join(path_to_save, f"{ind}.png"))
            


def run(noise = "no", num_images = 1, bezier = False):
   #noise = "no" #options: speckle, gauss, poisson, anything else for no noise
   #num_images = 1 #number of images to be created, default: 500
   #bezier = False #Whether to add Bezier curves


   folder_name = f"sample_{num_images}_imgs_{noise}_noise"
   path = os.path.join(cwd, "synthetic_images_de_novo")

   if not os.path.exists(path):
    os.makedirs(path)
   else:
    shutil.rmtree(path)# Removes all the subdirectories
    os.makedirs(path)


   path_to_save = os.path.join(path, folder_name)
   os.mkdir(path_to_save)
   generate(num_images, noise, bezier, path_to_save)


run()






















