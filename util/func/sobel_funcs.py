import os
from typing import BinaryIO, Tuple, Union
from PIL import Image, ImageFilter
import inspect

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import tensorflow as tf
from tensorflow import convert_to_tensor as ctt
from tensorflow.image import sobel_edges

class SobelEdges:
    """
    SobelEdges class is a set of tools to make it faster (in terms of my time) to open image
    files and process them with the sobel edges layer from tensor flow.
    """
    def __init__(self, imgfile:Union[str, BinaryIO], blur:int=5, resize:Tuple[int, int]=tuple()):
        """
        __init__ for SobelEdges:
            Parameters:
                imgfile (path): Pass the path to the image file you want to process.

                blur (int):     Pass the radius you want the GaussianBlur filter to use.

                resize (tuple): Pass a tuple of (width, height) if you want to resize the object
                                before processing it.
            Returns:
                None, initializes an instance of this SobelEdges object.
        """
        self.file_path = imgfile
        self.image = Image.open(self.file_path)
        if resize:
            self.image = self.image.resize(resize)
        self.shape = np.array(self.image).shape
        if len(self.shape) > 2:
            self.width, self.height, self.channels = self.shape
        else:
            self.width, self.height = self.shape
            self.channels = 1
        self.blur_amount = blur
        self.x_g, self.y_g = self._process_edges()
        self.angle_map = np.arctan2(self.y_g, self.x_g)

    def _process_edges(self):
        blurred = self.image.filter(ImageFilter.GaussianBlur(self.blur_amount))

        if self.channels == 3:
            blurred = blurred.convert("I")
            self.channels = 1
        blurred_arr = np.array(blurred, dtype='float')
        blurred_tensor = tf.reshape(ctt(blurred_arr, dtype=tf.float32), [1,
                                                                         self.width,
                                                                         self.height,
                                                                         self.channels])
        edge_tensor = sobel_edges(blurred_tensor)
        x_g = np.asarray(edge_tensor[0, :, :, :, 0])
        y_g = np.asarray(edge_tensor[0, :, :, :, 1])

        return (x_g, y_g)
    
    def hsv_edges(self):
        """
        This method combines the angle_map and bw_edges attributes into a hsv map where the sin of 
        the gradient is used as the hue and the magnitude of the gradient is the value attributes 
        of an hsv image tensor. That tensor is then converted into an rgb image for display.

        returns tf.tensor with shape (width, height, 3)
        """
        hue = (self.angle_map + np.pi) / (np.pi * 2)                            # angle_map is in radians from - to  this remaps it to 0 to 1
        saturation = np.ones((self.width, self.height, 1), dtype='float')
        value = self.bw_edges()
        value = (value - value.min()) / value.max()

        return np.concatenate((hue, saturation, value), axis=2) 
        
    def bw_edges(self):
        """
        Returns a grayscale representation of the sobel edges.
        """
        return np.sqrt(self.x_g**2 + self.y_g**2)

