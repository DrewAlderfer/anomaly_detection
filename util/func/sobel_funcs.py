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
        self.edges = self._process_edges()

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
        x_t = np.asarray(edge_tensor[0, :, :, :, 0])
        y_t = np.asarray(edge_tensor[0, :, :, :, 1])

        return np.sqrt(x_t**2 + y_t**2)

    def display_edges(self, **kwargs):
        """
        A function to display the output of the sobel_edges layer.
        Parameters:
            **kwargs:   Will take keyword arguments related to the pyplot.subplots and pyplot.imshow
                        objects.
                        Ex:
                            figsize=(10, 4), cmap='gray'
        Returns:
            A tuple with the created (fig, ax) objects for the image plot.
        """
        subplots_args = list(inspect.signature(plt.subplots).parameters)
        subplots_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in subplots_args}
        fig, ax = plt.subplots(**subplots_dict)
        imshow_args = list(inspect.signature(plt.imshow).parameters)
        imshow_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in imshow_args}
        ax.imshow(self.edges, **imshow_dict)
        return fig, ax
