# %%
# %load_ext autoreload

# %%
import os
import typing

import joblib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import tensorflow as tf
from IPython.display import display
from PIL import Image, ImageFilter
from tensorflow.keras.layers import Layer
from tensorflow.keras.losses import Loss
from tensorflow.keras.models import Model, Sequential

from util.models import new_model
from util.func.sobel_funcs import SobelEdges

from PIL import Image, ImageFilter


# %%
# %autoreload 2
# %aimport util.models
# %aimport util.func.sobel_funcs

# %%
class SobelLayer(tf.keras.layers.Layer):
    def __init__(self, blur, activation=None):
        super().__init__()
        self.activation = activation
        self.edgefinder = sobel_edge_finder
        self.blur = blur

    # def build(self, input_shape):
    #     self.w = self.add_weight(
    #             shape=(input_shape[-1]),
    #             initializer="random_normal",
    #             trainable=True)
    #             
    #     self.b = self.add_weight(
    #             shape=(1,), initializer="zeros", trainable=True)

    def call(self, input):
        blur = self.blur
        print(input.shape)
        x = self.edgefinder(input)
        x = x.hsv_edges()
        return tf.add(input, x)


# %%
class sobel_edge_finder():
    def __init__(self, input:np.ndarray):
        self.input = input
        self.width, self.height, _ = input.shape
        self.channels = 1
        # self.blur_amount = blur
        self.x_g, self.y_g = self._process_edges()
        self.angle_map = np.arctan2(self.y_g, self.x_g)

    def _process_edges(self):
        # image = Image.fromarray(self.input)
        # blurred = image.filter(ImageFilter.GaussianBlur(self.blur_amount))
        # blurred_arr = np.array(blurred, dtype='float')
        # blurred_tensor = tf.reshape(tf.convert_to_tensor(blurred_arr, dtype=tf.float32), [1,
        #                                                                  self.width,
        #                                                                  self.height,
        #                                                                  self.channels])
        edge_tensor = tf.image.sobel_edges(self.input)
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


# %%
image = "./data/screw/train/good/002.png"
input = Image.open(image)
input = np.asarray(input.resize((256, 256)))
print(type(input))
edges = SobelEdges(image, blur=4, resize=(256, 256))
hsv_edges = edges.hsv_edges()
print(type(hsv_edges), hsv_edges.shape)

# %%
img_size = (256, 256)
mut_img_size = img_size + (3,)
print(img_size)

# %%
input = (input - input.min()) / input.max()
input = np.reshape(input, (*img_size, 1))
layer_input = np.concatenate((input, hsv_edges), axis=2)
print(input.shape, layer_input.shape)


# %%
def conv_layers(img_size):
    inputs = tf.keras.Input(shape=(img_size + (4,)))
    x = tf.keras.layers.Conv2D(8, 3, activation="relu", padding="same", strides=2)(inputs)
    x = tf.keras.layers.Conv2D(8, 3, padding="same", strides=2)(x)
    x = tf.keras.layers.Conv2D(48, 3, activation="relu", padding="same", strides=2)(x)
    x = tf.keras.layers.Conv2D(48, 3, padding="same", strides=2)(x)
    residual = x
    x = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same", strides=2)(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", strides=2)(x)

    x = tf.keras.layers.Conv2DTranspose(96, 3, padding="same", strides=2)(x)
    x = tf.keras.layers.Conv2DTranspose(48, 3, activation="relu", padding="same", strides=2)(x)
    x = tf.add(x, residual)
    x = tf.keras.layers.Conv2DTranspose(24, 3, padding="same", strides=4)(x)
    x = tf.keras.layers.Conv2DTranspose(16, 3, activation="relu", padding="same", strides=4)(x)

    outputs = tf.keras.layers.Conv2D(4, 3, activation="softmax", padding="same")(x)

    model = tf.keras.Model(inputs, outputs)
    return model

model = conv_layers(img_size=img_size)
model.summary()

# %%
inputs = np.reshape(layer_input, (1, *img_size, 4))
model.compile(optimizer='rmsprop', loss='mse')

# %%
# os.environ["XLA_FLAGS"]="--xla_gpu_cuda_data_dir=/opt/cuda"
os.environ["XLA_FLAGS"]="--xla_gpu_cuda_data_dir=/home/drew/conda/envs/tf_env/"
history = model.fit(inputs, inputs,
                    epochs=1,
                    validation_data=(inputs, inputs))

# %%
print(x[0,:,:,1].shape)
fig = plt.Figure()
ax = plt.subplot()
ax.imshow(mcolors.hsv_to_rgb(hsv_edges), cmap='gray')
plt.show()
