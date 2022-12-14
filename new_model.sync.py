# %%
# %load_ext autoreload

# %%
# %autoreload 2
# %aimport util.models
# %aimport util.func.sobel_funcs

# %%
import os

import joblib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython.display import display
from PIL import Image, ImageFilter
from tensorflow.keras.layers import Layer
from tensorflow.keras.losses import Loss
from tensorflow.keras.models import Model, Sequential

from util.models import new_model
from util.func.sobel_funcs import SobelEdges

# %%
image_bytes = tf.io.read_file("./data/screw/train/good/004.png") 
img_tensor = tf.image.decode_image(image_bytes)
img_tensor = tf.cast(img_tensor, tf.float32)
img_tensor = tf.image.resize(img_tensor, [200, 200])
width, height, channels = img_tensor.shape

# %%
img_tensor = tf.expand_dims(img_tensor, 0)
img_tensor.shape

# %%
output = tf.image.sobel_edges(img_tensor)
output_x = np.asarray(output[0, :, :, :, 0])
output_y = np.asarray(output[0, :, :, :, 1])

# %%
edge_data = np.sqrt(output_x**2 + output_y**2)

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(edge_data, cmap='gray')
ax2.imshow(output_y)
plt.show()

# %%
edges = np.sqrt(output_x**2 + output_y**2)
plt.imshow(edges, cmap='gray')
plt.show()

# %%
img_blur = Image.open('./data/screw/train/good/049.png')
img_blur = img_blur.filter(ImageFilter.GaussianBlur(radius=5))

# %%
img_blur = np.array(img_blur, dtype='float')
print(img_blur.shape)
img_blur_t = tf.reshape(tf.convert_to_tensor(img_blur, dtype=tf.float32), [1, 1024, 1024, 1])

# %%
edge_obj = SobelEdges("./data/metal_nut/train/good/002.png", blur=3)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.imshow(edge_obj.bw_edges(), cmap='gray')
ax1.axis('off')
ax2.imshow(mcolors.hsv_to_rgb(edge_obj.hsv_edges()))
plt.axis('off')
plt.savefig('./images/nut_graph.png')
plt.show()

# %%
screw_edges = SobelEdges("./data/screw/test/manipulated_front/011.png", blur=3)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.imshow(screw_edges.bw_edges(), cmap='gray')
ax1.axis('off')
ax2.imshow(mcolors.hsv_to_rgb(screw_edges.hsv_edges()))
ax2.axis('off')
plt.savefig('./images/screw_edges.png')
plt.show()


# %%
class SobelLayer(tf.keras.layers.Layer):
    def __init__(self, blur, activation=None):
        super().__init__()
        self.activation = activation
        self.edgefinder = SobelEdges
        self.blur = blur

    def build(self, input_shape):
        self.w = self.add_weight(
                shape=(input_shape[-1]),
                initializer="random_normal",
                trainable=True)
                
        self.b = self.add_weight(
                shape=(1,), initializer="zeros", trainable=True)

    def call(self, input):
        blur = self.blur + self.b
        x = self.edgefinder(input, self.blur)
        x = x.hsv_edges()
        return tf.matmul(tf.add(input, x), self.w)
        
