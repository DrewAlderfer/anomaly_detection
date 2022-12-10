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
edge_obj.display_edges(cmap='gray')

# %%
screw_edges = SobelEdges("./data/screw/test/manipulated_front/011.png", blur=0, resize=(200, 200))
fig, ax = screw_edges.display_edges(cmap='gray')
ax.axis('off')

# %%
