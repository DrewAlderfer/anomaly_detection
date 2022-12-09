import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.losses import Loss

# class EdgeLayer(Layer):
#     def __init__(self, )

class EdgeDetectionLayer(Layer):
  def __init__(self):
    super(EdgeDetectionLayer, self).__init__()

  def call(self, inputs):
    # Convert the input image to grayscale
    gray = tf.image.rgb_to_grayscale(inputs)

    # Blur the grayscale image using a Gaussian filter
    blurred = tf.image.gaussian_blur(gray, ksize=(5, 5), sigmaX=1.5)

    # Use the Sobel operator to find the horizontal and vertical gradients
    sobel_x = tf.image.sobel_edges(blurred, direction='x')
    sobel_y = tf.image.sobel_edges(blurred, direction='y')

    # Calculate the gradient magnitude and direction
    magnitude = tf.sqrt(tf.square(sobel_x) + tf.square(sobel_y))
    direction = tf.atan2(sobel_y, sobel_x)

    # Threshold the gradient magnitude to find the edges
    edges = tf.where(magnitude > 30, 1.0, 0.0)

    return edges
