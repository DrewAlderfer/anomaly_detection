import tensorflow as tf
from tensorflow.keras.models import Model


class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = tf.keras.models.Sequential([
            layers.Input(shape=(200, 200, 1)),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=1),
            layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=1), # (200, 200, 1, 16)
            # layers.Reshape((200, 200, 16, 1)),
            # layers.MaxPool3D((4, 4, 4), padding='same', data_format='channels_last'), # output should be (50, 50, 4, 1)
            # layers.Flatten(),
            # layers.Dense(100, activation='relu')
            layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=1),
            ])
        self.decoder = Sequential([
            layers.Conv2DTranspose(8, 3, strides=2, activation='relu', padding='same'),
            # layers.Dense(10000, activation='sigmoid'),
            # layers.Reshape((50, 50, 4, 1)),
            # layers.Conv3DTranspose(1, 4, activation="sigmoid", padding='same', strides=(4, 4, 4)),
            layers.Conv2D(1, 3, strides=2, activation='sigmoid', padding='same')
            ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
