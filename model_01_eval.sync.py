# %%
import os
import pprint as pp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import image_dataset_from_directory

# %%
pp.PrettyPrinter(indent=4)

# %% [markdown]
# ##### Initialize Model

# %%
conv_model = load_model("./models/conv_model_01/")

# %%
norm_layer = tf.keras.layers.Rescaling(1.0 / 255)

# %% [markdown]
# Normal Test Data Init
# -------------------

# %%
test_good = image_dataset_from_directory(
    "./data/screw/test/good/",
    labels=None,
    seed=142,
    color_mode="grayscale",
    batch_size=None,
    image_size=(200, 200),
)

# %%
test_good_norm = test_good.map(lambda x: (norm_layer(x)))
print(type(test_good_norm))

# %%
for img in test_good_norm:
    print(type(img))
    plt.imshow(img)
    plt.show()
    break

# %%
x_good_np = np.array(
    list(map(lambda x: x, test_good_norm.as_numpy_iterator())), "float32"
)
print(x_good_np.shape)

# %%
enc_x = conv_model.encoder(x_good_np).numpy()

# %%
dec_x = conv_model.decoder(enc_x).numpy()
print(dec_x.shape)

# %%
test = dec_x
plt.figure(figsize=(15, 3))
for i in range(5):
    # print(test[i].shape)
    ax = plt.subplot(1, 5, i + 1)
    plt.imshow(test[i])
    plt.axis("off")
plt.show()

# %% [markdown]
# ### Anomalous Test Data Init
# ----------------

# %%
test_anom = image_dataset_from_directory(
    "./data/screw/test/",
    labels=None,
    shuffle=False,
    color_mode="grayscale",
    batch_size=None,
    image_size=(200, 200),
)

# %%
test_anom_norm = test_anom.map(lambda x: (norm_layer(x)))
print(type(test_anom_norm))

# %%
for img in test_anom_norm:
    print(type(img))
    plt.imshow(img)
    plt.show()
    break

# %%
x_anom_np = np.array(
    list(map(lambda x: x, test_anom_norm.as_numpy_iterator())), "float32"
)
x_anom_np = x_anom_np[41:]
print(x_anom_np.shape)

# %%
enc_x_anom = conv_model.encoder(x_anom_np).numpy()

# %%
dec_x_anom = conv_model.decoder(enc_x_anom).numpy()
print(dec_x_anom.shape)

# %%
test = dec_x_anom
plt.figure(figsize=(15, 3))
for i in range(5):
    # print(test[i].shape)
    ax = plt.subplot(1, 5, i + 1)
    plt.imshow(test[i])
    plt.axis("off")
plt.show()

# %% [markdown]
# ## Calculating the Loss
# -----

# %% [markdown]
# ##### Normal Data Decoding Loss

# %%
print(dec_x.shape)

# %%
y_pred = dec_x.reshape(41, 40000)
y_true = x_good_np.reshape(41, 40000)
loss = np.mean(abs(y_true - y_pred), axis=-1)
plt.hist(loss, bins=12)
plt.xlabel("Loss Distribution for Normal Data")
plt.ylabel("No. of Samples")
plt.show()

# %% [markdown]
# ##### Anomalous Data Decoding Loss

# %%
dec_x_anom.shape

# %%
y_anom_pred = dec_x_anom.reshape(119, 40000)
y_anom_true = x_anom_np.reshape(119, 40000)
loss_anom = np.mean(abs(y_anom_true - y_anom_pred), axis=-1)

# %%
plt.hist(loss_anom, bins=12)
plt.xlabel("Loss Distribution for Anomalous Data")
plt.ylabel("No. of Samples")
plt.show()

# %%
loss_hist = np.histogram(loss, bins=20)
loss_max = loss_hist[0].max()
norm_bar_y = loss_hist[0] / loss_max
norm_bar_x = loss_hist[1][1:]

loss_anom_hist = np.histogram(loss_anom, bins=41)
anom_bar_y = loss_anom_hist[0] / loss_anom_hist[0].max()
anom_bar_x = loss_anom_hist[1][1:]

# %%
threshold_anom = np.mean(loss_anom) - np.std(loss_anom)
print("Threshold: ", threshold_anom)

# %%
norm_y, norm_x = np.histogram(loss, bins=20)
norm_y = norm_y / norm_y.max()

anom_y, anom_x = np.histogram(loss_anom, bins=20)
anom_y = anom_y / anom_y.max()

fig, ax = plt.subplots()
# plt.bar(x=norm_bar_x, height=norm_bar_y, width=np.diff(norm_bar_x)[0], alpha=0.2)
# plt.step(x=norm_bar_x, y=norm_bar_y, where='mid', alpha=.6)
ax.stairs(norm_y, norm_x, hatch=('...'), label="Normal")
ax.axvline(x=threshold_anom, c='black', linestyle='--', alpha=.8, label="Threshold")
# plt.bar(x=anom_bar_x, height=anom_bar_y, width=np.diff(anom_bar_x)[0], alpha=0.2)
ax.stairs(anom_y, anom_x, hatch='///', label="Anomalous")
# plt.step(x=anom_bar_x, y=anom_bar_y, where='mid', alpha=.6)
fig.tight_layout()
plt.xlabel("Loss Values")
plt.ylabel("Number of Values per Bin")
plt.title("Relative Distributions of Loss Values")
plt.legend()
plt.savefig("./images/anom_norm_dist.png")
plt.show()


# %%
def predict(model, data, threshold):
    encoded = model.encoder(data).numpy()
    reconstructions = model.decoder(encoded).numpy().reshape(119, 40000)
    model_loss = tf.keras.losses.mae(reconstructions, data.reshape(119, 40000))
    return tf.math.less(model_loss, threshold)


# %%
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/opt/cuda/"
preds = predict(conv_model, x_anom_np, threshold_anom)

# %%
accuracy = (119 - np.count_nonzero(preds)) / 119
print(f"Accuracy Score: {accuracy*100:.2f}")
