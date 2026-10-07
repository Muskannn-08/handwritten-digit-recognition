"""
model_def.py
Shared by train_model.py and finetune.py:
- the CNN architecture
- data augmentation (makes the model robust to real hand-drawn digits)
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_model():
    """CNN with BatchNorm + Dropout (reaches ~99.5% on MNIST)."""
    return keras.Sequential([
        layers.Input(shape=(28, 28, 1)),

        layers.Conv2D(32, 3, padding="same"),
        layers.BatchNormalization(), layers.Activation("relu"),
        layers.Conv2D(32, 3, padding="same"),
        layers.BatchNormalization(), layers.Activation("relu"),
        layers.MaxPooling2D(), layers.Dropout(0.25),

        layers.Conv2D(64, 3, padding="same"),
        layers.BatchNormalization(), layers.Activation("relu"),
        layers.Conv2D(64, 3, padding="same"),
        layers.BatchNormalization(), layers.Activation("relu"),
        layers.MaxPooling2D(), layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(128),
        layers.BatchNormalization(), layers.Activation("relu"),
        layers.Dropout(0.4),
        layers.Dense(10, activation="softmax"),
    ])


# Random rotation / shift / zoom (black background stays black)
_geometry = keras.Sequential([
    layers.RandomRotation(0.1, fill_mode="constant", fill_value=0.0),
    layers.RandomTranslation(0.12, 0.12, fill_mode="constant", fill_value=0.0),
    layers.RandomZoom(0.15, fill_mode="constant", fill_value=0.0),
])


def augment(x, y):
    """Applied to each training batch (not to the saved model)."""
    x = _geometry(x, training=True)

    # randomly make ~35% of the digits thicker (real pens are thicker than MNIST)
    thick = tf.nn.max_pool2d(x, ksize=2, strides=1, padding="SAME")
    use_thick = tf.random.uniform([tf.shape(x)[0], 1, 1, 1]) < 0.35
    x = tf.where(use_thick, thick, x)
    return x, y


def make_train_dataset(x, y, batch=128):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.shuffle(len(x)).batch(batch)
    ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def load_mnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    prep = lambda a: (a.astype("float32") / 255.0)[..., np.newaxis]
    return prep(x_train), y_train, prep(x_test), y_test
