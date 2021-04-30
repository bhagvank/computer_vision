

import re
import os
import random
import numpy as nump
import pandas as pd
import tensorflow as tflow
import matplotlib.pyplot as matplt
from tensorflow import keras as tkeras
from tensorflow.keras import layers as klayers
from tensorflow.keras.layers.experimental import preprocessing as kexp

try:
    tpu = tflow.distribute.cluster_resolver.TPUClusterResolver()
    print("Device:", tpu.master())
    tflow.config.experimental_connect_to_cluster(tpu)
    tflow.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tflow.distribute.experimental.TPUStrategy(tpu)
except:
    strategy = tflow.distribute.get_strategy()
print("Number of replicas:", strategy.num_replicas_in_sync)


AUTOTUNE = tflow.data.experimental.AUTOTUNE
BATCH_SIZE = 25 * strategy.num_replicas_in_sync
IMAGE_SIZE = [180, 180]
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]



train_images = tflow.data.TFRecordDataset(
    "gs://download.tensorflow.org/data/ChestXRay2017/train/images.tfrec"
)
train_paths = tflow.data.TFRecordDataset(
    "gs://download.tensorflow.org/data/ChestXRay2017/train/paths.tfrec"
)

ds = tflow.data.Dataset.zip((train_images, train_paths))



COUNT_NORMAL = len(
    [
        filename
        for filename in train_paths
        if "NORMAL" in filename.numpy().decode("utf-8")
    ]
)
print("Normal images count in training set: " + str(COUNT_NORMAL))

COUNT_PNEUMONIA = len(
    [
        filename
        for filename in train_paths
        if "PNEUMONIA" in filename.numpy().decode("utf-8")
    ]
)
print("Pneumonia images count in training set: " + str(COUNT_PNEUMONIA))




def get_label(file_path):
    
    parts = tflow.strings.split(file_path, "/")
    return parts[-2] == "PNEUMONIA"


def decode_img(img):
    
    img = tflow.image.decode_jpeg(img, channels=3)
    return tflow.image.resize(img, IMAGE_SIZE)


def process_path(image, path):
    label = get_label(path)
    img = decode_img(image)
    return img, label


ds = ds.map(process_path, num_parallel_calls=AUTOTUNE)



ds = ds.shuffle(10000)
train_ds = ds.take(4200)
val_ds = ds.skip(4200)



for image, label in train_ds.take(1):
    print("Image shape: ", image.numpy().shape)
    print("Label: ", label.numpy())



test_images = tflow.data.TFRecordDataset(
    "gs://download.tensorflow.org/data/ChestXRay2017/test/images.tfrec"
)
test_paths = tflow.data.TFRecordDataset(
    "gs://download.tensorflow.org/data/ChestXRay2017/test/paths.tfrec"
)
test_ds = tflow.data.Dataset.zip((test_images, test_paths))

test_ds = test_ds.map(process_path, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE)




def prepare_for_training(ds, cache=True):
    
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.batch(BATCH_SIZE)

    
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds




train_ds = prepare_for_training(train_ds)
val_ds = prepare_for_training(val_ds)

image_batch, label_batch = next(iter(train_ds))




def show_batch(image_batch, label_batch):
    matplt.figure(figsize=(10, 10))
    for n in range(25):
        ax = matplt.subplot(5, 5, n + 1)
        matplt.imshow(image_batch[n] / 255)
        if label_batch[n]:
            matplt.title("PNEUMONIA")
        else:
            matplt.title("NORMAL")
        matplt.axis("off")




show_batch(image_batch.numpy(), label_batch.numpy())






def conv_block(filters, inputs):
    x = klayers.SeparableConv2D(filters, 3, activation="relu", padding="same")(inputs)
    x = klayers.SeparableConv2D(filters, 3, activation="relu", padding="same")(x)
    x = klayers.BatchNormalization()(x)
    outputs = klayers.MaxPool2D()(x)

    return outputs


def dense_block(units, dropout_rate, inputs):
    x = klayers.Dense(units, activation="relu")(inputs)
    x = klayers.BatchNormalization()(x)
    outputs = klayers.Dropout(dropout_rate)(x)

    return outputs





def build_model():
    inputs = tkeras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    x = kexp.Rescaling(1.0 / 255)(inputs)
    x = klayers.Conv2D(16, 3, activation="relu", padding="same")(x)
    x = klayers.Conv2D(16, 3, activation="relu", padding="same")(x)
    x = klayers.MaxPool2D()(x)

    x = conv_block(32, x)
    x = conv_block(64, x)

    x = conv_block(128, x)
    x = klayers.Dropout(0.2)(x)

    x = conv_block(256, x)
    x = klayers.Dropout(0.2)(x)

    x = klayers.Flatten()(x)
    x = dense_block(512, 0.7, x)
    x = dense_block(128, 0.5, x)
    x = dense_block(64, 0.3, x)

    outputs = klayers.Dense(1, activation="sigmoid")(x)

    model = tkeras.Model(inputs=inputs, outputs=outputs)
    return model




initial_bias = nump.log([COUNT_PNEUMONIA / COUNT_NORMAL])
print("Initial bias: {:.5f}".format(initial_bias[0]))

TRAIN_IMG_COUNT = COUNT_NORMAL + COUNT_PNEUMONIA
weight_for_0 = (1 / COUNT_NORMAL) * (TRAIN_IMG_COUNT) / 2.0
weight_for_1 = (1 / COUNT_PNEUMONIA) * (TRAIN_IMG_COUNT) / 2.0

class_weight = {0: weight_for_0, 1: weight_for_1}

print("Weight for class 0: {:.2f}".format(weight_for_0))
print("Weight for class 1: {:.2f}".format(weight_for_1))


checkpoint_cb = tflow.tkeras.callbacks.ModelCheckpoint("xray_model.h5", save_best_only=True)

early_stopping_cb = tflow.tkeras.callbacks.EarlyStopping(
    patience=10, restore_best_weights=True
)



initial_learning_rate = 0.015
lr_schedule = tflow.tkeras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)


with strategy.scope():
    model = build_model()

    METRICS = [
        tflow.tkeras.metrics.BinaryAccuracy(),
        tflow.tkeras.metrics.Precision(name="precision"),
        tflow.tkeras.metrics.Recall(name="recall"),
    ]
    model.compile(
        optimizer=tflow.tkeras.optimizers.Adam(learning_rate=lr_schedule),
        loss="binary_crossentropy",
        metrics=METRICS,
    )

history = model.fit(
    train_ds,
    epochs=100,
    validation_data=val_ds,
    class_weight=class_weight,
    callbacks=[checkpoint_cb, early_stopping_cb],
)



fig, ax = matplt.subplots(1, 4, figsize=(20, 3))
ax = ax.ravel()

for i, met in enumerate(["precision", "recall", "binary_accuracy", "loss"]):
    ax[i].plot(history.history[met])
    ax[i].plot(history.history["val_" + met])
    ax[i].set_title("Model {}".format(met))
    ax[i].set_xlabel("epochs")
    ax[i].set_ylabel(met)
    ax[i].legend(["train", "val"])



model.evaluate(test_ds, return_dict=True)



for image, label in test_ds.take(1):
    matplt.imshow(image[0] / 255.0)
    matplt.title(CLASS_NAMES[label[0].numpy()])

prediction = model.predict(test_ds.take(1))[0]
scores = [1 - prediction, prediction]

for score, name in zip(scores, CLASS_NAMES):
    print("This image is %.2f percent %s" % ((100 * score), name))
