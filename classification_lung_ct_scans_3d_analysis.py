

import os
import zipfile
import numpy as nump
import tensorflow as tflow

import random

from scipy import ndimage

import nibabel as nib


from tensorflow import keras as tkeras
from tensorflow.keras import layers as klayers

url = "https://github.com/hasibzunair/3D-image-classification-tutorial/releases/download/v0.2/CT-0.zip"
filename = os.path.join(os.getcwd(), "CT-0.zip")
tkeras.utils.get_file(filename, url)

url = "https://github.com/hasibzunair/3D-image-classification-tutorial/releases/download/v0.2/CT-23.zip"
filename = os.path.join(os.getcwd(), "CT-23.zip")
tkeras.utils.get_file(filename, url)

os.makedirs("MosMedData")

with zipfile.ZipFile("CT-0.zip", "r") as z_fp:
    z_fp.extractall("./MosMedData/")

with zipfile.ZipFile("CT-23.zip", "r") as z_fp:
    z_fp.extractall("./MosMedData/")






def read_nifti_file(filepath):
    
    scan = nib.load(filepath)
    
    scan = scan.get_fdata()
    return scan


def normalize(volume):
    
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume


def resize_volume(img):
    
    desired_depth = 64
    desired_width = 128
    desired_height = 128
    
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    
    img = ndimage.rotate(img, 90, reshape=False)
    
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img


def process_scan(path):
    
    volume = read_nifti_file(path)
    
    volume = normalize(volume)
    
    volume = resize_volume(volume)
    return volume



normal_scan_paths = [
    os.path.join(os.getcwd(), "MosMedData/CT-0", x)
    for x in os.listdir("MosMedData/CT-0")
]

abnormal_scan_paths = [
    os.path.join(os.getcwd(), "MosMedData/CT-23", x)
    for x in os.listdir("MosMedData/CT-23")
]

print("CT scans with normal lung tissue: " + str(len(normal_scan_paths)))
print("CT scans with abnormal lung tissue: " + str(len(abnormal_scan_paths)))


abnormal_scans = nump.array([process_scan(path) for path in abnormal_scan_paths])
normal_scans = nump.array([process_scan(path) for path in normal_scan_paths])

abnormal_labels = nump.array([1 for _ in range(len(abnormal_scans))])
normal_labels = nump.array([0 for _ in range(len(normal_scans))])

x_train = nump.concatenate((abnormal_scans[:70], normal_scans[:70]), axis=0)
y_train = nump.concatenate((abnormal_labels[:70], normal_labels[:70]), axis=0)
x_val = nump.concatenate((abnormal_scans[70:], normal_scans[70:]), axis=0)
y_val = nump.concatenate((abnormal_labels[70:], normal_labels[70:]), axis=0)
print(
    "Number of samples in train and validation are %d and %d."
    % (x_train.shape[0], x_val.shape[0])
)





@tflow.function
def rotate(volume):
    

    def scipy_rotate(volume):
        
        angles = [-20, -10, -5, 5, 10, 20]
        
        angle = random.choice(angles)
        
        volume = ndimage.rotate(volume, angle, reshape=False)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return volume

    augmented_volume = tflow.numpy_function(scipy_rotate, [volume], tflow.float32)
    return augmented_volume


def train_preprocessing(volume, label):
    
    volume = rotate(volume)
    volume = tflow.expand_dims(volume, axis=3)
    return volume, label


def validation_preprocessing(volume, label):
    
    volume = tflow.expand_dims(volume, axis=3)
    return volume, label




train_loader = tflow.data.Dataset.from_tensor_slices((x_train, y_train))
validation_loader = tflow.data.Dataset.from_tensor_slices((x_val, y_val))

batch_size = 2

train_dataset = (
    train_loader.shuffle(len(x_train))
    .map(train_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)

validation_dataset = (
    validation_loader.shuffle(len(x_val))
    .map(validation_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)



import matplotlib.pyplot as plt

data = train_dataset.take(1)
images, labels = list(data)[0]
images = images.numpy()
image = images[0]
print("Dimension of the CT scan is:", image.shape)
plt.imshow(nump.squeeze(image[:, :, 30]), cmap="gray")




def plot_slices(num_rows, num_columns, width, height, data):
  
    data = nump.rot90(nump.array(data))
    data = nump.transpose(data)
    data = nump.reshape(data, (num_rows, num_columns, width, height))
    rows_data, columns_data = data.shape[0], data.shape[1]
    heights = [slc[0].shape[0] for slc in data]
    widths = [slc.shape[1] for slc in data[0]]
    fig_width = 12.0
    fig_height = fig_width * sum(heights) / sum(widths)
    f, axarr = plt.subplots(
        rows_data,
        columns_data,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": heights},
    )
    for i in range(rows_data):
        for j in range(columns_data):
            axarr[i, j].imshow(data[i][j], cmap="gray")
            axarr[i, j].axis("off")
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.show()



plot_slices(4, 10, 128, 128, image[:, :, :40])



def get_model(width=128, height=128, depth=64):
   

    inputs = tkeras.Input((width, height, depth, 1))

    x = klayers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = klayers.MaxPool3D(pool_size=2)(x)
    x = klayers.BatchNormalization()(x)

    x = klayers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = klayers.MaxPool3D(pool_size=2)(x)
    x = klayers.BatchNormalization()(x)

    x = klayers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = klayers.MaxPool3D(pool_size=2)(x)
    x = klayers.BatchNormalization()(x)

    x = klayers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = klayers.MaxPool3D(pool_size=2)(x)
    x = klayers.BatchNormalization()(x)

    x = klayers.GlobalAveragePooling3D()(x)
    x = klayers.Dense(units=512, activation="relu")(x)
    x = klayers.Dropout(0.3)(x)

    outputs = klayers.Dense(units=1, activation="sigmoid")(x)


    model = tkeras.Model(inputs, outputs, name="3dcnn")
    return model


model = get_model(width=128, height=128, depth=64)
model.summary()


initial_learning_rate = 0.0001
lr_schedule = tkeras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)
model.compile(
    loss="binary_crossentropy",
    optimizer=tkeras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=["acc"],
)

checkpoint_cb = tkeras.callbacks.ModelCheckpoint(
    "3d_image_classification.h5", save_best_only=True
)
early_stopping_cb = tkeras.callbacks.EarlyStopping(monitor="val_acc", patience=15)

epochs = 100
model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    shuffle=True,
    verbose=2,
    callbacks=[checkpoint_cb, early_stopping_cb],
)


fig, ax = plt.subplots(1, 2, figsize=(20, 3))
ax = ax.ravel()

for i, metric in enumerate(["acc", "loss"]):
    ax[i].plot(model.history.history[metric])
    ax[i].plot(model.history.history["val_" + metric])
    ax[i].set_title("Model {}".format(metric))
    ax[i].set_xlabel("epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(["train", "val"])


model.load_weights("3d_image_classification.h5")
prediction = model.predict(nump.expand_dims(x_val[0], axis=0))[0]
scores = [1 - prediction[0], prediction[0]]

class_names = ["normal", "abnormal"]
for score, name in zip(scores, class_names):
    print(
        "This model is %.2f percent confident that CT scan is %s"
        % ((100 * score), name)
    )
