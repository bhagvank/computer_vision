

import matplotlib.pyplot as plot
import numpy as nump
import tensorflow as tflow
import tensorflow_datasets as tflowds

from tensorflow.keras import layers





(train_ds, val_ds, test_ds), metadata = tflowds.load(
    'tf_flowers',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)




num_classes = metadata.features['label'].num_classes
print(num_classes)




get_label_name = metadata.features['label'].int2str

image, label = next(iter(train_ds))
_ = plot.imshow(image)
_ = plot.title(get_label_name(label))



IMG_SIZE = 180

resize_and_rescale = tflow.keras.Sequential([
  layers.experimental.preprocessing.Resizing(IMG_SIZE, IMG_SIZE),
  layers.experimental.preprocessing.Rescaling(1./255)
])




result = resize_and_rescale(image)
_ = plot.imshow(result)




print("Min and max pixel values:", result.numpy().min(), result.numpy().max())




data_augmentation = tflow.keras.Sequential([
  layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
  layers.experimental.preprocessing.RandomRotation(0.2),
])


image = tflow.expand_dims(image, 0)





plot.figure(figsize=(10, 10))
for i in range(9):
  augmented_image = data_augmentation(image)
  ax = plot.subplot(3, 3, i + 1)
  plot.imshow(augmented_image[0])
  plot.axis("off")




model = tflow.keras.Sequential([
  resize_and_rescale,
  data_augmentation,
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  # Rest of your model
])




aug_ds = train_ds.map(
  lambda x, y: (resize_and_rescale(x, training=True), y))




batch_size = 32
AUTOTUNE = tflow.data.AUTOTUNE

def prepare(ds, shuffle=False, augment=False):
  # Resize and rescale all datasets
  ds = ds.map(lambda x, y: (resize_and_rescale(x), y), 
              num_parallel_calls=AUTOTUNE)

  if shuffle:
    ds = ds.shuffle(1000)

  # Batch all datasets
  ds = ds.batch(batch_size)

  # Use data augmentation only on the training set
  if augment:
    ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), 
                num_parallel_calls=AUTOTUNE)

  # Use buffered prefecting on all datasets
  return ds.prefetch(buffer_size=AUTOTUNE)


# In[15]:


train_ds = prepare(train_ds, shuffle=True, augment=True)
val_ds = prepare(val_ds)
test_ds = prepare(test_ds)





model = tflow.keras.Sequential([
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])





model.compile(optimizer='adam',
              loss=tflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])





epochs=5
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)





loss, acc = model.evaluate(test_ds)
print("Accuracy", acc)





def random_invert_img(x, p=0.5):
  if  tflow.random.uniform([]) < p:
    x = (255-x)
  else:
    x
  return x





def random_invert(factor=0.5):
  return layers.Lambda(lambda x: random_invert_img(x, factor))

random_invert = random_invert()





plot.figure(figsize=(10, 10))
for i in range(9):
  augmented_image = random_invert(image)
  ax = plot.subplot(3, 3, i + 1)
  plot.imshow(augmented_image[0].numpy().astype("uint8"))
  plot.axis("off")





class RandomInvert(layers.Layer):
  def __init__(self, factor=0.5, **kwargs):
    super().__init__(**kwargs)
    self.factor = factor

  def call(self, x):
    return random_invert_img(x)





_ = plot.imshow(RandomInvert()(image)[0])





(train_ds, val_ds, test_ds), metadata = tflowds.load(
    'tf_flowers',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)





image, label = next(iter(train_ds))
_ = plot.imshow(image)
_ = plot.title(get_label_name(label))




def visualize(original, augmented):
  fig = plot.figure()
  plot.subplot(1,2,1)
  plot.title('Original image')
  plot.imshow(original)

  plot.subplot(1,2,2)
  plot.title('Augmented image')
  plot.imshow(augmented)





flipped = tflow.image.flip_left_right(image)
visualize(image, flipped)




grayscaled = tflow.image.rgb_to_grayscale(image)
visualize(image, tflow.squeeze(grayscaled))
_ = plot.colorbar()





saturated = tflow.image.adjust_saturation(image, 3)
visualize(image, saturated)





bright = tflow.image.adjust_brightness(image, 0.4)
visualize(image, bright)





cropped = tflow.image.central_crop(image, central_fraction=0.5)
visualize(image,cropped)





rotated = tflow.image.rot90(image)
visualize(image, rotated)





for i in range(3):
  seed = (i, 0)  # tuple of size (2,)
  stateless_random_brightness = tflow.image.stateless_random_brightness(
      image, max_delta=0.95, seed=seed)
  visualize(image, stateless_random_brightness)





for i in range(3):
  seed = (i, 0)  # tuple of size (2,)
  stateless_random_contrast = tflow.image.stateless_random_contrast(
      image, lower=0.1, upper=0.9, seed=seed)
  visualize(image, stateless_random_contrast)





for i in range(3):
  seed = (i, 0)  # tuple of size (2,)
  stateless_random_crop = tflow.image.stateless_random_crop(
      image, size=[210, 300, 3], seed=seed)
  visualize(image, stateless_random_crop)




(train_datasets, val_ds, test_ds), metadata = tflowds.load(
    'tf_flowers',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)





def resize_and_rescale(image, label):
  image = tflow.cast(image, tflow.float32)
  image = tflow.image.resize(image, [IMG_SIZE, IMG_SIZE])
  image = (image / 255.0)
  return image, label





def augment(image_label, seed):
  image, label = image_label
  image, label = resize_and_rescale(image, label)
  image = tflow.image.resize_with_crop_or_pad(image, IMG_SIZE + 6, IMG_SIZE + 6)
  # Make a new seed
  new_seed = tflow.random.experimental.stateless_split(seed, num=1)[0, :]
  # Random crop back to the original size
  image = tflow.image.stateless_random_crop(
      image, size=[IMG_SIZE, IMG_SIZE, 3], seed=seed)
  # Random brightness
  image = tflow.image.stateless_random_brightness(
      image, max_delta=0.5, seed=new_seed)
  image = tflow.clip_by_value(image, 0, 1)
  return image, label




counter = tflow.data.experimental.Counter()
train_ds = tflow.data.Dataset.zip((train_datasets, (counter, counter)))




train_ds = (
    train_ds
    .shuffle(1000)
    .map(augment, num_parallel_calls=AUTOTUNE)
    .batch(batch_size)
    .prefetch(AUTOTUNE)
)




val_ds = (
    val_ds
    .map(resize_and_rescale, num_parallel_calls=AUTOTUNE)
    .batch(batch_size)
    .prefetch(AUTOTUNE)
)





test_ds = (
    test_ds
    .map(resize_and_rescale, num_parallel_calls=AUTOTUNE)
    .batch(batch_size)
    .prefetch(AUTOTUNE)
)



rng = tflow.random.Generator.from_seed(123, alg='philox')


def f(x, y):
  seed = rng.make_seeds(2)[0]
  image, label = augment((x, y), seed)
  return image, label





train_ds = (
    train_datasets
    .shuffle(1000)
    .map(f, num_parallel_calls=AUTOTUNE)
    .batch(batch_size)
    .prefetch(AUTOTUNE)
)





val_ds = (
    val_ds
    .map(resize_and_rescale, num_parallel_calls=AUTOTUNE)
    .batch(batch_size)
    .prefetch(AUTOTUNE)
)





test_ds = (
    test_ds
    .map(resize_and_rescale, num_parallel_calls=AUTOTUNE)
    .batch(batch_size)
    .prefetch(AUTOTUNE)
)


