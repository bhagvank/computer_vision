

import numpy as nump
import time

import PIL.Image as PILImage
import matplotlib.pylab as matplot

import tensorflow as tflow
import tensorflow_hub as tflowhub




classifier_model ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4" #@param {type:"string"}





IMAGE_SHAPE = (224, 224)

classifier = tflow.keras.Sequential([
    tflowhub.KerasLayer(classifier_model, input_shape=IMAGE_SHAPE+(3,))
])




grace_hopper = tflow.keras.utils.get_file('image.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')
grace_hopper = PILImage.open(grace_hopper).resize(IMAGE_SHAPE)
grace_hopper




grace_hopper = nump.array(grace_hopper)/255.0
grace_hopper.shape



result = classifier.predict(grace_hopper[nump.newaxis, ...])
result.shape




predicted_class = nump.argmax(result[0], axis=-1)
predicted_class




labels_path = tflow.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = nump.array(open(labels_path).read().splitlines())




matplot.imshow(grace_hopper)
matplot.axis('off')
predicted_class_name = imagenet_labels[predicted_class]
_ = matplot.title("Prediction: " + predicted_class_name.title())




data_root = tflow.keras.utils.get_file(
  'flower_photos','https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
   untar=True)




batch_size = 32
img_height = 224
img_width = 224

train_ds = tflow.keras.preprocessing.image_dataset_from_directory(
  str(data_root),
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)




class_names = nump.array(train_ds.class_names)
print(class_names)




normalization_layer = tflow.keras.layers.experimental.preprocessing.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))




AUTOTUNE = tflow.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)




for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break




result_batch = classifier.predict(train_ds)




predicted_class_names = imagenet_labels[nump.argmax(result_batch, axis=-1)]
predicted_class_names




matplot.figure(figsize=(10,9))
matplot.subplots_adjust(hspace=0.5)
for n in range(30):
  matplot.subplot(6,5,n+1)
  matplot.imshow(image_batch[n])
  matplot.title(predicted_class_names[n])
  matplot.axis('off')
_ = matplot.suptitle("ImageNet predictions")




feature_extractor_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4" #@param {type:"string"}




feature_extractor_layer = tflowhub.KerasLayer(
    feature_extractor_model, input_shape=(224, 224, 3), trainable=False)




feature_batch = feature_extractor_layer(image_batch)
print(feature_batch.shape)




num_classes = len(class_names)

model = tflow.keras.Sequential([
  feature_extractor_layer,
  tflow.keras.layers.Dense(num_classes)
])

model.summary()




predictions = model(image_batch)




predictions.shape




model.compile(
  optimizer=tflow.keras.optimizers.Adam(),
  loss=tflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['acc'])




class CollectBatchStats(tflow.keras.callbacks.Callback):
  def __init__(self):
    self.batch_losses = []
    self.batch_acc = []

  def on_train_batch_end(self, batch, logs=None):
    self.batch_losses.append(logs['loss'])
    self.batch_acc.append(logs['acc'])
    self.model.reset_metrics()

batch_stats_callback = CollectBatchStats()

history = model.fit(train_ds, epochs=2,
                    callbacks=[batch_stats_callback])




matplot.figure()
matplot.ylabel("Loss")
matplot.xlabel("Training Steps")
matplot.ylim([0,2])
matplot.plot(batch_stats_callback.batch_losses)




matplot.figure()
matplot.ylabel("Accuracy")
matplot.xlabel("Training Steps")
matplot.ylim([0,1])
matplot.plot(batch_stats_callback.batch_acc)




predicted_batch = model.predict(image_batch)
predicted_id = nump.argmax(predicted_batch, axis=-1)
predicted_label_batch = class_names[predicted_id]




matplot.figure(figsize=(10,9))
matplot.subplots_adjust(hspace=0.5)
for n in range(30):
  matplot.subplot(6,5,n+1)
  matplot.imshow(image_batch[n])
  matplot.title(predicted_label_batch[n].title())
  matplot.axis('off')
_ = matplot.suptitle("Model predictions")




t = time.time()

export_path = "/tmp/saved_models/{}".format(int(t))
model.save(export_path)

export_path




reloaded = tflow.keras.models.load_model(export_path)




result_batch = model.predict(image_batch)
reloaded_result_batch = reloaded.predict(image_batch)



abs(reloaded_result_batch - result_batch).max()


