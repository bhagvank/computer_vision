from numpy import asarray
from numpy import unique
from numpy import argmax
from tensorflow.keras.datasets.mnist import load_data as load_mnist_data
from tensorflow.keras import Sequential as keras_Seq
from tensorflow.keras.layers import Dense as keras_Dense
from tensorflow.keras.layers import Conv2D as keras_Conv2D
from tensorflow.keras.layers import MaxPool2D as keras_MaxPool2D
from tensorflow.keras.layers import Flatten as keras_Flatten
from tensorflow.keras.layers import Dropout as keras_Dropout

(train_xcoord, train_ycoord), (test_xcoord, test_ycoord) = load_mnist_data()

train_xcoord = train_xcoord.reshape((train_xcoord.shape[0], train_xcoord.shape[1], train_xcoord.shape[2], 1))
test_xcoord = test_xcoord.reshape((test_xcoord.shape[0], test_xcoord.shape[1], test_xcoord.shape[2], 1))

in_shape = train_xcoord.shape[1:]

n_classes = len(unique(train_ycoord))
print(in_shape, n_classes)

train_xcoord = train_xcoord.astype('float32') / 255.0
test_xcoord = test_xcoord.astype('float32') / 255.0

model = keras_Seq()
model.add(keras_Conv2D(32, (3,3), activation='relu', kernel_initializer='he_uniform', input_shape=in_shape))
model.add(keras_MaxPool2D((2, 2)))
model.add(keras_Flatten())
model.add(keras_Dense(100, activation='relu', kernel_initializer='he_uniform'))
model.add(keras_Dropout(0.5))
model.add(keras_Dense(n_classes, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_xcoord, train_ycoord, epochs=10, batch_size=128, verbose=0)
loss, acc = model.evaluate(test_xcoord, test_ycoord, verbose=0)
print('Accuracy: %.3f' % acc)

image = train_xcoord[0]
yhat = model.predict(asarray([image]))
print('Predicted class is %d' % argmax(yhat))
