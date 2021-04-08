import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar100
from tensorflow.keras import Input
from numpy import expand_dims
from tensorflow.keras.utils import to_categorical # Утилиты для one-hot encoding


def load_data():
    # load data
    (trainX, trainY), (testX, testY) = cifar100.load_data()
    #one-hot encoding
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    # convert from integers to floats
    train_norm = trainX.astype('float32')
    test_norm = testX.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # preprocess input
    train_norm = preprocess_input(train_norm)
    test_norm = preprocess_input(test_norm)
    return (train_norm, trainY), (test_norm, testY)

(trainX, trainY), (testX, testY) = load_data()


new_input = Input(shape=(32, 32, 3))
vgg = VGG16(weights="imagenet", include_top=False, input_tensor=new_input)
x = keras.layers.Flatten()(vgg.layers[-1].output)
x = keras.layers.Dense(64, activation='relu')(x)
outputs = keras.layers.Dense(100, activation='softmax')(x)
model = Model(inputs=[new_input], outputs=[outputs])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


from tensorflow.keras.callbacks import CSVLogger

csv_logger = CSVLogger('log_keras_ic_vgg16_cifer100.csv', append=True, separator=';')

import time


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


time_cb = TimeHistory()


history = model.fit(trainX, trainY, epochs=15, batch_size=64, validation_data=(testX, testY), callbacks=[time_cb, csv_logger])

loss, acc = model.evaluate(test_images, test_labels, verbose=0)
print("Total time: " + str(sum(time_cb.times)) + "s")
print("Test loss: " + str(loss) + "\nTest acc: " + str(acc))

with open("keras_ic_vgg16_cifar100_result.txt", "w") as fout:
    fout.write("Total time: " + str(sum(time_cb.times)) + "s\n")

    fout.write("Test loss: " + str(loss) + "\nTest acc: " + str(acc) + "\n")
    fout.write(str(time_cb.times))
