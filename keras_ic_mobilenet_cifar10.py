from tensorflow import keras
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.models import Model
from keras.datasets import cifar10
from keras import Input
from keras.utils import np_utils # Утилиты для one-hot encoding


test_name = "keras_ic_mobilenet_cifar10"


def load_data():
    # load data
    (trainX, trainY), (testX, testY) = cifar10.load_data()
    #one-hot encoding
    trainY = np_utils.to_categorical(trainY)
    testY = np_utils.to_categorical(testY)
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
mobile = MobileNet(weights="imagenet", include_top=False, input_tensor=new_input)
x = keras.layers.Flatten()(mobile.layers[-1].output)
x = keras.layers.Dense(64, activation='relu')(x)
outputs = keras.layers.Dense(10, activation='softmax')(x)
model = Model(inputs=[new_input], outputs=[outputs])


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# Callbacks
from keras.callbacks import CSVLogger

csv_logger = CSVLogger('log_'+test_name+'.csv', append=True, separator=';')

import time


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


time_cb = TimeHistory()

# fit model
start = time.time()
history = model.fit(trainX, trainY, epochs=15, batch_size=64, validation_data=(testX, testY), callbacks=[time_cb, csv_logger])
end = time.time()

with open(test_name+"_result.txt", "w") as fout:
    fout.write("Total time: " + str(end - start) + "s\n")

    loss, acc = model.evaluate(testX, testY, verbose=0)
    fout.write("Test loss: " + str(loss) + "\nTest acc: " + str(acc) + "\n")
    fout.write(str(time_cb.times))
