import keras
from keras.datasets import mnist # Датасет

test_name = "keras_test"

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


from keras.callbacks import CSVLogger

csv_logger = CSVLogger('log_keras_test.csv', append=True, separator=';')

import time


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


time_cb = TimeHistory()


start = time.time()
history = model.fit(
    train_images,
    train_labels,
    epochs=5,
    batch_size=128,
    validation_data=(test_images, test_labels),
    callbacks=[csv_logger, time_cb]
)
end = time.time()

with open("keras_test_result.txt", "w") as fout:
    fout.write("Total time: " + str(end - start) + "s\n")

    loss, acc = model.evaluate(test_images, test_labels, verbose=0)
    fout.write("Test loss: " + str(loss) + "\nTest acc: " + str(acc) + "\n")
    fout.write(str(time_cb.times))