import tensorflow as tf
from tensorflow import keras

print(tf.config.list_physical_devices('GPU'))


print("Keras version: ", keras.__version__)
print("Tensorflow version: ", tf.__version__)
