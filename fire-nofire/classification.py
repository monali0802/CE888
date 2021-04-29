
#########################################################
# import libraries

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from config import Config_classification
from config import new_size

batch_size = Config_classification.get('batch_size')
image_size = (new_size.get('width'), new_size.get('height'))
epochs = Config_classification.get('Epochs')


#########################################################
# Function definition

def classify():
    """
    This function load the trained model from the previous task and evaluates the performance of that over the test
    data set.
    :return: None, Plot the Confusion matrix for the test data on the binary classification
    """

    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        "frames/Test", seed=1337, batch_size=batch_size, image_size=image_size, shuffle=True
    )
    AUTOTUNE = tf.data.AUTOTUNE
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)


    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "frames/Test", seed=1337, image_size=image_size, batch_size=batch_size, shuffle=True
    )

    model_fire = load_model('resnet152_model.h5')
    test_loss, accuracy = model_fire.evaluate(test_ds)
    print('Test accuracy :', accuracy)