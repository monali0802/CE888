
#########################################################
# import libraries

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# from plotdata import plot_confusion_matrix
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
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "frames/Test", seed=1337, image_size=image_size, batch_size=batch_size, shuffle=True
    )

    model_fire = load_model('Output/Models/model_fire_resnet_weighted_40_no_metric_simple')

    _ = model_fire.evaluate(test_ds, batch_size=batch_size)
