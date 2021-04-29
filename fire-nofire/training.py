#########################################################
# import libraries

import os.path
import tensorflow as tf
import matplotlib.pyplot as plt

from config import new_size
from config import Config_classification

#########################################################
# Global parameters and definition
image_size = (new_size.get('width'), new_size.get('height'))
batch_size = Config_classification.get('batch_size')
epochs = Config_classification.get('Epochs')


#########################################################
# Function definition

def train_keras():
    print(" --------- Training --------- ")

    dir_fire = 'frames/Training/Fire/'
    dir_no_fire = 'frames/Training/No_Fire/'

    # 0 is Fire and 1 is NO_Fire
    fire = len([name for name in os.listdir(dir_fire) if os.path.isfile(os.path.join(dir_fire, name))])
    no_fire = len([name for name in os.listdir(dir_no_fire) if os.path.isfile(os.path.join(dir_no_fire, name))])
    total = fire + no_fire
    weight_for_fire = (1 / fire) * total / 2.0
    weight_for_no_fire = (1 / no_fire) * total / 2.0

    print("Weight for class fire : {:.2f}".format(weight_for_fire))
    print("Weight for class No_fire : {:.2f}".format(weight_for_no_fire))

    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        "frames/Training", validation_split=0.2, subset="training", seed=1337,
        batch_size=batch_size, image_size=image_size, shuffle=True
    )

    validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        "frames/Training", validation_split=0.2, subset="validation", seed=1337,
        shuffle=True, batch_size=batch_size, image_size=image_size
    )

    class_names = train_dataset.class_names
    plt.figure(figsize=(10, 10))
    for images, labels in train_dataset.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")

    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)
    ])
    for image, _ in train_dataset.take(1):
        plt.figure(figsize=(10, 10))
        # first_image = image[0]
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            augmented_image = data_augmentation(image)
            plt.imshow(augmented_image[0] / 255)
            plt.axis('off')

    preprocess_input = tf.keras.applications.resnet.preprocess_input
    rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255, offset=-1)

    input_shape = image_size + (3,)
    base_model = tf.keras.applications.ResNet152(input_shape=input_shape,
                                                 include_top=False,
                                                 weights='imagenet')

    image_batch, label_batch = next(iter(train_dataset))
    feature_batch = base_model(image_batch)
    print(feature_batch.shape)

    base_model.trainable = False
    base_model.summary()

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)
    print(feature_batch_average.shape)

    prediction_layer = tf.keras.layers.Dense(1)
    prediction_batch = prediction_layer(feature_batch_average)
    print(prediction_batch.shape)

    inputs = tf.keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    base_model.trainable = True

    # Let's take a look to see how many layers are in the base model
    print("Number of layers in the base model: ", len(base_model.layers))

    # Fine-tune from this layer onwards
    fine_tune_at = 3

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    base_learning_rate = 0.0001
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate / 10),
                  metrics=['accuracy'])

    model.summary()

    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(filepath='resnet152_model.h5', save_best_only=True),
    ]

    history_fine = model.fit(train_dataset,
                             epochs=epochs,
                             callbacks=my_callbacks,
                             validation_data=validation_dataset)

    acc = history_fine.history['accuracy']
    val_acc = history_fine.history['val_accuracy']

    loss = history_fine.history['loss']
    val_loss = history_fine.history['val_loss']

    tf.keras.utils.plot_model(
        model,
        to_file="model.png",
        show_shapes=True,
        show_layer_names=True
    )

    epochss = range(len(acc))
    plt.plot(epochss, acc, 'bo', label='Training accuracy')
    plt.plot(epochss, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()

    plt.plot(epochss, loss, 'bo', label='Training loss')
    plt.plot(epochss, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()

