import tensorflow as tf
import tensorflow_datasets as tfds
# import tensorflow_hub as hub
import json
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import numpy as np

print('Using:')
print('🟣 tf version:', tf.__version__)
print('🟣 tf.keras version:', tf.keras.__version__)
print('🟣 tfds version:', tfds.__version__)

config = tfds.download.DownloadConfig(register_checksums=True)
dataset, dataset_info = tfds.load(
    'oxford_flowers102',
    split=['train', 'validation', 'test'],
    as_supervised=True,
    with_info=True,
    download_and_prepare_kwargs={'download_config': config}
)
training_set, validation_set, test_set = dataset
dataset_info

num_training_examples = dataset_info.splits['train'].num_examples
num_validation_examples = dataset_info.splits['validation'].num_examples
num_test_examples = dataset_info.splits['test'].num_examples

print(f'There are {num_training_examples} images in the training set')
print(f'There are {num_validation_examples} images in the validation set')
print(f'There are {num_test_examples} images in the test set')

num_classes = dataset_info.features['label'].num_classes
print(f'There are {num_classes} classes in the dataset')

for image, label in training_set.take(1):
    image = image.numpy().squeeze()
    label = label.numpy()

    plt.imshow(image, cmap=plt.cm.binary)
    plt.colorbar()
    plt.title(str(label))
    plt.show()

for image, label in training_set.take(1):
    image = image.numpy().squeeze()
    label = label.numpy()

    # Define class_names using the dataset_info
    class_names = dataset_info.features['label'].names

    plt.imshow(image, cmap=plt.cm.binary)
    plt.colorbar()
    # Use the class_names list to get the class name based on the label
    plt.title(class_names[label])
    plt.show()

batch_size = 32
image_size = 224

def format_image(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    return image, label

training_batches = training_set.shuffle(num_training_examples//4).map(format_image).batch(batch_size).prefetch(1)
validation_batches = validation_set.map(format_image).batch(batch_size).prefetch(1)
testing_batches = test_set.map(format_image).batch(batch_size).prefetch(1)

from keras import layers
from keras.layers import Input, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.models import Model
from keras.applications import MobileNetV2
from keras.optimizers import Adam

num_classes = len(class_names)
inputs = Input((image_size, image_size, 3))
base_model = MobileNetV2(input_tensor=inputs, include_top=False, weights='imagenet')

def build_model(size, num_classes):
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, x)
    return model

model = build_model(image_size, num_classes)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

EPOCHS = 10
history = model.fit(
    training_batches,
    epochs=EPOCHS,
    validation_data=validation_batches
)

training_accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']
training_loss = history.history['loss']
validation_loss = history.history['val_loss']
epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, training_accuracy, label='Training Accuracy')
plt.plot(epochs_range, validation_accuracy, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, training_loss, label='Training Loss')
plt.plot(epochs_range, validation_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

model.save('oxford_flower_model.keras')

from google.colab import files
files.download('oxford_flower_model.keras')

loss, accuracy = model.evaluate(testing_batches)
print('Loss on the TEST Set: {:.3f}'.format(loss))
print(' Accuracy on the TEST Set: {:.3%}'.format(accuracy))

