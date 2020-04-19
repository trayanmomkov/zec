# Copyright 2020 Trayan Momkov
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import sys
print('###>>> Python version: ', sys.version, '\n')

import tensorflow as tf
print('###>>> TensorFlow version: ', tf.version.VERSION, '\n')

import numpy
import os
import errno
from datetime import datetime as dt
from scipy import ndimage
from matplotlib import pyplot as plt
from os.path import join

if hasattr(__builtins__, 'raw_input'):
    input = raw_input

IMAGE_SIZE_IN_PIXELS = 32
NUMBER_OF_PIXELS = IMAGE_SIZE_IN_PIXELS * IMAGE_SIZE_IN_PIXELS


def mkdir(dir_path):
    """
    Creates dir with its parents without error if already exists.
    """
    try:
        os.makedirs(dir_path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(dir_path):
            pass


def save_tflite(saved_model_path, datetime):
    mkdir('tflite_models')
    tflite_filepath = join('tflite_models', datetime + ".tflite")

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
    converter.experimental_new_converter = True
    tflite_model = converter.convert()

    open(tflite_filepath, "wb").write(tflite_model)
    print('###>>> tflite model saved at:', tflite_filepath, '\n')


def save_model(model):
    mkdir('saved_models')
    datetime = dt.today().strftime('%Y-%b-%d_%H-%M-%S')
    saved_model_path = join('saved_models', datetime)

    model.save(saved_model_path)
    print('###>>> Model saved at: ', saved_model_path, '\n')

    save_tflite(saved_model_path, datetime)


def plot_results(history, hp):
    # plot loss during training
    plt.subplot(2, 1, 1)
    plt.title('Loss' + ' (layers: ' + str(hp['layers'])
              + ', rate: ' + str(numpy.format_float_positional(hp['rate'])) + ' dropout: ' + str(hp['drop']) + ')')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.legend()

    # plot accuracy during training
    plt.subplot(2, 1, 2)
    plt.title('accuracy')
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='validation')
    plt.legend()

    plt.show()


def show_image(image_data):
    image_data = image_data.reshape(IMAGE_SIZE_IN_PIXELS, IMAGE_SIZE_IN_PIXELS)
    plt.imshow(ndimage.rotate(image_data, 90), aspect="auto", origin="lower")
    plt.show()


def show_predictions(model, dataset):
    X, y = prepare_data(X=dataset[:, 1:], y=dataset[:, :1])
    predictions = model.predict(X)
    for i, prediction in enumerate(list(predictions)):
        result = prediction[0]
        digit = '0' if result < 0.5 else '8'
        confidence = 1-2*(1-result) if result > 0.5 else 1-2*result
        correctness = 'WRONG!' if digit != ('0' if y[i] == 0 else '8') else ''
        print('###>>> Prediction: ' + digit + ' Confidence: ' + str(round(confidence, 2)) + ' ' + correctness)
        show_image(X[i, :])


def prepare_data(X, y):
    X = numpy.array(list(map(lambda x: x / 255.0, X)))
    y = numpy.array(list(map(lambda x: 0 if x == 0 else 1, y)))
    return X, y


def create_model_and_train(dataset, layers, drop, rate, epochs, batch):

    # Create model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(layers[0], input_dim=NUMBER_OF_PIXELS, activation='relu'))
    model.add(tf.keras.layers.Dropout(drop))

    for units in layers[1:]:
        # Hidden layers
        model.add(tf.keras.layers.Dense(units, activation='relu'))
        model.add(tf.keras.layers.Dropout(drop))

    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Prepare data
    X, y = prepare_data(X=dataset[:, 1:], y=dataset[:, :1])

    # Train
    history = model.fit(X, y,
                        epochs=epochs,
                        batch_size=batch,
                        validation_split=0.14)  # 14% for validation

    return model, history


def main():
    # Train
    hyper_parameters = {'layers': [1024, 1024], 'rate': 0.00001, 'drop': 0.2, 'epochs': 20, 'batch': 10}

    training_and_validation_dataset = numpy.loadtxt(join('dataset', 'training.csv'), delimiter=',')

    model, history = create_model_and_train(
        dataset=training_and_validation_dataset,
        layers=hyper_parameters['layers'],
        drop=hyper_parameters['drop'],
        rate=hyper_parameters['rate'],
        epochs=hyper_parameters['epochs'],
        batch=hyper_parameters['batch']
    )

    # Show result
    plot_results(history=history, hp=hyper_parameters)
    val_loss = history.history['val_loss'][-1]
    val_accuracy = history.history['val_accuracy'][-1]

    print('###>>>', hyper_parameters)
    print('###>>> Final validation loss:', round(val_loss, 2), 'Final validation accuracy:', round(val_accuracy, 2), '\n')

    # Save the model and generate tflite file
    save_model(model)

    # Predict
    test_dataset = numpy.loadtxt(join('dataset', 'test.csv'), delimiter=',')
    show_predictions(model, test_dataset)


if __name__ == '__main__':
    main()
