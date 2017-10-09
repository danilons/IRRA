#!/usr/bin/env python3
# coding: utf-8
import numpy as np
import cv2
import glob
import os
from keras.layers import Activation, Reshape, Dropout
from keras.layers import AtrousConvolution2D, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D
from keras.models import Sequential
from keras import callbacks, optimizers


IMAGE_PATH = '/home/danilonunes/segmentation/image/'
LABEL_PATH = '/home/danilonunes/segmentation/label/'
IMAGE_SIZE = (384, 384, 3)
LEARNING_RATE = 1e-4
WEIGHTS = 'conversion/converted/dilation8_pascal_voc.npy'
CHECKPOINT_PATH = 'checkpoint'
N_CLASSES = 106
BATCH_SIZE = 2
FILENAME = 'checkpoint/dilatedNet_keras.h5'
EPOCHS = 2
STEPS_PER_EPOCH = int(4367/BATCH_SIZE)  # train set size
VAL_STEPS_PER_EPOCH = int(4317/BATCH_SIZE)  # test set size
MEAN_FILE = 'mean.npy'


def build_model(n_classes=106, trainable=True):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', name='conv1_1', input_shape=IMAGE_SIZE))
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu', name='conv2_1'))
    model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu', name='conv3_1'))
    model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu', name='conv3_2'))
    model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu', name='conv4_1'))
    model.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu', name='conv4_2'))
    model.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu', name='conv4_3'))

    model.add(Conv2D(512, (3, 3), padding='same', dilation_rate=(2, 2), activation='relu', name='conv5_1'))
    model.add(Conv2D(512, (3, 3), padding='same', dilation_rate=(2, 2), activation='relu', name='conv5_2'))
    model.add(Conv2D(512, (3, 3), padding='same', dilation_rate=(2, 2), activation='relu', name='conv5_3'))

    model.add(Conv2D(4096, (7, 7), padding='same', dilation_rate=(4, 4), activation='relu', name='fc6'))
    model.add(Dropout(0.5))
    model.add(Conv2D(4096, (1, 1), activation='relu', name='fc7'))
    model.add(Dropout(0.5))

    model.add(Conv2D(n_classes, (1, 1), activation='linear', name='fc-final-{}'.format(n_classes)))
    model.add(Conv2DTranspose(n_classes, (16, 16), strides=(1, 1), padding='same'))
    model.add(UpSampling2D(size=(8, 8)))

    if trainable:
      _, curr_width, curr_height, curr_channels = model.layers[-1].output_shape
      model.add(Reshape((curr_width, curr_height, curr_channels)))
      model.add(Activation('softmax'))

    print(model.summary())
    return model


def generate_samples(mu, mode='train', batch_size=2):
    while True:
        fnames = glob.glob(os.path.join(IMAGE_PATH, mode, '*.jpg'))
        images = []
        labels = []
        for fname in fnames:
            image = cv2.imread(fname) - mu
            image = image[:, :, (2, 1, 0)]
            label = cv2.imread(os.path.join(LABEL_PATH, mode, os.path.basename(fname).replace('.jpg', '.png')),0)
            images.append(image)
            labels.append(label[:, :, np.newaxis])
            images.append(cv2.flip(image, 1))
            labels.append(cv2.flip(label, 1)[:, :, np.newaxis])
            if len(images) == batch_size:
                yield (np.array(images), np.array(labels))
                images = []
                labels = []


def main():
    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)

    mean = np.load(MEAN_FILE)
    model = build_model(N_CLASSES, trainable=True)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=LEARNING_RATE, momentum=0.9),
                  metrics=['accuracy'])

    model_checkpoint = callbacks.ModelCheckpoint(CHECKPOINT_PATH + '/snapshot_ep{epoch:02d}-vl{val_loss:.4f}.hdf5', monitor='loss')
    tensorboard_cback = callbacks.TensorBoard(log_dir='{}/tboard'.format(CHECKPOINT_PATH),
                                              histogram_freq=0,
                                              write_graph=False,
                                              write_images=False)

    csv_log_cback = callbacks.CSVLogger('{}/history.log'.format(CHECKPOINT_PATH))
    reduce_lr_cback = callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                  factor=0.2,
                                                  patience=5,
                                                  verbose=1,
                                                  min_lr=0.05 * LEARNING_RATE)

    weights_data = np.load(WEIGHTS, encoding='latin1').item()
    for layer in model.layers:
        if layer.name in weights_data.keys():
            print("Loading weight for layer {}".format(layer.name))
            layer_weights = weights_data[layer.name]
            layer.set_weights((layer_weights['weights'],
                               layer_weights['biases']))

    model.fit_generator(generate_samples(mean, mode='train', batch_size=BATCH_SIZE),
                        steps_per_epoch=STEPS_PER_EPOCH,
                        epochs=EPOCHS,
                        validation_data=generate_samples(mean, mode='test', batch_size=BATCH_SIZE),
                        validation_steps=VAL_STEPS_PER_EPOCH,
                        callbacks=[model_checkpoint,
                                   tensorboard_cback,
                                   csv_log_cback,
                                   reduce_lr_cback])

    model.save(FILENAME)

if __name__ == "__main__":
    main()