"""Train a simple CNN-Capsule Network on the CIFAR10 small images dataset.

Without Data Augmentation:
It gets to 75% validation accuracy in 10 epochs,
and 79% after 15 epochs, and overfitting after 20 epochs

With Data Augmentation:
It gets to 75% validation accuracy in 10 epochs,
and 79% after 15 epochs, and 83% after 30 epcohs.
In my test, highest validation accuracy is 83.79% after 50 epcohs.

This is a fast Implement, just 20s/epcoh with a gtx 1070 gpu.
"""

from __future__ import print_function
import keras
from keras import backend as K
from keras.layers import Layer
from keras import activations
from keras import utils
from keras.datasets import fashion_mnist
from keras.models import Model
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator

batch_size = 128
num_classes = 10
epochs = 5
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
img_rows, img_cols = 28, 28
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)


# if K.image_data_format() == 'channels_first':
#     x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
#     x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
#     input_shape = (1, img_rows, img_cols)
# else:
#     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
#     input_shape = (img_rows, img_cols, 1)
# A common Conv2D model

input_image = Input(shape=(28, 28, 1))
x = Conv2D(32, (3, 3), activation='relu')(input_image)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = AveragePooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = Conv2D(128, (3, 3), activation='relu')(x)


"""now we reshape it as (batch_size, input_num_capsule, input_dim_capsule)
then connect a Capsule layer.

the output of final model is the lengths of 10 Capsule, whose dim=16.

the length of Capsule is the proba,
so the problem becomes a 10 two-classification problem.
"""
# x = Reshape((-1, 128))(x)
# capsule = Capsule(10, 16, 3, True)(x)
# print(capsule.shape)
# output = Lambda(lambda z: K.sqrt(K.sum(K.square(z), -1)))(capsule)
x = Flatten()(x)
x = Dense(512)(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)
x = Dense(num_classes)(x)
output = Activation('softmax')(x)
model = Model(inputs=input_image, outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.summary()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# we can compare the performance with or without data augmentation
# data_augmentation = True
#
# if not data_augmentation:
#     print('Not using data augmentation.')
#     model.fit(
#         x_train,
#         y_train,
#         batch_size=batch_size,
#         epochs=epochs,
#         validation_data=(x_test, y_test),
#         shuffle=True)
# else:
#     print('Using real-time data augmentation.')
#     # This will do preprocessing and realtime data augmentation:
    # datagen = ImageDataGenerator(
    #     featurewise_center=False,  # set input mean to 0 over the dataset
    #     samplewise_center=False,  # set each sample mean to 0
    #     featurewise_std_normalization=False,  # divide inputs by dataset std
    #     samplewise_std_normalization=False,  # divide each input by its std
    #     zca_whitening=False,  # apply ZCA whitening
    #     zca_epsilon=1e-06,  # epsilon for ZCA whitening
    #     rotation_range=0,  # randomly rotate images in 0 to 180 degrees
    #     width_shift_range=0.1,  # randomly shift images horizontally
    #     height_shift_range=0.1,  # randomly shift images vertically
    #     shear_range=0.,  # set range for random shear
    #     zoom_range=0.,  # set range for random zoom
    #     channel_shift_range=0.,  # set range for random channel shifts
    #     fill_mode='nearest',  # set mode for filling points outside the input boundaries
    #     cval=0.,  # value used for fill_mode = "constant"
    #     horizontal_flip=True,  # randomly flip images
    #     vertical_flip=False,  # randomly flip images
    #     rescale=None,  # set rescaling factor (applied before any other transformation)
    #     preprocessing_function=None,  # set function that will be applied on each input
    #     data_format=None,  # image data format, either "channels_first" or "channels_last"
    #     validation_split=0.0)  # fraction of images reserved for validation (strictly between 0 and 1)
    #
    # # Compute quantities required for feature-wise normalization
    # # (std, mean, and principal components if ZCA whitening is applied).
    # datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    # model.fit_generator(
    #     datagen.flow(x_train, y_train, batch_size=batch_size),
    #     epochs=epochs,
    #     validation_data=(x_test, y_test),
    #     workers=3)
