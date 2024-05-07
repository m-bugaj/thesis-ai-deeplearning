# import os
# import tensorflow as tf
# from keras.models import Model
# from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout, Concatenate, Input, AveragePooling2D
# from keras.utils import to_categorical
# import cv2
# import numpy as np
# from PIL import Image

# class MnistClassifier:
#     def inception_module(self, x, filters):
#         # 1x1 conv
#         conv1 = Conv2D(filters[0], (1, 1), padding='same', activation='relu')(x)

#         # 1x1 conv -> 3x3 conv
#         conv3 = Conv2D(filters[1], (1, 1), padding='same', activation='relu')(x)
#         conv3 = Conv2D(filters[2], (3, 3), padding='same', activation='relu')(conv3)

#         # 1x1 conv -> 5x5 conv
#         conv5 = Conv2D(filters[3], (1, 1), padding='same', activation='relu')(x)
#         conv5 = Conv2D(filters[4], (5, 5), padding='same', activation='relu')(conv5)

#         # 3x3 max pooling -> 1x1 conv
#         pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
#         pool = Conv2D(filters[5], (1, 1), padding='same', activation='relu')(pool)

#         # concatenate filters
#         out = Concatenate()([conv1, conv3, conv5, pool])
#         return out

#     def load_png(self, data_path):
#         images = []
#         labels = []

#         for folder in os.listdir(data_path):
#             folder_path = os.path.join(data_path, folder)
#             for image_file in os.listdir(folder_path):
#                 img = Image.open(os.path.join(folder_path, image_file))
#                 img = img.convert('RGB')  # Ensure all images are RGB
#                 img = img.resize((224, 224))  # Resize image to 224x224 pixels
#                 img = np.array(img)  # Convert image to a numpy array
#                 img = img.astype('float16') / 255.0  # Normalize data to float values from 0 to 1
#                 images.append(img)

#                 label = folder
#                 labels.append(label)

#         return np.array(images), np.array(labels)

    
#     def train_model(self, model_name, compile_optimizer, compile_loss, fit_epochs, fit_batch_size):

#         input_layer = Input(shape=(224, 224, 3))

#         # Initial convolution and pooling layers
#         x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(input_layer)
#         x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

#         # Inception modules
#         x = self.inception_module(x, [64, 96, 128, 16, 32, 32])
#         x = self.inception_module(x, [128, 128, 192, 32, 96, 64])
#         x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

#         x = self.inception_module(x, [192, 96, 208, 16, 48, 64])
#         x = self.inception_module(x, [160, 112, 224, 24, 64, 64])
#         x = self.inception_module(x, [128, 128, 256, 24, 64, 64])
#         x = self.inception_module(x, [112, 144, 288, 32, 64, 64])
#         x = self.inception_module(x, [256, 160, 320, 32, 128, 128])
#         x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

#         x = self.inception_module(x, [256, 160, 320, 32, 128, 128])
#         x = self.inception_module(x, [384, 192, 384, 48, 128, 128])

#         # Finishing layers
#         x = AveragePooling2D((7, 7), strides=(1, 1))(x)
#         x = Flatten()(x)
#         x = Dropout(0.4)(x)
#         x = Dense(10, activation='softmax')(x)  # 1000 classes in ImageNet

#         # Create model
#         model = Model(inputs=input_layer, outputs=x)
#         model.summary()

#         model.compile(optimizer=compile_optimizer, loss=compile_loss, metrics=['accuracy'])

#         x_train, y_train = self.load_png('C:\\Users\\BUGI\\Desktop\\Learn\\Magisterka\\DANE\\archive\\raw-img')
#         x_test, y_test = self.load_png('C:\\Users\\BUGI\\Desktop\\Learn\\Magisterka\\DANE\\archive\\test')

#         # Przekonwertowanie etykiety na kodowanie one-hot
#         y_train = to_categorical(y_train)
#         y_test = to_categorical(y_test)

#         model.fit(x_train, y_train, epochs=fit_epochs, batch_size=fit_batch_size)

#         # Generowanie ścieżki do pliku JSON
#         json_file_path = os.path.join('models', 'model' + model_name + '.json')
#         weights_file_path = os.path.join('models', 'model' + model_name + '.weights.h5')

#         # Zapisanie modelu do pliku JSON
#         json_string = model.to_json()
#         with open(json_file_path, 'w') as json_file:
#             json_file.write(json_string)

#         model.save_weights(weights_file_path)

#         score = model.evaluate(x_test, y_test, verbose=0)
#         print('Test loss:', score[0])
#         print('Test accuracy:', score[1])

import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout, Concatenate, Input, AveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

class MnistClassifier:
    def inception_module(self, x, filters):
        # 1x1 conv
        conv1 = Conv2D(filters[0], (1, 1), padding='same', activation='relu')(x)
        # 1x1 conv -> 3x3 conv
        conv3 = Conv2D(filters[1], (1, 1), padding='same', activation='relu')(x)
        conv3 = Conv2D(filters[2], (3, 3), padding='same', activation='relu')(conv3)
        # 1x1 conv -> 5x5 conv
        conv5 = Conv2D(filters[3], (1, 1), padding='same', activation='relu')(x)
        conv5 = Conv2D(filters[4], (5, 5), padding='same', activation='relu')(conv5)
        # 3x3 max pooling -> 1x1 conv
        pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
        pool = Conv2D(filters[5], (1, 1), padding='same', activation='relu')(pool)
        # concatenate filters
        out = Concatenate()([conv1, conv3, conv5, pool])
        return out
    
    def train_model(self, model_name, compile_optimizer, compile_loss, fit_epochs, fit_batch_size):
        input_layer = Input(shape=(224, 224, 3))
        # Initial convolution and pooling layers
        x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(input_layer)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        # Inception modules
        x = self.inception_module(x, [64, 96, 128, 16, 32, 32])
        x = self.inception_module(x, [128, 128, 192, 32, 96, 64])
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = self.inception_module(x, [192, 96, 208, 16, 48, 64])
        x = self.inception_module(x, [160, 112, 224, 24, 64, 64])
        x = self.inception_module(x, [128, 128, 256, 24, 64, 64])
        x = self.inception_module(x, [112, 144, 288, 32, 64, 64])
        x = self.inception_module(x, [256, 160, 320, 32, 128, 128])
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = self.inception_module(x, [256, 160, 320, 32, 128, 128])
        x = self.inception_module(x, [384, 192, 384, 48, 128, 128])
        # Finishing layers
        x = AveragePooling2D((7, 7), strides=(1, 1))(x)
        x = Flatten()(x)
        x = Dropout(0.4)(x)
        x = Dense(10, activation='softmax')(x)  # Assuming 10 classes for MNIST-like example

        # Create model
        model = Model(inputs=input_layer, outputs=x)
        model.summary()

        model.compile(optimizer=compile_optimizer, loss=compile_loss, metrics=['accuracy'])

        train_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            'C:\\Users\\BUGI\\Desktop\\Learn\\Magisterka\\DANE\\archive\\raw-img',
            target_size=(224, 224),
            batch_size=fit_batch_size,
            class_mode='categorical')

        test_generator = test_datagen.flow_from_directory(
            'C:\\Users\\BUGI\\Desktop\\Learn\\Magisterka\\DANE\\archive\\test',
            target_size=(224, 224),
            batch_size=fit_batch_size,
            class_mode='categorical')

        model.fit(train_generator, steps_per_epoch=train_generator.samples // fit_batch_size, epochs=fit_epochs)

        model.save('models\\' + model_name + '.h5')

        score = model.evaluate(test_generator, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

