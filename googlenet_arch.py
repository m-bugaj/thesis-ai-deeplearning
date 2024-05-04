import os
import tensorflow as tf
from keras.models import Model
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout, Concatenate, Input, AveragePooling2D
from keras.utils import to_categorical
import cv2
import numpy as np

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

    def load_mnist_png(self, data_path):
        images = []
        labels = []

        for folder in os.listdir(data_path):
            folder_path = os.path.join(data_path, folder)
            for image_file in os.listdir(folder_path):
                img = cv2.imread(os.path.join(folder_path, image_file), cv2.IMREAD_GRAYSCALE)
                img = img.astype('float32') / 255.0 # Normalizacja danych do wartości zmienno przecinkowych od 0 do 1
                images.append(img)

                label = int(folder)
                labels.append(label)

        return np.array(images), np.array(labels)
    
    def train_model(self, model_name, filters, filter_size, activation, compile_optimizer, compile_loss, fit_epochs, fit_batch_size):

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
        x = Dense(1000, activation='softmax')(x)  # 1000 classes in ImageNet

        # Create model
        model = Model(inputs=input_layer, outputs=x)
        model.summary()

        model.compile(optimizer=compile_optimizer, loss=compile_loss, metrics=['accuracy'])

        x_train, y_train = self.load_mnist_png('C:\\Users\\BUGI\\Desktop\\Learn\\MachineLearning\\FirstProjMNIST\\DATA\\mnistpng\\mnist_png\\mnist_png\\training')
        x_test, y_test = self.load_mnist_png('C:\\Users\\BUGI\\Desktop\\Learn\\MachineLearning\\FirstProjMNIST\\DATA\\mnistpng\\mnist_png\\mnist_png\\testing')

        # Przekonwertowanie etykiety na kodowanie one-hot
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        model.fit(x_train, y_train, epochs=fit_epochs, batch_size=fit_batch_size)

        # Generowanie ścieżki do pliku JSON
        json_file_path = os.path.join('models', 'model' + model_name + '.json')
        weights_file_path = os.path.join('models', 'model' + model_name + '.weights.h5')

        # Zapisanie modelu do pliku JSON
        json_string = model.to_json()
        with open(json_file_path, 'w') as json_file:
            json_file.write(json_string)

        model.save_weights(weights_file_path)

        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])