import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from keras.utils import to_categorical
import cv2
import numpy as np

class MnistClassifier:

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

        model = Sequential();

        # Dodanie warstwy konwolucyjnej CNN
        model.add(Conv2D(filters=filters, kernel_size=filter_size, activation=activation, input_shape=(28, 28, 1)))

        # Dodanie warstwy max-pooling
        model.add(MaxPooling2D((2, 2)))

        # Spłaszcz dane
        model.add(Flatten())

        # Dodaj warstwy ukryte
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))

        # Dodaj warstwę wyjściową
        model.add(Dense(10, activation='softmax'))

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