import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, AveragePooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization
from keras.utils import to_categorical
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
from pynvml import *

class MnistClassifier:

    def __init__(self):
        # Inicjalizacja NVML
        nvmlInit()
        self.device_count = nvmlDeviceGetCount()
    
    def __del__(self):
        # Zamykanie NVML przy zniszczeniu obiektu
        nvmlShutdown()

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
    
    # Ta część została dodana w celu optymalizacji zbioru danych do architekury LeNet-5. Z racji, że architektura wymaga małych rozmiarów obrazów 32x32, postanowiono najpierw wstępnie przeskalować obraz do wartości o zadowalającej jakości, a następnie przyciąć obraz do wartości 32x32. Działa to głównie wtedy gdy wykrywany obiekt znajduje się na środku obrazu.
    def preprocess_image(self, image, seed=None):
        # Konwersja do PIL Image, jeśli to konieczne
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image.astype(np.uint8))
        
        # Przeskalowanie do 50x50
        resized_image = image.resize((55, 55))
        
        # Przycięcie do 32x32 (z góry, dołu, lewej i prawej)
        width, height = resized_image.size
        left = (width - 32) // 2
        top = (height - 32) // 2
        right = left + 32
        bottom = top + 32
        cropped_image = resized_image.crop((left, top, right, bottom))
        
        return np.array(cropped_image) / 255.0  # Normalizacja do [0, 1]
    
    def get_gpu_usage(self):
        gpu_usages = []
        for i in range(self.device_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            info = nvmlDeviceGetMemoryInfo(handle)
            utilization = nvmlDeviceGetUtilizationRates(handle)
            gpu_usages.append({
                'gpu': i,
                'memory_total': info.total,
                'memory_free': info.free,
                'memory_used': info.used,
                'gpu_utilization': utilization.gpu,
                'memory_utilization': utilization.memory
            })
        return gpu_usages
    
    def log_gpu_usage(self, epoch):
        gpu_usages = self.get_gpu_usage()
        for usage in gpu_usages:
            print(f"Epoch {epoch}: GPU {usage['gpu']}: Memory Total: {usage['memory_total']} | Memory Free: {usage['memory_free']} | Memory Used: {usage['memory_used']} | GPU Utilization: {usage['gpu_utilization']}% | Memory Utilization: {usage['memory_utilization']}%")


    
    def train_model(self, model_name, compile_optimizer, compile_loss, fit_epochs, fit_batch_size):

        model = Sequential();

        # LeNet-5 Implementation

        # C1: (None,32,32,1) -> (None,28,28,6).
        model.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=(32,32,1), padding='valid'))
        model.add(BatchNormalization())
        # P1: (None,28,28,6) -> (None,14,14,6).
        model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

        # C2: (None,14,14,6) -> (None,10,10,16).
        model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
        model.add(BatchNormalization())
        # P2: (None,10,10,16) -> (None,5,5,16).
        model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        # model.add(Dropout(0.4))
        # Flatten: (None,5,5,16) -> (None, 400).
        model.add(Flatten())

        # FC1: (None, 400) -> (None,120).
        model.add(Dense(120, activation='tanh'))
        model.add(Dropout(0.5))

        # FC2: (None,120) -> (None,84).
        model.add(Dense(84, activation='tanh'))
        model.add(Dropout(0.5))

        # FC3: (None,84) -> (None,10).
        model.add(Dense(10, activation='softmax'))


        model.compile(optimizer=compile_optimizer, loss=compile_loss, metrics=['accuracy'])
        

        train_datagen = ImageDataGenerator(
            preprocessing_function=self.preprocess_image
        )

        test_datagen = ImageDataGenerator(
            preprocessing_function=self.preprocess_image
        )

        # Generatory danych
        train_generator = train_datagen.flow_from_directory(
            'DANE/archive/raw-img',
            target_size=(32, 32),  # Update target_size to (32, 32)
            batch_size=fit_batch_size,
            class_mode='categorical',
            shuffle=True  # Ensure shuffling of data
        )

        test_generator = test_datagen.flow_from_directory(
            'DANE/archive/test',
            target_size=(32, 32),  # Update target_size to (32, 32)
            batch_size=fit_batch_size,
            class_mode='categorical',
            shuffle=False  # No need to shuffle test data
        )
        
        # # Pobieranie jednej partii obrazów i etykiet
        # images, labels = next(train_generator)

        # fig, axes = plt.subplots(10, 2, figsize=(10, 50))

        # for i in range(10):

        #     # Wybieranie pierwszego obrazu z partii
        #     image = images[i]

        #     # Przetworzony obraz w skali szarości
        #     axes[i, 1].imshow(image.squeeze(), cmap='gray')
        #     axes[i, 1].set_title(f'Przetworzony obraz {i+1} (skala szarości)')
        #     axes[i, 1].axis('off')

        # # Wyświetlanie obrazu
        # # plt.imshow(image.squeeze(), cmap='gray')
        # plt.title('Przetworzony obraz')
        # plt.axis('off')
        # plt.show()

        # print("Kształt obrazu:", image.shape)
        # print("Etykieta:", labels[0])

        for epoch in range(fit_epochs):
            model.fit(train_generator, steps_per_epoch=train_generator.samples // fit_batch_size, epochs=1, validation_data=test_generator)
            self.log_gpu_usage(epoch + 1)

        # model.fit(train_generator, steps_per_epoch=train_generator.samples // fit_batch_size, epochs=fit_epochs, validation_data=test_generator)

        # Generowanie ścieżki do pliku JSON
        json_file_path = os.path.join('models', 'model' + model_name + '.json')
        weights_file_path = os.path.join('models', 'model' + model_name + '.weights.h5')

        # Zapisanie modelu do pliku JSON
        json_string = model.to_json()
        with open(json_file_path, 'w') as json_file:
            json_file.write(json_string)

        model.save_weights(weights_file_path)

        score = model.evaluate(test_generator, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])