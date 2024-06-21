import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, AveragePooling2D, Dropout
# from keras.api.preprocessing.image import ImageDataGenerator
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization
from keras.utils import to_categorical
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import time
from pynvml import *
from keras.callbacks import TensorBoard
import datetime

class MnistClassifier:

    def __init__(self):
        # Inicjalizacja NVML
        nvmlInit()
        self.device_count = nvmlDeviceGetCount()
        self.gpu_usage_data = []
    
    def __del__(self):
        # Zamykanie NVML przy zniszczeniu obiektu
        nvmlShutdown()

    def display_history(self, history):
        pd.DataFrame(history).plot(figsize=(8,5))
        plt.grid(True)
        plt.gca().set_ylim(0,1)
        plt.savefig('out/history.png')
        plt.show()

    def display_combined_history(self, history, gpu_usage_data):
        # Tworzenie wykresu łączonego dla historii trenowania i danych GPU
        history_df = pd.DataFrame(history)
        gpu_usage_df = pd.DataFrame(gpu_usage_data)
        
        fig, ax1 = plt.subplots(figsize=(8, 5))

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy / Loss')
        ax1.plot(history_df.index, history_df['accuracy'], 'b-', label='Accuracy')
        ax1.plot(history_df.index, history_df['val_accuracy'], 'g-', label='Validation Accuracy')
        ax1.plot(history_df.index, history_df['loss'], 'r-', label='Loss')
        ax1.plot(history_df.index, history_df['val_loss'], 'y-', label='Validation Loss')
        ax1.legend(loc='upper left')
        
        ax2 = ax1.twinx()
        ax2.set_ylabel('GPU Utilization (%)')
        ax2.plot(gpu_usage_df.index, gpu_usage_df['gpu_utilization'], 'k-', label='GPU Utilization')
        ax2.legend(loc='upper right')

        fig.tight_layout()
        plt.grid(True)
        plt.savefig('out/combined_history.png')
        plt.show()

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
    def preprocess_image(self, image):
        # Upewnienie się, że obraz jest typu float32 i znormalizowany
        image = image.astype(np.float32) / 255.0

        # Przekształć do skali szarości, jeśli nie jest
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Przeskalowanie do 55x55
        resized_image = cv2.resize(image, (55, 55))

        # Przycięcie do 32x32 (z góry, dołu, lewej i prawej)
        width, height = resized_image.shape[1], resized_image.shape[0]
        left = (width - 32) // 2
        top = (height - 32) // 2
        right = left + 32
        bottom = top + 32
        cropped_image = resized_image[top:bottom, left:right]

        # Upewnienie się, że obraz ma tylko jeden kanał
        cropped_image = np.expand_dims(cropped_image, axis=-1)

        return cropped_image

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
        usage_data = {
            'epoch': epoch,
            'gpu_utilization': gpu_usages[0]['gpu_utilization'],
            'memory_utilization': gpu_usages[0]['memory_utilization']
        }
        self.gpu_usage_data.append(usage_data)
        print(f"Epoch {epoch}: GPU {gpu_usages[0]['gpu']}: Memory Total: {gpu_usages[0]['memory_total']} | Memory Free: {gpu_usages[0]['memory_free']} | Memory Used: {gpu_usages[0]['memory_used']} | GPU Utilization: {gpu_usages[0]['gpu_utilization']}% | Memory Utilization: {gpu_usages[0]['memory_utilization']}%")


    
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
            color_mode='grayscale',
            shuffle=True  # Ensure shuffling of data
        )

        test_generator = test_datagen.flow_from_directory(
            'DANE/archive/test',
            target_size=(32, 32),  # Update target_size to (32, 32)
            batch_size=fit_batch_size,
            class_mode='categorical',
            color_mode='grayscale',
            shuffle=False  # No need to shuffle test data
        )

        # Konfiguracja TensorBoard
        log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

        image_callback = TensorBoardImageCallback(log_dir, train_generator, test_generator)

        # Konfiguracja TensorFlow Profiler
        profile_dir = os.path.join(log_dir, 'profiler')
        tf.profiler.experimental.start(profile_dir)
        
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

        full_history = {
            'accuracy': [],
            'loss': [],
            'val_accuracy': [],
            'val_loss': []
        }

        for epoch in range(fit_epochs):
            history = model.fit(train_generator, steps_per_epoch=train_generator.samples // fit_batch_size, epochs=1, validation_data=test_generator, callbacks = [tensorboard_callback, image_callback])
            self.log_gpu_usage(epoch + 1)
            
            full_history['accuracy'].append(history.history['accuracy'][0])
            full_history['loss'].append(history.history['loss'][0])
            full_history['val_accuracy'].append(history.history['val_accuracy'][0])
            full_history['val_loss'].append(history.history['val_loss'][0])

        # Zatrzymanie TensorFlow Profiler
        tf.profiler.experimental.stop()

        self.display_history(full_history)
        self.display_combined_history(full_history, self.gpu_usage_data)

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



class TensorBoardImageCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, train_data, test_data):
        super().__init__()
        self.log_dir = log_dir
        self.train_data = train_data
        self.test_data = test_data
        self.writer = tf.summary.create_file_writer(os.path.join(log_dir, 'images'))

    def on_epoch_end(self, epoch, logs=None):
        # Wybierz kilka przykładów z danych treningowych i walidacyjnych
        train_images, _ = next(self.train_data)
        test_images, _ = next(self.test_data)
        
        # Przygotuj obrazy do zapisu
        with self.writer.as_default():
            tf.summary.image('Training Images', train_images, step=epoch)
            tf.summary.image('Test Images', test_images, step=epoch)