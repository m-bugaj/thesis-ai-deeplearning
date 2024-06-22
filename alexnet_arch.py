import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout
from keras.layers import BatchNormalization
from keras.layers.core import Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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

        # # AlexNet Implementation
        # model.add(Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), input_shape=(227, 227, 3)))
        # model.add(Activation("relu"))
        # model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

        # model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='same'))
        # model.add(Activation("relu"))
        # model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

        # model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
        # model.add(Activation("relu"))
        # model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
        # model.add(Activation("relu"))
        # model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
        # model.add(Activation("relu"))
        # model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

        # model.add(Flatten())
        # model.add(Dense(units=4096))
        # model.add(Activation("relu"))
        # model.add(Dropout(0.5))
        # model.add(Dense(units=4096))
        # model.add(Activation("relu"))
        # model.add(Dropout(0.5))
        # model.add(Dense(units=10)) # 10 - liczba klas w ImageNet
        # model.add(Activation("softmax"))

        # Warstwa 1: Conv2D -> Activation -> BatchNormalization -> MaxPooling2D
        model.add(Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), input_shape=(227, 227, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

        # Warstwa 2: Conv2D -> Activation -> BatchNormalization -> MaxPooling2D
        model.add(Conv2D(filters=256, kernel_size=(5,5), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

        # Warstwa 3: Conv2D -> Activation -> BatchNormalization
        model.add(Conv2D(filters=384, kernel_size=(3,3), padding='same', activation='relu'))
        model.add(BatchNormalization())

        # Warstwa 4: Conv2D -> Activation -> BatchNormalization
        model.add(Conv2D(filters=384, kernel_size=(3,3), padding='same', activation='relu'))
        model.add(BatchNormalization())

        # Warstwa 5: Conv2D -> Activation -> BatchNormalization -> MaxPooling2D
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

        # Warstwa 6: Flatten
        model.add(Flatten())

        # Warstwa 7: Dense -> Activation -> Dropout
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))

        # Warstwa 8: Dense -> Activation -> Dropout
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))

        # Warstwa wyjściowa: Dense -> Activation
        num_classes = 10  # liczba klas
        model.add(Dense(num_classes, activation='softmax'))

        # Wyświetlenie podsumowania modelu
        model.summary()


        model.compile(optimizer=compile_optimizer, loss=compile_loss, metrics=['accuracy'])

        train_datagen = ImageDataGenerator(
            rescale=1./255
        )
        test_datagen = ImageDataGenerator(
            rescale=1./255
        )

        train_generator = train_datagen.flow_from_directory(
            'DANE/archive/raw-img',
            target_size=(227, 227),
            batch_size=fit_batch_size,
            class_mode='categorical')

        test_generator = test_datagen.flow_from_directory(
            'DANE/archive/test',
            target_size=(227, 227),
            batch_size=fit_batch_size,
            class_mode='categorical')
        
        # Konfiguracja TensorBoard
        log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

        image_callback = TensorBoardImageCallback(log_dir, train_generator, test_generator)

        # Konfiguracja TensorFlow Profiler
        profile_dir = os.path.join(log_dir, 'profiler')
        tf.profiler.experimental.start(profile_dir)


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