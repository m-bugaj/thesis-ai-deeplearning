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
import psutil
import time

class MnistClassifier:

    # def __init__(self):
    #     # Inicjalizacja NVML
    #     nvmlInit()
    #     self.device_count = nvmlDeviceGetCount()
    #     self.gpu_usage_data = []
    
    # def __del__(self):
    #     # Zamykanie NVML przy zniszczeniu obiektu
    #     nvmlShutdown()

    def display_history(self, history, training_time):
        # df = pd.DataFrame(history)
        # ax = df.plot(figsize=(8, 5))
        # plt.grid(True)
        # # plt.gca().set_ylim(0, 1)

        # # Dodanie tekstu pod wykresem
        # plt.subplots_adjust(bottom=0.2)  # Adjust bottom to make space for text
        # plt.text(0.5, -0.15, f'Czas treningu: {training_time:.2f} sekund', color='red', 
        #          ha='center', va='top', transform=ax.transAxes, fontsize=12)

        # plt.savefig('out/history.png')
        # plt.show()

        df = pd.DataFrame(history)
        
        fig, ax1 = plt.subplots(figsize=(8, 5))

        # Skala po lewej stronie dla accuracy
        ax1.plot(df.index, df['accuracy'], 'b-', label='Accuracy')
        ax1.plot(df.index, df['val_accuracy'], 'g-', label='Validation Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.legend(loc='upper left')

        # Tworzenie drugiej osi y dla loss
        ax2 = ax1.twinx()
        ax2.plot(df.index, df['loss'], 'r-', label='Loss')
        ax2.plot(df.index, df['val_loss'], 'm-', label='Validation Loss')
        ax2.set_ylabel('Loss', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.legend(loc='upper right')

        fig.tight_layout()  # Dostosowanie layoutu, żeby elementy się nie nakładały
        ax1.grid(True)

        # Dodanie tekstu pod wykresem
        plt.subplots_adjust(bottom=0.2)  # Adjust bottom to make space for text
        plt.text(0.5, -0.15, f'Czas treningu: {training_time:.2f} sekund', color='red', 
                 ha='center', va='top', transform=ax1.transAxes, fontsize=12)

        plt.savefig('out/history.png')
        plt.show()

    def display_combined_history(self, history, gpu_usage_data, training_time):
        # Tworzenie wykresu łączonego dla historii trenowania i danych GPU
        history_df = pd.DataFrame(history)
        gpu_usage_df = pd.DataFrame(gpu_usage_data)
        
        fig, ax1 = plt.subplots(figsize=(8, 5))

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy / Loss')
        ax1.plot(history_df.index, history_df['accuracy'], 'b-', label='Accuracy')
        ax1.plot(history_df.index, history_df['val_accuracy'], 'g-', label='Validation Accuracy')
        ax1.plot(history_df.index, history_df['loss'], 'r-', label='Loss')
        ax1.plot(history_df.index, history_df['val_loss'], 'm-', label='Validation Loss')
        ax1.legend(loc='upper left')
        
        ax2 = ax1.twinx()
        ax2.set_ylabel('GPU Utilization (%) / Memory Utilization (%)')
        ax2.plot(gpu_usage_df.index, gpu_usage_df['gpu_utilization'], 'k-', label='GPU Utilization')
        ax2.plot(gpu_usage_df.index, gpu_usage_df['memory_utilization'], 'c-', label='Memory Utilization')
        ax2.legend(loc='upper right')

        fig.tight_layout()
        plt.grid(True)
        # Dodanie tekstu pod wykresem
        plt.subplots_adjust(bottom=0.2)  # Adjust bottom to make space for text
        plt.text(0.5, -0.15, f'Czas treningu: {training_time:.2f} sekund', color='red', 
                 ha='center', va='top', transform=ax1.transAxes, fontsize=12)
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

        def on_epoch_end(self, epoch, logs=None):
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
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=6)

        # Callbacks definitions
        system_usage_logger = SystemUsageLogger(log_dir=log_dir, log_frequency=3)
        image_callback = TensorBoardImageCallback(log_dir, train_generator, test_generator, log_frequency=3)
        log_gpu_usage_callback = GPUUsageLogger()
        profiler_callback = ProfilerCallback(log_dir=log_dir, log_frequency=6)
        measuring_time = MeasuringTime()
        
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

        start_time = time.time()

        history = model.fit(train_generator, 
                                steps_per_epoch=train_generator.samples // fit_batch_size, 
                                epochs=fit_epochs, validation_data=test_generator, 
                                callbacks = [measuring_time, log_gpu_usage_callback, system_usage_logger, tensorboard_callback, image_callback, profiler_callback])
        stop_time = time.time()

        # Pobieranie danych z `gpu_logger` po zakończeniu trenowania
        gpu_usage_data = log_gpu_usage_callback.on_train_end()

        training_time = measuring_time.on_train_end()

        # Wyświetlenie historii trenowania oraz danych dotyczących użycia GPU
        self.display_history(history.history, training_time)
        self.display_combined_history(history.history, gpu_usage_data, training_time)


        
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
        print("Total time: {}".format(stop_time-start_time))


class TensorBoardImageCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, train_data, test_data, log_frequency=1):
        super().__init__()
        self.log_dir = log_dir
        self.train_data = train_data
        self.test_data = test_data
        self.writer = tf.summary.create_file_writer(os.path.join(log_dir, 'images'))
        self.log_frequency = log_frequency

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.log_frequency == 0:
            # Wybierz kilka przykładów z danych treningowych i walidacyjnych
            train_images, _ = next(self.train_data)
            test_images, _ = next(self.test_data)
            
            # Przygotuj obrazy do zapisu
            with self.writer.as_default():
                tf.summary.image('Training Images', train_images, step=epoch)
                tf.summary.image('Test Images', test_images, step=epoch)

class SystemUsageLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, log_frequency=3):
        super(SystemUsageLogger, self).__init__()
        self.log_dir = log_dir
        self.file_writer = tf.summary.create_file_writer(log_dir)
        self.log_frequency = log_frequency
    
    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.log_frequency == 0:    
            # Log system usage
            memory_info = psutil.virtual_memory()
            used_ram = memory_info.used / (1024 ** 3)  # Convert to GB
            available_ram = memory_info.available / (1024 ** 3)  # Convert to GB
            cpu_usage = psutil.cpu_percent()

            with self.file_writer.as_default():
                tf.summary.scalar('used_ram', used_ram, step=epoch)
                tf.summary.scalar('available_ram', available_ram, step=epoch)
                tf.summary.scalar('cpu_usage', cpu_usage, step=epoch)

class GPUUsageLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        super(GPUUsageLogger, self).__init__()
        nvmlInit()
        self.device_count = nvmlDeviceGetCount()
        self.gpu_usage_data = []

    def __del__(self):
        # Zamykanie NVML przy zniszczeniu obiektu
        nvmlShutdown()

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
    
    def on_epoch_end(self, epoch, logs=None):
        self.log_gpu_usage(epoch+1)

    def on_train_end(self, logs=None):
        return self.gpu_usage_data

class ProfilerCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, log_frequency=3):
        super(ProfilerCallback, self).__init__()
        self.log_dir = log_dir
        self.log_frequency = log_frequency

    def on_epoch_begin(self, epoch, logs=None):
        if epoch % self.log_frequency == 0: 
            # Konfiguracja TensorFlow Profiler
            options = tf.profiler.experimental.ProfilerOptions(host_tracer_level=2) # Adjust CPU tracing level. Values are: 1 - critical info only, 2 - info, 3 - verbose. [default value is 2]
            profile_dir = os.path.join(self.log_dir, 'profiler')
            tf.profiler.experimental.start(profile_dir, options=options)
    
    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.log_frequency == 0: 
            # Zatrzymanie TensorFlow Profiler
            tf.profiler.experimental.stop()

class MeasuringTime(tf.keras.callbacks.Callback):
    def __init__(self):
        super(MeasuringTime, self).__init__()
        self.start_time = 0
        self.total_time = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        end_time = time.time()
        execution_time = end_time - self.start_time
        self.total_time += execution_time

    def on_train_end(self, logs=None):
        print("Training time: {}".format(self.total_time))
        return self.total_time