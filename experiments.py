# # from googlenet_arch import GoogLeNet
# # from alexnet_arch import AlexNet
# from lenet5_arch import LeNet5
# from keras.optimizers import Adam

# class Experiments:

#     def get_models(self):
#         # gn = GoogLeNet()
#         # an = AlexNet()
#         ln = LeNet5()

#         # model_name = ['f-32_fs-[3_3]_a-relu_co-adam_cl-categorical_crossentropy_fe-10_fbs-64']
#         # filters = [32]
#         # filter_size = [(3, 3)]
#         # activation = ['relu']
#         # compile_optimizer = ['adam']
#         # compile_loss = ['categorical_crossentropy']
#         # fit_epochs = [10]
#         # fit_batch_size = [64]

#         # for i in range(len(model_name)):
#         #     mc.train_model(model_name[i], filters[i], filter_size[i], activation[i], compile_optimizer[i], compile_loss[i], fit_epochs[i], fit_batch_size[i])

#         # GoogLeNet
#         model_names = ['bs-8__af-RELU__lr-Adam1e-03__20240626-230515']
#         arch_name = ['GoogLeNet']
#         # compile_optimizers = [Adam(lr = 1e-3)]
#         compile_optimizers = [Adam(learning_rate=1e-3)]
#         compile_losses = ['categorical_crossentropy']
#         fit_epochs = [25]
#         fit_batch_sizes = [16]
#         log_custom_dir = 'E:\!MAGISTERKA\LOGI'

#         # Iteracja po modelach i ich parametrach do treningu
#         for i in range(len(model_names)):
#             ln.train_model(model_names[i], arch_name[i], compile_optimizers[i], compile_losses[i], fit_epochs[i], fit_batch_sizes[i], log_custom_dir)


# if __name__ == "__main__":
#     experiments = Experiments()
#     experiments.get_models()

# from googlenet_arch import GoogLeNet
# from alexnet_arch import AlexNet
from lenet5_arch import LeNet5
# from keras.optimizers import adam_experimental as Adam
# from keras.optimizer_v1 import Adam
from keras.optimizer_v2.adam import Adam
from keras.optimizer_v1 import SGD
from keras import backend as K
import tensorflow as tf
import os
import IPython
# from keras.optimizers import sgd_experimental as SGD
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
import math


class Experiments:

    def __init__(self):
        self.models = {
            'LeNet5': LeNet5
            # 'AlexNet': AlexNet
            # 'GoogLeNet': GoogLeNet
        }

    def get_models(self):
        return self.models
    
    def fix_gpu(self):
        devices = tf.config.experimental.list_physical_devices('GPU')
        if devices:
            try:
                for device in devices:
                    tf.config.experimental.set_memory_growth(device, True)
                config = tf.compat.v1.ConfigProto()
                session = tf.compat.v1.InteractiveSession(config=config)
                print("GPU memory growth enabled")
                return session
            except RuntimeError as e:
                print(e)
        else:
            print("No GPU devices found")
            return None

    def train_all_models(self, learning_rate=1e-3, batch_sizes=[128], epochs=3):

        def scheduler(epoch, lr):
            if epoch < 7:
                return lr
            else:
                return lr * math.exp(-0.1)
        
        # models = self.get_models()
        activation_functions = ['leaky_relu']
        # activation_functions = ['leaky_relu', 'elu']
        # activation_functions = ['relu', 'elu']
        # optimizer = Adam(learning_rate=learning_rate)
        # optimizer_name = f'Adam{learning_rate:.0e}'
        optimizer_configs = [
            # {
            #     "name": "SGD1e-3",
            #     "optimizer": SGD(learning_rate=0.001),
            #     "callbacks": []
            # },
            # {
            #     "name": "Adam1e-3",
            #     "optimizer": Adam(learning_rate=1e-3),
            #     "callbacks": []
            # },
            # {
            #     "name": "SGD_ReduceLROnPlateau_monitor-'val_loss'_factor-0.1_patience-3_min_lr-1e-6",
            #     "optimizer": SGD(learning_rate=0.01),
            #     "callbacks": [ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-5)]
            # },
            {
                "name": f"Adam_LearningRateScheduler_startLR-1e-3_expDecay-0.1_after{7}epochs",
                "optimizer": Adam(lr=1e-3),
                "callbacks": [LearningRateScheduler(scheduler)]
            }
            # {
            #     "name": "Adam1e-3_ReduceLROnPlateau_monitor-'val_loss'_factor-0.1_patience-5_min_lr-1e-5",
            #     "optimizer": Adam(learning_rate=1e-3),
            #     "callbacks": [ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-5)]
            # }
            # {
            #     "name": "SGD1e-3_ReduceLROnPlateau_monitor-'val_loss'_factor-0.1_patience-5_min_lr-1e-5",
            #     "optimizer": SGD(learning_rate=1e-3),
            #     "callbacks": [ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-5)]
            # }
        ]
        loss_function = 'categorical_crossentropy'
        log_custom_dir = 'E:\!MAGISTERKA\LOGI'

        for arch_name, model_class in self.models.items():
            for batch_size in batch_sizes:
                for activation_function in activation_functions:
                    for opt_config in optimizer_configs:
                        # session = self.fix_gpu()
                        # IPython.Application.instance().kernel.do_shutdown(True)

                        optimizer_configs = [
                            # {
                            #     "name": "SGD1e-3",
                            #     "optimizer": SGD(learning_rate=0.001),
                            #     "callbacks": []
                            # },
                            # {
                            #     "name": "Adam1e-3",
                            #     "optimizer": Adam(learning_rate=1e-3),
                            #     "callbacks": []
                            # },
                            {
                                "name": f"Adam_LearningRateScheduler_startLR-1e-3_expDecay-0.1_after{7}epochs",
                                "optimizer": Adam(lr=1e-3),
                                "callbacks": [LearningRateScheduler(scheduler)]
                            }
                            # {
                            #     "name": "Adam1e-3_ReduceLROnPlateau_monitor-'val_loss'_factor-0.1_patience-5_min_lr-1e-5",
                            #     "optimizer": Adam(learning_rate=1e-3),
                            #     "callbacks": [ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-5)]
                            # }
                            # {
                            #     "name": "SGD1e-3_ReduceLROnPlateau_monitor-'val_loss'_factor-0.1_patience-5_min_lr-1e-5",
                            #     "optimizer": SGD(learning_rate=1e-3),
                            #     "callbacks": [ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-5)]
                            # }
                        ]

                        # Tworzenie nowej instancji modelu
                        model_instance = model_class()

                        # Generowanie czytelnej nazwy modelu
                        readable_model_name = f'bs-{batch_size}__af-{activation_function}__opt-{opt_config["name"]}'
                        print(f'Training {readable_model_name} for {arch_name}...')
                        
                        # optimizer = Adam(learning_rate=learning_rate)
                        
                        # Trening modelu
                        model_instance.train_model(
                            model_name=readable_model_name,
                            arch_name=arch_name,
                            compile_optimizer=opt_config["optimizer"],
                            compile_loss=loss_function,
                            fit_epochs=epochs,
                            fit_batch_size=batch_size,
                            activation_function=activation_function,
                            log_custom_dir=log_custom_dir,
                            callbacks=opt_config["callbacks"]
                        )
                        print(f'Training of {readable_model_name} completed.')

                        # Usuwanie model_instance po zakończeniu iteracji
                        del model_instance, optimizer_configs


if __name__ == "__main__":
    #     # Ustaw zmienne środowiskowe
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Wyłącza informacje debugowania TensorFlow
    # os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Pozwala TensorFlow na dynamiczne zarządzanie pamięcią GPU

    # # Ustaw ścieżki na dysk D
    # os.environ['CUDA_CACHE_PATH'] = 'D:\\tensorflow_cache'
    # os.environ['TF_LOG_DIR'] = 'D:\\tensorflow_logs'

    # # Tworzenie nieistniejących katalogów
    # if not os.path.exists('D:\\tensorflow_cache'):
    #     os.makedirs('D:\\tensorflow_cache')
    # if not os.path.exists('D:\\tensorflow_logs'):
    #     os.makedirs('D:\\tensorflow_logs')

    experiments = Experiments()
    experiments.train_all_models()
