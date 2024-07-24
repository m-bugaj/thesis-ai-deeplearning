# import gc
# import os
# import random
# import time
# from pathlib import Path

# import matplotlib.pyplot as plt
# import numpy as np
# from tabulate import tabulate

# import tensorflow_addons as tfa  # main package

# import tensorflow_similarity as tfsim  # main package
# import tensorflow_similarity.visualization as tfsim_visualization
# import tensorflow_similarity.callbacks as tfsim_callbacks
# import tensorflow_similarity.augmenters as tfsim_augmenters
# import tensorflow_similarity.losses as tfsim_losses
# import tensorflow_similarity.architectures as tfsim_architectures

# # INFO messages are not printed.
# # This must be run before loading other modules.
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import os

import keras.datasets
import keras.datasets.cifar10
import keras.optimizer_experimental
import keras.optimizer_experimental.sgd
import keras.optimizer_v1
import keras.optimizers
# import keras.optimizers.schedules
# import keras.optimizers.schedules.learning_rate_schedule
# from keras.optimizers.schedules import LearningRateSchedule
# from keras.optimizers.schedules import ExponentialDecay
from keras.optimizer_v2 import learning_rate_schedule
# from keras.optimizers import sgd_experimental.
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
# import keras_cv
# keras
# from keras import ops
from keras import layers
from keras import regularizers
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np

# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)

AUTO = tf.data.AUTOTUNE
BATCH_SIZE = 128
EPOCHS = 5
CROP_TO = 32
SEED = 26

PROJECT_DIM = 2048
LATENT_DIM = 512
WEIGHT_DECAY = 0.0005
# keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
print(f"Total training examples: {len(x_train)}")
print(f"Total test examples: {len(x_test)}")

strength = [0.4, 0.4, 0.3, 0.1]

# random_flip = layers.RandomFlip(mode="horizontal_and_vertical")
# random_crop = layers.RandomCrop(CROP_TO, CROP_TO)
# random_brightness = layers.RandomBrightness(0.8 * strength[0])
# random_contrast = layers.RandomContrast((1 - 0.8 * strength[1], 1 + 0.8 * strength[1]))
# random_saturation = keras_cv.layers.RandomSaturation(
#     (0.5 - 0.8 * strength[2], 0.5 + 0.8 * strength[2])
# )
# random_hue = keras_cv.layers.RandomHue(0.2 * strength[3], [0,255])
# grayscale = keras_cv.layers.Grayscale()

# def flip_random_crop(image):
#     # With random crops we also apply horizontal flipping.
#     image = random_flip(image)
#     image = random_crop(image)
#     return image


# def color_jitter(x):
#     x = random_brightness(x)
#     x = random_contrast(x)
#     x = random_saturation(x)
#     x = random_hue(x)
#     # Affine transformations can disturb the natural range of
#     # RGB images, hence this is needed.
#     # x = ops.clip(x, 0, 255)
#     x = tf.clip_by_value(x, 0, 255)
#     return x


# def color_drop(x):
#     x = grayscale(x)
#     # x = ops.tile(x, [1, 1, 3])
#     x = tf.tile(x, [1, 1, 3])
#     return x




# Random flip (horizontal and vertical)
random_flip = layers.RandomFlip(mode="horizontal_and_vertical")

# Random crop
CROP_TO = 32  # Adjust based on your requirements
random_crop = layers.RandomCrop(CROP_TO, CROP_TO)

# Random brightness
def random_brightness(image, factor):
    delta = factor * tf.random.uniform([], -1, 1)
    return tf.image.adjust_brightness(image, delta)

# Random contrast
def random_contrast(image, lower, upper):
    return tf.image.random_contrast(image, lower, upper)

# Random saturation
def random_saturation(image, lower, upper):
    return tf.image.random_saturation(image, lower, upper)

# Random hue
def random_hue(image, max_delta):
    return tf.image.random_hue(image, max_delta)

# Grayscale
def grayscale(image):
    return tf.image.rgb_to_grayscale(image)

# Flip and random crop
def flip_random_crop(image):
    image = random_flip(image)
    image = random_crop(image)
    return image

# Color jitter
def color_jitter(x):
    x = random_brightness(x, 0.8 * strength[0])
    x = random_contrast(x, 1 - 0.8 * strength[1], 1 + 0.8 * strength[1])
    x = random_saturation(x, 0.5 - 0.8 * strength[2], 0.5 + 0.8 * strength[2])
    x = random_hue(x, 0.2 * strength[3])
    x = tf.clip_by_value(x, 0, 255)
    return x

# Color drop
def color_drop(x):
    x = grayscale(x)
    x = tf.tile(x, [1, 1, 3])
    return x






# def random_apply(func, x, p):
#     if keras.random.uniform([], minval=0, maxval=1) < p:
#         return func(x)
#     else:
#         return x
    
def random_apply(func, x, p):
    if tf.random.uniform([], minval=0, maxval=1) < p:
        return func(x)
    else:
        return x

def custom_augment(image):
    # As discussed in the SimCLR paper, the series of augmentation
    # transformations (except for random crops) need to be applied
    # randomly to impose translational invariance.
    image = flip_random_crop(image)
    image = random_apply(color_jitter, image, p=0.8)
    image = random_apply(color_drop, image, p=0.2)
    return image




ssl_ds_one = tf.data.Dataset.from_tensor_slices(x_train)
ssl_ds_one = (
    ssl_ds_one.shuffle(1024, seed=SEED)
    .map(custom_augment, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

ssl_ds_two = tf.data.Dataset.from_tensor_slices(x_train)
ssl_ds_two = (
    ssl_ds_two.shuffle(1024, seed=SEED)
    .map(custom_augment, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

# We then zip both of these datasets.
ssl_ds = tf.data.Dataset.zip((ssl_ds_one, ssl_ds_two))







# def create_interleaved_dataset(x_train, batch_size, seed, auto):
#     def custom_augment_fn(x):
#         return custom_augment(x)

#     def map_fn(x):
#         return tf.data.Dataset.from_tensors(x).map(custom_augment_fn, num_parallel_calls=auto)

#     dataset = tf.data.Dataset.from_tensor_slices(x_train)
#     dataset = dataset.shuffle(1024, seed=seed)
    
#     dataset = dataset.interleave(
#         map_fn,
#         cycle_length=2,
#         num_parallel_calls=auto
#     )
    
#     dataset = dataset.batch(batch_size).prefetch(auto)
#     return dataset

# # Create the dataset
# ssl_ds = create_interleaved_dataset(x_train, BATCH_SIZE, SEED, AUTO)













# # Visualize a few augmented images.
# sample_images_one = next(iter(ssl_ds_one))
# plt.figure(figsize=(10, 10))
# for n in range(25):
#     ax = plt.subplot(5, 5, n + 1)
#     plt.imshow(sample_images_one[n].numpy().astype("int"))
#     plt.axis("off")
# plt.show()

# # Ensure that the different versions of the dataset actually contain
# # identical images.
# sample_images_two = next(iter(ssl_ds_two))
# plt.figure(figsize=(10, 10))
# for n in range(25):
#     ax = plt.subplot(5, 5, n + 1)
#     plt.imshow(sample_images_two[n].numpy().astype("int"))
#     plt.axis("off")
# plt.show()

import resnet_cifar10_v2

N = 2
DEPTH = N * 9 + 2
NUM_BLOCKS = ((DEPTH - 2) // 9) - 1


def get_encoder():
    # Input and backbone.
    inputs = layers.Input((CROP_TO, CROP_TO, 3))
    x = layers.Rescaling(scale=1.0 / 127.5, offset=-1)(
        inputs
    )
    x = resnet_cifar10_v2.stem(x)
    x = resnet_cifar10_v2.learner(x, NUM_BLOCKS)
    x = layers.GlobalAveragePooling2D(name="backbone_pool")(x)

    # Projection head.
    x = layers.Dense(
        PROJECT_DIM, use_bias=False, kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dense(
        PROJECT_DIM, use_bias=False, kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
    )(x)
    outputs = layers.BatchNormalization()(x)
    return keras.Model(inputs, outputs, name="encoder")


def get_predictor():
    model = keras.Sequential(
        [
            # Note the AutoEncoder-like structure.
            layers.Input((PROJECT_DIM,)),
            layers.Dense(
                LATENT_DIM,
                use_bias=False,
                kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
            ),
            layers.ReLU(),
            layers.BatchNormalization(),
            layers.Dense(PROJECT_DIM),
        ],
        name="predictor",
    )
    return model


# def compute_loss(p, z):
#     # The authors of SimSiam emphasize the impact of
#     # the `stop_gradient` operator in the paper as it
#     # has an important role in the overall optimization.
#     # z = ops.stop_gradient(z)
#     z = tf.stop_gradient(z)
#     p = keras.utils.normalize(p, axis=1, order=2)
#     z = keras.utils.normalize(z, axis=1, order=2)
#     # Negative cosine similarity (minimizing this is
#     # equivalent to maximizing the similarity).
#     # return -ops.mean(ops.sum((p * z), axis=1))
#     return -tf.reduce_mean(tf.reduce_sum(p * z, axis=1))

def compute_loss(p, z):
    z = tf.stop_gradient(z)
    # Normalizacja przy użyciu TensorFlow
    p = tf.math.l2_normalize(p, axis=1)
    z = tf.math.l2_normalize(z, axis=1)
    # Obliczanie strat
    return -tf.reduce_mean(tf.reduce_sum(p * z, axis=1))



class SimSiam(keras.Model):
    def __init__(self, encoder, predictor):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.loss_tracker = keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]
    


    def call(self, inputs, training=False):
        ds_one, ds_two = inputs
        z1 = self.encoder(ds_one, training=training)
        z2 = self.encoder(ds_two, training=training)
        p1 = self.predictor(z1, training=training)
        p2 = self.predictor(z2, training=training)
        return z1, z2, p1, p2

    # def train_step(self, data):
    #     # Unpack the data.
    #     ds_one, ds_two = data

    #     # Forward pass through the encoder and predictor.
    #     with tf.GradientTape() as tape:
    #         z1, z2 = self.encoder(ds_one), self.encoder(ds_two)
    #         p1, p2 = self.predictor(z1), self.predictor(z2)
    #         # Note that here we are enforcing the network to match
    #         # the representations of two differently augmented batches
    #         # of data.
    #         loss = compute_loss(p1, z2) / 2 + compute_loss(p2, z1) / 2

    #     # Compute gradients and update the parameters.
    #     learnable_params = (
    #         self.encoder.trainable_variables + self.predictor.trainable_variables
    #     )
    #     gradients = tape.gradient(loss, learnable_params)
    #     self.optimizer.apply_gradients(zip(gradients, learnable_params))

    #     # Monitor loss.
    #     self.loss_tracker.update_state(loss)
    #     return {"loss": self.loss_tracker.result()}

    def train_step(self, data):
        ds_one, ds_two = data

        with tf.GradientTape() as tape:
            z1, z2 = self.encoder(ds_one), self.encoder(ds_two)
            p1, p2 = self.predictor(z1), self.predictor(z2)
            loss = compute_loss(p1, z2) / 2 + compute_loss(p2, z1) / 2

        learnable_params = self.encoder.trainable_variables + self.predictor.trainable_variables
        gradients = tape.gradient(loss, learnable_params)
        self.optimizer.apply_gradients(zip(gradients, learnable_params))

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


# Create a cosine decay learning scheduler.
num_training_samples = len(x_train)
steps = EPOCHS * (num_training_samples // BATCH_SIZE)
lr_decayed_fn = learning_rate_schedule.CosineDecay(
    initial_learning_rate=0.03, decay_steps=steps
)

# Create an early stopping callback.
early_stopping = keras.callbacks.EarlyStopping(
    monitor="loss", patience=5, restore_best_weights=True
)

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath="simsiamModel/model",
    save_weights_only=False,  # Zapisuje cały model, a nie tylko wagi
    monitor='loss',  # Metryka do monitorowania
    mode='min',  # Tryb minimalizacji (np. minimalizowanie val_loss)
    save_best_only=True  # Zapisuje tylko najlepszy model
)

# Sprawdzenie dostępnych urządzeń
print("Dostępne urządzenia:", tf.config.list_physical_devices())

# Sprawdzenie dostępności GPU
print("GPU jest dostępne:", tf.config.list_physical_devices('GPU'))

# Compile model and start training.
simsiam = SimSiam(get_encoder(), get_predictor())

# Wywołanie modelu na fikcyjnych danych, aby zbudować model
dummy_input = (tf.random.uniform((1, CROP_TO, CROP_TO, 3)), tf.random.uniform((1, CROP_TO, CROP_TO, 3)))
_ = simsiam(dummy_input)

# keras.optimizer_experimental.sgd.
simsiam.compile(optimizer=keras.optimizer_experimental.sgd.SGD(lr_decayed_fn, momentum=0.6))
history = simsiam.fit(ssl_ds, epochs=EPOCHS, callbacks=[early_stopping, model_checkpoint_callback])

# Visualize the training progress of the model.
plt.plot(history.history["loss"])
plt.grid()
plt.title("Negative Cosine Similairty")
plt.show()
