import keras
import tensorflow as tf
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# Needed to set up Tensorflow, otherwise crashes for some reason
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

batch_size = 32


def readucr(filename):
    data = np.loadtxt(filename, delimiter="\t")
    y = data[:, 0]
    x = data[:, 1:]
    return x, y.astype(int)


def get_training_data():
    x_train, y_train = readucr("output.tsv")

    # Standardise the data
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

    idx = np.random.permutation(len(x_train))
    x_train = x_train[idx]
    y_train = y_train[idx]

    return x_train, y_train


def make_model(input_shape):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    drop = keras.layers.Dropout(rate=0.5)(gap)

    output_layer = keras.layers.Dense(1, activation="relu")(drop)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)


def train_model(epochs, trained):
    x_train, y_train = get_training_data()

    if trained:
        model = keras.models.load_model("best_model.h5")
    else:
        model = make_model(input_shape=x_train.shape[1:])

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            "best_model.h5", save_best_only=True, monitor="val_loss"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
    ]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error"),
        metrics=tf.keras.metrics.MeanSquaredError(name="mean_squared_error", dtype=None),
    )

    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_split=0.2,
        verbose=1,
    )


def get_fitness_function(ind):
    trained_model = keras.models.load_model("best_model.h5")

    model_prediction = trained_model.predict(ind)

    return model_prediction
