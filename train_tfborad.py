import tensorflow as tf
import numpy as np

(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

log_dir = "./logdir"
tensorboard_cb = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    profile_batch='1,2'
)

model.fit(x_train, y_train, epochs=3, callbacks=[tensorboard_cb])
