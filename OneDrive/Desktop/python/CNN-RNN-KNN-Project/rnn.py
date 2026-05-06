# Import libraries
import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# -----------------------------------
# STEP 1: Load MNIST Dataset
# -----------------------------------

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# -----------------------------------
# STEP 2: Normalize Data
# -----------------------------------

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# -----------------------------------
# STEP 3: One-Hot Encoding
# -----------------------------------

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# -----------------------------------
# STEP 4: Build RNN Model
# -----------------------------------

model = Sequential()

# RNN Layer
model.add(
    SimpleRNN(
        128,
        input_shape=(28, 28),
        activation='relu'
    )
)

# Output Layer
model.add(Dense(10, activation='softmax'))

# -----------------------------------
# STEP 5: Compile Model
# -----------------------------------

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------------
# STEP 6: Train Model
# -----------------------------------

history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_data=(x_test, y_test)
)

# -----------------------------------
# STEP 7: Evaluate Model
# -----------------------------------

loss, accuracy = model.evaluate(x_test, y_test)

print("\nTest Accuracy:", accuracy)

# -----------------------------------
# STEP 8: Save Model
# -----------------------------------

model.save("rnn_model.h5")

print("\nModel saved as rnn_model.h5")
