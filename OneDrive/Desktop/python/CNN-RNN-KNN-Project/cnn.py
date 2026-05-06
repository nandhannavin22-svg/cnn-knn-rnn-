# Import libraries
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense

# -----------------------------------
# STEP 1: Load MNIST Dataset
# -----------------------------------

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# -----------------------------------
# STEP 2: Show Sample Image
# -----------------------------------

plt.imshow(x_train[0], cmap='gray')
plt.title(f"Label: {y_train[0]}")
plt.show()

# -----------------------------------
# STEP 3: Normalize Data
# -----------------------------------

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# -----------------------------------
# STEP 4: Reshape Data
# -----------------------------------

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# -----------------------------------
# STEP 5: One-Hot Encoding
# -----------------------------------

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# -----------------------------------
# STEP 6: Build CNN Model
# -----------------------------------

model = Sequential()

# Convolution Layer
model.add(
    Conv2D(
        32,
        (3,3),
        activation='relu',
        input_shape=(28,28,1)
    )
)

# Pooling Layer
model.add(MaxPooling2D((2,2)))

# Flatten Layer
model.add(Flatten())

# Hidden Layer
model.add(Dense(128, activation='relu'))

# Output Layer
model.add(Dense(10, activation='softmax'))

# -----------------------------------
# STEP 7: Compile Model
# -----------------------------------

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------------
# STEP 8: Train Model
# -----------------------------------

history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_data=(x_test, y_test)
)

# -----------------------------------
# STEP 9: Evaluate Model
# -----------------------------------

loss, accuracy = model.evaluate(x_test, y_test)

print("\nTest Accuracy:", accuracy)

# -----------------------------------
# STEP 10: Save Model
# -----------------------------------

model.save("cnn_model.h5")

print("\nModel saved as cnn_model.h5")