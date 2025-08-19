import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print(f"Training data shape: {x_train.shape}")
print(f"Test data shape: {x_test.shape}")

# Reshape data to include the channel dimension (1 for grayscale)
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

# Normalize the pixel values to the range [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Convert integer labels to one-hot encoded vectors
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

print(f"\nTraining data shape after reshaping: {x_train.shape}")
print(f"First training label after one-hot encoding: {y_train[0]}")

# Define the CNN model
model = keras.Sequential([
    # First Conv-Pool Block
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    
    # Second Conv-Pool Block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Third Conv-Pool Block
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    # Flatten layer to transition from convolutional to dense layers
    layers.Flatten(),
    
    # Fully connected layers for classification
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax') # 10 units for 10 classes
])

# Display the model summary
model.summary()

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("\nStarting model training...")
# Train the model
history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

# Evaluate the model on the test dataset
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc:.4f}")

# Make a prediction on a single test image
prediction = model.predict(x_test[0:1])
predicted_class = np.argmax(prediction[0])
true_class = np.argmax(y_test[0])

print(f"\nPrediction for the first test image: {predicted_class}")
print(f"Actual digit: {true_class}")