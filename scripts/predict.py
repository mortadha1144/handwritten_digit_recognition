import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import os

# Construct the absolute path to the model file
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
model_path = os.path.join(project_root, "models", "mnist_cnn.h5")


# Load the model
model = tf.keras.models.load_model(model_path)

# Load the test data
(_, _), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Prepare the data
X_test = X_test / 255.0
X_test = X_test[..., tf.newaxis]


# Function to predict a random sample
def predict_random_sample():
    index = random.randint(0, len(X_test) - 1)
    image = X_test[index]

    # The model expects a batch of images, so we add a dimension
    image = np.expand_dims(image, axis=0)

    # Predict the sample
    prediction = model.predict(image)
    predicted_label = np.argmax(prediction)

    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(f"Predicted: {predicted_label}, Actual: {y_test[index]}")
    plt.show()


# Predict 10 random samples
for _ in range(10):
    predict_random_sample()
