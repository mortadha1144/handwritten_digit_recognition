import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import os

# Construct the absolute path to the model file
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
# model_path = os.path.join(project_root, "models", "mnist_cnn.h5")
# use this model for example 2
model_path = os.path.join(project_root, "models", "mnist_cnn_example_2.h5")

custom_image_dir_path = os.path.join(project_root, "digits")

# Load the model
model = tf.keras.models.load_model(model_path)


def preprocess_custom_image(image_path):
    # Open image and convert to grayscale
    img = Image.open(image_path).convert("L")
    # Resize image to 28x28
    img = img.resize((28, 28))

    # Convert image to numpy array and normalize
    img_array = np.invert(np.array(img)) / 255.0
    # Reshape for the model (1 image, 28x28 pixels, 1 channel)
    img_array = img_array.reshape(1, 28, 28, 1)
    # Predict the image
    return img_array


# Predict custom images from [/digits] directory
image_number = 1
while os.path.isfile(os.path.join(custom_image_dir_path, f"digit{image_number}.png")):
    try:
        custom_image_path = os.path.join(
            custom_image_dir_path, f"digit{image_number}.png"
        )
        custom_image_array = preprocess_custom_image(custom_image_path)
        prediction = model.predict(custom_image_array)
        plt.imshow(custom_image_array.squeeze(), cmap=plt.cm.binary)
        plt.title(f"Predicted: {np.argmax(prediction)}")
        plt.show()
        image_number += 1
    except:
        print(f"Error processing image {image_number}")
