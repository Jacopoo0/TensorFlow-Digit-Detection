import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import sys
from PIL import Image as ImagePl # type:ignore

model_path = "mio_modello.keras"
model = tf.keras.models.load_model(model_path)

class_names = ["0","1","2","3","4","5","6","7","8","9"]

def preprocess_image(img_path, img_height=28, img_wide=28):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (28, 28))
    gray = tf.image.rgb_to_grayscale(img)

    plt.imshow(gray, cmap="gray")
    plt.axis("off")
    plt.show()

    print(f"shape= {gray.shape}")
    gray = tf.squeeze(gray, axis= -1)
    print(f"shape= {gray.shape}")
    gray = tf.expand_dims(gray, axis=0)

    print(f"shape= {gray.shape}")

    return gray


def make_prediction(image_path):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][predicted_class_idx]
    predicted_class = class_names[predicted_class_idx]

    return predicted_class, confidence 




for i in range(10):
    img_path = f"test/mio_numero_{i}.jpg"
    predicted_class, confidence = make_prediction(img_path)
    print(f"Classe predetta: {predicted_class} con confidenza: {confidence / 100:.2f}%")

predicted_class, confidence = make_prediction(img_path)
print(f"classe predetta: {predicted_class}, con confidenza: {confidence/100:.2f}%")