# Handwritten Digit Recognition (MNIST-like)

This project uses a deep learning model (TensorFlow/Keras) to recognize handwritten digits, similar to the MNIST dataset.

## Project Structure

- `main.py`: Main script for digit prediction.
- `mio_modello.keras`: Pre-trained Keras model for digit recognition.
- `test/`: Folder containing test images (`mio_numero_0.jpg`, ..., `mio_numero_9.jpg`, etc.).
- `requirements.txt`: Python dependencies.

## Requirements

- Python 3.x
- TensorFlow
- NumPy
- Matplotlib
- Pillow

Install dependencies with:

```sh
pip install -r requirements.txt
```

## Usage

To run digit recognition on the test images, execute:

```sh
python main.py
```

The script will:
- Load the `mio_modello.keras` model
- Preprocess each image in the `test/` folder
- Display the preprocessed image
- Predict the digit and print the predicted class with confidence

## Main Functions

- `preprocess_image(img_path)`: Preprocesses the image (resize, grayscale conversion, normalization).
- `make_prediction(image_path)`: Returns the predicted digit and confidence.

## Example Output

```
Predicted class: 0 with confidence: 0.98%
Predicted class: 1 with confidence: 0.99%
...
```

## Notes

- Test images should be in `.jpg` format and will be resized to 28x28 pixels.
- The model must be pre-trained and saved as `mio_modello.keras`.
