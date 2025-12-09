# app/model_utils.py

import base64
import io

import numpy as np
from PIL import Image
from tensorflow import keras

from .config import CLASS_NAMES, IMG_SIZE, MODEL_PATH


def load_model():
    """Load and return the trained Keras model."""
    print(f"Loading model from: {MODEL_PATH}")
    model = keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
    return model


def decode_base64_image(data_url: str) -> np.ndarray:
    """
    Takes a data URL from the browser (e.g. 'data:image/jpeg;base64,...')
    and returns a RGB numpy array.
    """
    # Strip header
    header, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(img)


def preprocess_image(rgb_array: np.ndarray) -> np.ndarray:
    """
    Takes an RGB image array and prepares a batch for the model.
    Assumes model has a Rescaling(1./255) layer inside.
    """
    img = Image.fromarray(rgb_array)
    img = img.resize(IMG_SIZE)
    img_array = np.asarray(img).astype("float32")

    # If your model does NOT have a Rescaling layer, uncomment:
    # img_array /= 255.0

    batch = np.expand_dims(img_array, axis=0)  # (1, H, W, 3)
    return batch


def predict_from_data_url(model, data_url: str):
    """
    Given a base64 data URL (from canvas.toDataURL), return (class_name, confidence).
    """
    rgb = decode_base64_image(data_url)
    batch = preprocess_image(rgb)
    preds = model.predict(batch, verbose=0)
    idx = int(np.argmax(preds[0]))
    cls = CLASS_NAMES[idx]
    conf = float(preds[0][idx])
    return cls, conf
