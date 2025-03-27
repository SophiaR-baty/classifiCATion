import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import sys

def load_model():
    """Load the pre-trained MobileNetV2 model."""
    model = MobileNetV2(weights="imagenet")
    return model

def preprocess_image(img_path):
    """Load and preprocess the image."""
    img = image.load_img(img_path, target_size=(224, 224))  # Resize to model's expected size
    img_array = image.img_to_array(img)  # Convert to NumPy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Normalize
    return img_array

def predict_cat_probability(img_path: str) -> float:
    """Predict how likely the image contains a cat."""
    model = load_model()
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)  # Get model predictions
    decoded_predictions = decode_predictions(predictions, top=20)[0]  # Decode top 20 predictions
    print(decoded_predictions)

    # Check if "cat" appears in the top predictions
    cat_labels = ["tabby", "lynx", "cougar", "leopard", "snow_leopard", "jaguar", "lion", "tiger", "cheetah", "tiger_cat", "Persian_cat", "Siamese_cat", "Egyptian_cat"]
    cat_pobabilities = [prob for (_, label, prob) in decoded_predictions if label in cat_labels]
    cat_prob = max(cat_pobabilities) if cat_pobabilities else 0

    return float(cat_prob)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python cat_classifier.py <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    predict_cat_probability(img_path)
