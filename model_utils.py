import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

def load_model():
    """
    Loads the pre-trained MobileNetV2 model.
    """
    # Load MobileNetV2 model trained on ImageNet data
    model = MobileNetV2(weights='imagenet')
    print("MobileNetV2 model loaded successfully.")
    return model

def prepare_image(img_path):
    """
    Prepares an image for prediction by MobileNetV2.
    """
    # MobileNetV2 expects 224x224 images
    img = image.load_img(img_path, target_size=(224, 224))
    
    # Convert to array
    x = image.img_to_array(img)
    
    # Expand dimensions (model expects a batch of images)
    x = np.expand_dims(x, axis=0)
    
    # Preprocess the input (scaling/normalization expected by MobileNetV2)
    x = preprocess_input(x)
    return x

def make_prediction(model, img_path):
    """
    Uses the model to predict the class of the image.
    """
    processed_image = prepare_image(img_path)
    preds = model.predict(processed_image)
    
    # Decode predictions to get human-readable labels
    # Returns a list of lists of tuples (class_name, class_description, score)
    decoded_preds = decode_predictions(preds, top=3)[0]
    
    results = []
    for pred in decoded_preds:
        results.append({
            "class": pred[1],
            "probability": f"{pred[2]*100:.2f}%"
        })
    
    return results
