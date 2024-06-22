import cv2
import numpy as np
from tensorflow.keras.models import load_model

def predict_process(path):
    classes = ["Bird-drop", "Clean", "Dusty", "Electrical-damage", "Physical-Damage", "Snow-Covered"]
    model_path = './solar.h5'
    
    # Load the model
    try:
        gru_model = load_model(model_path)
    except Exception as e:
        print("Error loading the model:", e)
        return None
    
    # Read and resize the image
    try:
        frame = cv2.imread(path)
        if frame is None:
            print("Error: Unable to load the image from", path)
            return None
        img = cv2.resize(frame, (244, 244))
    except Exception as e:
        print("Error reading or resizing the image:", e)
        return None
    
    # Preprocess the image
    try:
        img = img.reshape(1, 244, 244, 3)
    except Exception as e:
        print("Error reshaping the image:", e)
        return None
    
    # Make predictions
    try:
        predictions = gru_model.predict(img)
        prediction_index = np.argmax(predictions)
        prediction = classes[prediction_index]
        return prediction
    except Exception as e:
        print("Error making predictions:", e)
        return None
