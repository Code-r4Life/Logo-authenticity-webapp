from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import numpy as np

def load_model_tf(logo_path, brand_path):
    logo_model = load_model(logo_path)
    brand_model = load_model(brand_path)
    return logo_model, brand_model

def preprocess_image(model, image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0]
    return prediction

def predict_image(logo_model, brand_model, image_path):
    prediction = preprocess_image(logo_model, image_path)
    if(prediction < 0.5):
        st = "Prediction: Fake"
        return st
    else:
        brand_array = preprocess_image(brand_model, image_path)
        st = "Prediction: Real, Brand: "
        if(np.argmax(brand_array) == 0):
            return st + "Adidas"
        elif(np.argmax(brand_array) == 1):
            return st + "Gucci"
        else:
            return st + "Nike"