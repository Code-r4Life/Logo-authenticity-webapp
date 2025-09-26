from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import numpy as np

def image_checker(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0]
    return prediction


authenticity_model = load_model('models/authenticity_classifier_mobilenet.keras')
brand_model = load_model('models/brand_classifier_mobilenet.keras')

img_path = r'C:\Users\RadhaKrishna\Downloads\adidas_shoe_stitch_9717_3_600.jpg'
prediction = image_checker(img_path, authenticity_model)
if(prediction < 0.5):
    print("Prediction: Fake")
else:
    brand_array = image_checker(img_path, brand_model)
    print("Prediction: Real")
    if(np.argmax(brand_array) == 0):
        print("Brand: Adidas")
    elif(np.argmax(brand_array) == 1):
        print("Brand: Gucci")
    else:
        print("Brand: Nike")