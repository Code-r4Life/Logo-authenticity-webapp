from flask import Flask, render_template, request, jsonify
from utils.logo_model_loader import load_model_tf, predict_image
from flask_cors import CORS
import os

# === Setup Flask ===
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# === Load ML Models ===
logo_model, brand_model = load_model_tf("models/authenticity_classifier_mobilenet.keras", "models/brand_classifier_mobilenet.keras")

static_path = 'static/Logos/'

# === Routes ===
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    image_url = None
    if request.method == 'POST':
        image = request.files['image']
        image_path = os.path.join(static_path, image.filename)
        image.save(image_path)
        image_url = '/static/Logos/' + image.filename
        prediction = predict_image(logo_model, brand_model, image_path)
    return render_template('fake_real_styles.html', image_url=image_url, prediction=prediction)

# === Run Server ===
if __name__ == '__main__':
    app.run(debug=True)