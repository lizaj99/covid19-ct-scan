from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import tensorflow as tf
import os

app = Flask(__name__)

# Load the models
import os
from tensorflow.keras.models import load_model

model1 = load_model("/home/ubuntu/covid19-ct-scan/project_app/models/model_15e.h5")
model2 = load_model("/home/ubuntu/covid19-ct-scan/project_app/models/densenet121_best_model.h5")
model3 = load_model("/home/ubuntu/covid19-ct-scan/project_app/models/mobilenetv2.h5")
# model1 = load_model("/app/models/model_15e.h5")
# model2 = load_model("/app/models/densenet121_best_model.h5")
# model3 = load_model("/app/models/mobilenetv2.h5")

# Function to preprocess image according to model requirements
def preprocess_image(image, model_type):
    if model_type == "InceptionV3":
        # Model 1's preprocessing steps
        image = image.resize((299, 299), Image.Resampling.LANCZOS)
        image = np.array(image)
        if image.ndim == 2:
            image = np.stack((image,) * 3, axis=-1)
        elif image.shape[2] == 1:  # Single channel
            image = np.concatenate((image, image, image), axis=-1)
        image = image / 255.0
    elif model_type == "DenseNet121":
        # Model 2's preprocessing steps
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize((224, 224))
        image = np.array(image)
        image = image / 255.0
    elif model_type == "MobileNetV2":
        # Model 3's preprocessing steps
        image = image.resize((224, 224))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = np.array(image)
        image = image / 255.0
    
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        # Check if the post request has the file part
        if "file" not in request.files:
            return "No file part"
        file = request.files["file"]
        if file.filename == "":
            return "No selected file"
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join("/tmp", filename)
            file.save(filepath)
            
            # Load the image
            image = Image.open(filepath)
            
            # Get the selected model
            selected_model = request.form["model"]

            # Preprocess the image according to the selected model
            if selected_model == "InceptionV3":
                processed_image = preprocess_image(image, "InceptionV3")
                prediction = model1.predict(processed_image)
            elif selected_model == "DenseNet121":
                processed_image = preprocess_image(image, "DenseNet121")
                prediction = model2.predict(processed_image)
            elif selected_model == "MobileNetV2":
                processed_image = preprocess_image(image, "MobileNetV2")
                prediction = model3.predict(processed_image)
            if prediction[0]>0.5:
                result =  "Prediction Result: " + str(prediction[0]) + "\nCovid-19 positive"
            # Interpret the prediction
            else:
                result =  "Prediction Result: " + str(prediction[0]) + "\nCovid-19 negative"     
            return result
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
