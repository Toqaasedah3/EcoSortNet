from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO

app = Flask(__name__)
CORS(app)

# Load your model
model = tf.keras.models.load_model("garbage_classifier_model.h5")
class_names = ['battery', 'biological', 'cardboard', 'clothes', 'glass',
               'metal', 'paper', 'plastic', 'shoes', 'trash']

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img = image.load_img(BytesIO(file.read()), target_size=(224, 224))  # ✅ FIXED
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]
    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    confidence = float(predictions[predicted_index]) * 100

    return jsonify({
        'class': predicted_class,
        'confidence': confidence
    })

if __name__ == '__main__':
    app.run(debug=True)
