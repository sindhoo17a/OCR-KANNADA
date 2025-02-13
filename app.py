# app.py

from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
import cv2
import numpy as np
import os
import string

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Character set
characters = list(string.ascii_letters) + list(string.digits) + [' ', '.', ',', '!', '?'] + list('ಕನ್ನಡದ')
char_to_num = {char: i for i, char in enumerate(characters)}
num_to_char = {i: char for i, char in enumerate(characters)}

# Load the trained model
model = tf.keras.models.load_model('models/kannada_ocr_model.h5')

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 32))
    img = img.reshape(1, 32, 128, 1)
    img = img / 255.0  # Normalize the image
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    img = preprocess_image(filepath)
    prediction = model.predict(img)
    text = decode_prediction(prediction)

    return jsonify({'text': text})

def decode_prediction(prediction):
    prediction = np.argmax(prediction, axis=-1)
    sentence = ''.join([num_to_char[num] for num in prediction[0] if num != char_to_num[' ']])
    return sentence

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
