# train_ocr_model.py

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Dense, LSTM, Bidirectional
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import string

# Character set
characters = list(string.ascii_letters) + list(string.digits) + [' ', '.', ',', '!', '?'] + list('ಕನ್ನಡದ')  # Add Kannada characters
char_to_num = {char: i for i, char in enumerate(characters)}
num_to_char = {i: char for i, char in enumerate(characters)}

# Define the model
def create_model():
    input_img = Input(shape=(32, 128, 1), name='image')
    x = Conv2D(32, (3, 3), activation='relu')(input_img)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Reshape((8, 512))(x)  # Adjust according to the image size
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dense(len(characters), activation='softmax')(x)

    model = Model(inputs=input_img, outputs=x)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load dataset
def load_data(data_path):
    images = []
    labels = []
    label_files = os.listdir(os.path.join(data_path, 'labels'))
    for label_file in label_files:
        if label_file.endswith('.txt'):
            with open(os.path.join(data_path, 'labels', label_file), 'r', encoding='utf-8') as f:
                label = f.read().strip()
            labels.append([char_to_num[char] for char in label])
            image_file = label_file.replace('.txt', '.png')
            img = cv2.imread(os.path.join(data_path, 'images', image_file), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (128, 32))
            images.append(img)
    images = np.array(images).reshape(-1, 32, 128, 1) / 255.0
    labels = pad_sequences(labels, padding='post', value=char_to_num[' '])
    labels = np.expand_dims(labels, -1)
    return images, labels

# Training
def train_model(train_path, test_path):
    train_images, train_labels = load_data(train_path)
    test_images, test_labels = load_data(test_path)
    model = create_model()
    model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(test_images, test_labels))
    model.save('models/kannada_ocr_model.h5')

if __name__ == '__main__':
    train_path = 'dataset/train'
    test_path = 'dataset/test'
    train_model(train_path, test_path)
