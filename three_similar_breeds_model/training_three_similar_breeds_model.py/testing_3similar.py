import tensorflow as tf
import numpy as np
import cv2
import os

# =============================
# CONFIG
# =============================
IMG_SIZE = (224, 224)
MODEL_PATH = "best_three_breeds_model.keras"

# IMPORTANT: must match training order
class_labels = ['Birman', 'Ragdoll', 'Siamese']  
# ⚠️ Replace with your actual folder names

# =============================
# LOAD MODEL
# =============================
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded")

# =============================
# IMAGE PREPROCESSING FUNCTION
# =============================
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)

    img = tf.keras.applications.mobilenet_v3.preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    return img

# =============================
# PREDICTION FUNCTION
# =============================
def predict_image(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    
    class_index = np.argmax(prediction)
    confidence = prediction[0][class_index]

    predicted_class = class_labels[class_index]

    print(f"🐱 Predicted Breed: {predicted_class}")
    print(f"📊 Confidence: {confidence * 100:.2f}%")

# =============================
# TEST IMAGE
# =============================
image_path = r"C:\Users\ezwan\Documents\DEGREE\FINAL YEAR PROEJCT\Data Preprocessing\similar_breeds\test\Siamese\Siamese_73.jpg"
predict_image(image_path)
