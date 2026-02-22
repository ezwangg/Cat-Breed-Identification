import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input
import os

# --- Settings ---
model_path = 'best_resnet_model.h5'
IMG_SIZE = (224, 224)

# --- Load model ---
model = load_model(model_path)
print("✅ Model loaded successfully!")

# --- Class labels (adjust based on your train generator or class folder names) ---
class_labels = ['Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair', 'Egyptian_Mau',
                'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue', 'Siamese', 'Sphynx']

# --- User input ---
img_path = input("📷 Enter the full path to a cat image: ").strip()

if not os.path.exists(img_path):
    print("❌ File not found. Please check the path.")
else:
    # --- Load and preprocess image ---
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)

    # --- Predict ---
    predictions = model.predict(img_preprocessed)
    predicted_index = np.argmax(predictions)
    predicted_label = class_labels[predicted_index]
    confidence = predictions[0][predicted_index]

    # --- Display ---
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"🐱 Predicted: {predicted_label} ({confidence:.2%})")
    plt.show()
