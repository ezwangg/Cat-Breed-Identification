import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from keras import models   # For loading .keras model later

# =============================
# PATH CONFIGURATION
# =============================
train_dir = r"C:\Users\ezwan\Documents\DEGREE\FINAL YEAR PROEJCT\Data Preprocessing\cat_vs_noncat_dataset\train"
val_dir = r"C:\Users\ezwan\Documents\DEGREE\FINAL YEAR PROEJCT\Data Preprocessing\cat_vs_noncat_dataset\val"
test_dir = r"C:\Users\ezwan\Documents\DEGREE\FINAL YEAR PROEJCT\Data Preprocessing\cat_vs_noncat_dataset\test"

# =============================
# DATA PREPARATION
# =============================
IMG_SIZE = (224, 224)
BATCH_SIZE = 64

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# =============================
# MODEL CREATION
# =============================
base_model = MobileNetV3Large(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze layers

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =============================
# TRAINING
# =============================
checkpoint = ModelCheckpoint('best_cat_noncat_model.keras', monitor='val_accuracy', save_best_only=True)
earlystop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=[checkpoint, earlystop]
)

# Save final trained model (optional)
model.save("final_cat_noncat_model.keras")
print("💾 Saved final model as final_cat_noncat_model.keras")

# =============================
# EVALUATION
# =============================
loss, accuracy = model.evaluate(test_generator)
print(f"✅ Test Accuracy: {accuracy * 100:.2f}%")
print(f"📉 Test Loss: {loss:.4f}")

# =============================
# CONFUSION MATRIX & REPORT
# =============================
y_pred = model.predict(test_generator)
y_pred_classes = (y_pred > 0.5).astype("int32")
y_true = test_generator.classes

cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Cat', 'Cat'],
            yticklabels=['Non-Cat', 'Cat'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix: Cat vs Non-Cat')
plt.show()

print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=['Non-Cat', 'Cat']))

# =============================
# TRAINING CURVE
# =============================
plt.figure(figsize=(10,5))
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Cat vs Non-Cat Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# =============================
# TFLITE EXPORT
# =============================
print("\n🔄 Converting .keras model to TFLite...")

# Load the saved Keras model (best checkpoint)
model = models.load_model("best_cat_noncat_model.keras")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Recommended for mobile
tflite_model = converter.convert()

# Save TFLite model
with open("best_cat_noncat_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Saved TFLite model as best_cat_noncat_model.tflite")
