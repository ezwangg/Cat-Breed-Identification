import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from keras import models   # for loading .keras before converting to TFLite

# ============================================================
# 📁 PATH CONFIGURATION
# ============================================================
train_dir = r"C:\Users\ezwan\Documents\DEGREE\FINAL YEAR PROEJCT\Data Preprocessing\cat_dataset_split\train"
val_dir   = r"C:\Users\ezwan\Documents\DEGREE\FINAL YEAR PROEJCT\Data Preprocessing\cat_dataset_split\val"
test_dir  = r"C:\Users\ezwan\Documents\DEGREE\FINAL YEAR PROEJCT\Data Preprocessing\cat_dataset_split\test"

print("Train exists:", os.path.exists(train_dir))
print("Val exists:", os.path.exists(val_dir))
print("Test exists:", os.path.exists(test_dir))

# ============================================================
# 🔧 DATA PREPARATION
# ============================================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# Use correct MobileNetV3 preprocessing:
preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_datagen   = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen  = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
)

val_generator = val_datagen.flow_from_directory(
    val_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False
)

num_classes = train_generator.num_classes
class_labels = list(train_generator.class_indices.keys())

# ============================================================
# 🧠 MODEL CREATION — MobileNetV3
# ============================================================
base_model = tf.keras.applications.MobileNetV3Large(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False  # freeze

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation="relu")(x)
output = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ============================================================
# 🚀 TRAINING
# ============================================================
callbacks = [
    EarlyStopping(patience=5, monitor="val_loss", restore_best_weights=True),
    ModelCheckpoint("best_mobilenetv3_model.keras", save_best_only=True, monitor="val_accuracy")
]

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Save final model (optional)
model.save("final_mobilenetv3_model.keras")
print("💾 Saved final model as final_mobilenetv3_model.keras")

# ============================================================
# 📊 EVALUATION
# ============================================================
test_loss, test_acc = model.evaluate(test_generator)
print(f"\n✅ Test Accuracy: {test_acc * 100:.2f}%")
print(f"📉 Test Loss: {test_loss:.4f}")

# ============================================================
# 🔍 CONFUSION MATRIX
# ============================================================
y_pred_prob = model.predict(test_generator)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = test_generator.classes

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix – 12 Breeds")
plt.show()

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

# ============================================================
# 📈 TRAINING CURVE
# ============================================================
plt.figure(figsize=(14,5))

plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.show()

# ============================================================
# 📱 TFLite CONVERSION
# ============================================================
print("\n🔄 Converting best .keras model to TFLite...")

model = models.load_model("best_mobilenetv3_model.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # good for mobile

tflite_model = converter.convert()

with open("best_mobilenetv3_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Saved TFLite model as best_mobilenetv3_model.tflite")
