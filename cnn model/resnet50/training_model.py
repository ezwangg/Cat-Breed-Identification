import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 🔁 Absolute paths to your dataset folders (replace with your actual paths)
train_dir = r'C:\Users\ezwan\Documents\DEGREE\FINAL YEAR PROEJCT\Data Preprocessing\cat_dataset_split\train'
val_dir = r'C:\Users\ezwan\Documents\DEGREE\FINAL YEAR PROEJCT\Data Preprocessing\cat_dataset_split\val'
test_dir = r'C:\Users\ezwan\Documents\DEGREE\FINAL YEAR PROEJCT\Data Preprocessing\cat_dataset_split\test'

# ✅ Check if folders exist
print("Train path exists:", os.path.exists(train_dir))
print("Val path exists:", os.path.exists(val_dir))
print("Test path exists:", os.path.exists(test_dir))

# # Image size expected by ResNet50 
IMG_SIZE = (224, 224)
BATCH_SIZE = 64
EPOCHS = 15

# # Data generators
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Build model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base model

# Custom top layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile model
model.compile(optimizer=Adam(learning_rate=0.00001),
              loss='categorical_crossentropy',  
              metrics=['accuracy'])

# Callbacks
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint('best_resnet_model.h5', save_best_only=True)
]

# Train model
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks
)

# =======================
# 📊 Evaluate on Test Set
# =======================
test_loss, test_acc = model.evaluate(test_generator)
print(f"\n✅ Test Accuracy: {test_acc:.4f}")
print(f"❌ Test Loss: {test_loss:.4f}")

# ================================
# 🔹 Confusion Matrix & Report
# ================================
# Get predictions
y_pred_probs = model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Classification report
print("\n📌 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

# ================================
# 🔹 Example Predictions
# ================================
def plot_example_predictions(generator, y_pred, class_labels, n=5):
    """Show n random example predictions with confidence"""
    indices = np.random.choice(len(y_pred), n, replace=False)
    images, labels = [], []

    for idx in indices:
        img_path = generator.filepaths[idx]
        true_label = class_labels[y_true[idx]]
        pred_label = class_labels[y_pred[idx]]
        confidence = np.max(y_pred_probs[idx]) * 100

        img = plt.imread(img_path)
        plt.figure()
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"True: {true_label}\nPred: {pred_label} ({confidence:.2f}%)")
        plt.show()

plot_example_predictions(test_generator, y_pred, class_labels, n=5)

# ================================
# 📈 Plot Training History
# ================================
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Val Accuracy', marker='x')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Val Loss', marker='x')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
