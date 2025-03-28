import tensorflow as tf
import pandas as pd
import numpy as np
from PIL import Image, ImageFilter
import os
from sklearn.preprocessing import LabelBinarizer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Paths to images and labels
IMAGES_PATH = 'images/generated/gauges/'
LABELS_CSV = 'images/generated/labels.csv'

# Image parameters
IMG_HEIGHT = 28
IMG_WIDTH = 28
CHANNELS = 1

# Read labels CSV
labels_df = pd.read_csv(LABELS_CSV)
labels_df['filename'] = labels_df['filename'].apply(lambda x: os.path.join(IMAGES_PATH, x))


# Prepare images and labels
def load_image(image_path):
    image = Image.open(image_path)
    image = image.convert("L")  # Convert to grayscale
    image = image.filter(ImageFilter.FIND_EDGES)
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize
    return image


# Load images and labels
images = np.array([load_image(img_path) for img_path in labels_df['filename']])
images = np.expand_dims(images, axis=-1)  # Ensure shape (num_samples, 224, 224, 1)
labels = labels_df['classification_label'].values

# One-hot encode labels
label_binarizer = LabelBinarizer()
labels = label_binarizer.fit_transform(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Model definition
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(12, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(6, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()

history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Plot training history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)
fig, axs = plt.subplots(2, 1)

# Plot loss
axs[0].plot(epochs, loss, '-', label='Training loss')
axs[0].plot(epochs, val_loss, 'b', label='Validation loss')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].grid(True)
axs[0].legend(loc='best')

# Plot accuracy
axs[1].plot(epochs, acc, '-', label='Training acc')
axs[1].plot(epochs, val_acc, 'b', label='Validation acc')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Accuracy')
axs[1].grid(True)
axs[1].legend(loc='best')
plt.show()

# Model predictions
y_pred = model.predict(X_test)
y_test_labels = np.argmax(y_test, axis=1)
y_pred_labels = np.argmax(y_pred, axis=1)

# Confusion matrix and classification report
results = confusion_matrix(y_test_labels, y_pred_labels)
clf_report = classification_report(y_test_labels, y_pred_labels)
print(clf_report)

# Get class names from LabelBinarizer
class_names = label_binarizer.classes_  # Ensures the correct number of labels

# Plot confusion matrix heatmap
ax = plt.subplot()
sns.heatmap(results, annot=True, cmap='Blues', annot_kws={"size": 20}, ax=ax, fmt='g')
plt.title(f"Gauge Classification Accuracy: {test_acc:.2f}")
ax.set_xlabel('Predicted labels', fontsize=12)
ax.set_ylabel('True labels', fontsize=12)
# Ensure the number of ticks matches the classes
ax.set_xticks(np.arange(len(class_names)))
ax.set_yticks(np.arange(len(class_names)))
ax.xaxis.set_ticklabels(class_names, fontsize=10, rotation=90)
ax.yaxis.set_ticklabels(class_names, fontsize=10, rotation=0)
plt.savefig("images/confusion_matrix.png", dpi=600)
plt.show()

