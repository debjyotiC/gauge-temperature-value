import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# Paths to images and labels
IMAGES_PATH = 'images/generated/gauges/'
LABELS_CSV = 'images/generated/labels.csv'

# Image parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
CHANNELS = 3

# Read labels CSV
labels_df = pd.read_csv(LABELS_CSV)
labels_df['filename'] = labels_df['filename'].apply(lambda x: os.path.join(IMAGES_PATH, x))


# Prepare images and labels
def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img = np.array(img) / 255.0  # Normalize to [0, 1]
    return img


images = np.array([load_image(img_path) for img_path in labels_df['filename']])
labels = labels_df['classification_label'].values

# One-hot encode labels
label_binarizer = LabelBinarizer()
labels = label_binarizer.fit_transform(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(labels.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Train the model
EPOCHS = 10
BATCH_SIZE = 32

history = model.fit(X_train, y_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")


