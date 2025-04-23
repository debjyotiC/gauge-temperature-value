import tensorflow as tf
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths to images and labels
IMAGES_PATH = 'images/generated/gauges/'
LABELS_CSV = 'images/generated/gauge_labels.csv'

# Image parameters
IMG_HEIGHT = 162
IMG_WIDTH = 162
CHANNELS = 1
BATCH_SIZE = 500
EPOCHS = 50

# Read labels CSV
labels_df = pd.read_csv(LABELS_CSV)
labels_df['filename'] = labels_df['filename'].apply(lambda x: os.path.join(IMAGES_PATH, x))
labels_df['label_class'] = labels_df['label_class'].astype(str)

# Data Augmentation and Normalization
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=10,
    horizontal_flip=True,
    validation_split=0.3
)

test_datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.3
)

# Train and validation generators
train_generator = train_datagen.flow_from_dataframe(
    dataframe=labels_df,
    x_col='filename',
    y_col='label_class',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    shuffle=True,
    subset='training'
)

validation_generator = test_datagen.flow_from_dataframe(
    dataframe=labels_df,
    x_col='filename',
    y_col='label_class',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    shuffle=False,
    subset='validation'
)

num_classes = len(train_generator.class_indices)


# Model Definition
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)),
    tf.keras.layers.Conv2D(4, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(8, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(12, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(train_generator, epochs=EPOCHS, validation_data=validation_generator)

# Save the model
model.save("models/gauge-classifier.keras")

# Evaluate the model
val_loss, val_acc = model.evaluate(validation_generator)
print(f"Validation Accuracy: {val_acc * 100:.2f}%")

# Plot training history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

fig, axs = plt.subplots(2, 1)
axs[0].plot(epochs, loss, '-', label='Training loss')
axs[0].plot(epochs, val_loss, 'b', label='Validation loss')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].grid(True)
axs[0].legend(loc='best')

axs[1].plot(epochs, acc, '-', label='Training acc')
axs[1].plot(epochs, val_acc, 'b', label='Validation acc')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Accuracy')
axs[1].grid(True)
axs[1].legend(loc='best')
plt.show()

# Model predictions
y_pred_probs = model.predict(validation_generator)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = validation_generator.classes
class_names = list(validation_generator.class_indices.keys())

# Classification report and confusion matrix
print(classification_report(y_true, y_pred, target_names=class_names))
cm = confusion_matrix(y_true, y_pred)

# Confusion matrix heatmap
plt.figure(figsize=(10, 8))
ax = sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title(f"Gauge Classification Accuracy: {val_acc:.2f}")
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("images/confusion_matrix.png", dpi=600)
plt.show()
