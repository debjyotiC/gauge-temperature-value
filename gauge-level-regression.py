import tensorflow as tf
import pandas as pd
import numpy as np
from PIL import Image, ImageFilter
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Paths to images and labels
IMAGES_PATH = 'images/generated/gauges/'
LABELS_CSV = 'images/generated/gauge_labels.csv'

# Image parameters
IMG_HEIGHT = 162
IMG_WIDTH = 162
CHANNELS = 1
BATCH_SIZE = 20

# Read labels CSV
labels_df = pd.read_csv(LABELS_CSV)
labels_df['filename'] = labels_df['filename'].apply(lambda x: os.path.join(IMAGES_PATH, x))

# Prepare images and labels
def load_image(image_path):
    image = Image.open(image_path)
    image = image.convert("L")  # Convert to grayscale
    image = image.filter(ImageFilter.FIND_EDGES)
    # image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize
    return image

images = np.array([load_image(img_path) for img_path in labels_df['filename']])
labels = labels_df['label_regression'].values.reshape(-1, 1)  # Ensure labels are 2D

# Reshape images to match TensorFlow input requirements
images = images.reshape(-1, IMG_HEIGHT, IMG_WIDTH, CHANNELS)

# Scale labels to [0, 1] range
scaler = MinMaxScaler()
labels = scaler.fit_transform(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Data Augmentation
datagen_train = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    # width_shift_range=0.1,
    # height_shift_range=0.1,
    # zoom_range=0.1,
    horizontal_flip=True
)

datagen_test = tf.keras.preprocessing.image.ImageDataGenerator()

train_generator = datagen_train.flow(X_train, y_train, batch_size=BATCH_SIZE)
test_generator = datagen_test.flow(X_test, y_test, batch_size=BATCH_SIZE)

# Define Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(162, 162, CHANNELS)),

    tf.keras.layers.Conv2D(4, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(8, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(12, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])



model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])  # Mean Absolute Error as additional metric

# Print the model summary
model.summary()

model.save("models/gauge-regressor.keras")

history = model.fit(train_generator, epochs=50, validation_data=test_generator)

# Evaluate the model
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {test_mae}")

# Predictions and regression metrics
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Residual Analysis
residuals = np.abs(y_test - y_pred.flatten())
sorted_residuals = np.sort(residuals)[::-1]  # Sort residuals in descending order
cumulative_error = np.cumsum(sorted_residuals) / np.sum(sorted_residuals) * 100
data_points = np.arange(1, len(cumulative_error) + 1)

# Print regression results
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R²): {r2}")

# Plot training history
epochs = range(1, len(history.history['mae']) + 1)
fig, axs = plt.subplots(2, 1, figsize=(8, 6))

# Plot loss
axs[0].plot(epochs, history.history['loss'], '-', label='Training loss')
axs[0].plot(epochs, history.history['val_loss'], 'b', label='Validation loss')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].grid(True)
axs[0].legend(loc='best')

# Plot MAE
axs[1].plot(epochs, history.history['mae'], '-', label='Training MAE')
axs[1].plot(epochs, history.history['val_mae'], 'b', label='Validation MAE')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Mean Absolute Error (MAE)')
axs[1].grid(True)
axs[1].legend(loc='best')

plt.tight_layout()
plt.savefig("images/metrics.png", dpi=300)
# plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Ideal line
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values')
plt.grid(True)
plt.savefig("images/pred_vs_actual.png", dpi=300)
plt.show()