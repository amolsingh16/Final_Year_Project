import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

m_main = tf.keras.models.load_model("plant_disease_model.h5", compile=False)
m_focal = (
    tf.keras.models.load_model("plant_disease_model_focal_candidate.h5", compile=False)
    if os.path.exists("plant_disease_model_focal_candidate.h5")
    else m_main
)
m_orig = (
    tf.keras.models.load_model("plant_disease_model_original_backup.h5", compile=False)
    if os.path.exists("plant_disease_model_original_backup.h5")
    else None
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_data = test_datagen.flow_from_directory(
    "dataset/test",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False,
)

# Load test batches and apply horizontal Test-Time Augmentation (axis=2 is width)
X_list = [test_data[i][0] for i in range(len(test_data))]
X_orig = np.vstack(X_list)
X_hflip = X_orig[:, :, ::-1, :]

pred_focal = 0.65 * m_focal.predict(X_orig, verbose=0) + 0.35 * m_focal.predict(X_hflip, verbose=0)
pred_main = 0.65 * m_main.predict(X_orig, verbose=0) + 0.35 * m_main.predict(X_hflip, verbose=0)

if m_orig is not None:
    pred_orig = 0.65 * m_orig.predict(X_orig, verbose=0) + 0.35 * m_orig.predict(X_hflip, verbose=0)
    pred = 0.50 * pred_focal + 0.35 * pred_main + 0.15 * pred_orig
else:
    pred = 0.60 * pred_focal + 0.40 * pred_main

y_pred = np.argmax(pred, axis=1)
y_true = test_data.classes

print("\n=======================================================")
print(f"Overall Ensemble + TTA Test Accuracy: {accuracy_score(y_true, y_pred) * 100:.2f}%")
print("=======================================================\n")
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=list(test_data.class_indices.keys())))