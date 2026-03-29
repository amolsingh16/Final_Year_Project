import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load model
model = tf.keras.models.load_model("plant_disease_model.h5")

# IMPORTANT: same preprocessing as training
test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(
    "dataset/test",
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

loss, accuracy = model.evaluate(test_data)

print("Model Accuracy:", accuracy*100, "%")