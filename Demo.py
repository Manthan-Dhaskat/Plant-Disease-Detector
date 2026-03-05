import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json

# Load trained model
model = load_model("model/Plant_disease_model.h5")

# Load class names
with open('class_name.json') as f:
    class_names = json.load(f)

# Image path
img_path = r"data\Tomato_Leaf_Mold\6ad9e92c-8dff-450a-b289-5408ee37c014___Crnl_L.Mold 6953.JPG"

# Load image
img = image.load_img(img_path, target_size=(224, 224))

# Convert to array
img_array = image.img_to_array(img)

# Expand dimension (batch size = 1)
img_array = np.expand_dims(img_array, axis=0)

# Prediction (NO manual normalization needed because model already has Rescaling layer)
predictions = model.predict(img_array)

predicted_class = np.argmax(predictions[0])
confidence = np.max(predictions[0]) * 100

print("Predicted Disease:", class_names[predicted_class])
print("Confidence: {:.2f}%".format(confidence))
print("Raw predictions:", predictions)