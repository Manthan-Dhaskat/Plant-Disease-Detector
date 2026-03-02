import tensorflow as tf
from keras import layers,models
import matplotlib.pyplot as plt
import json
from tensorflow.keras.callbacks import EarlyStopping

IMG_SIZE = 224
BATCH_SIZE = 32

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

train_dataset = tf.keras.utils.image_dataset_from_directory(
    'data',
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    'data',
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
)

class_name=train_dataset.class_names

with open("class_name.json","w") as f:
    json.dump(class_name,f)


print("class_name saved")
print(class_name)

model = models.Sequential([
    
    layers.Rescaling(1./255,input_shape=(IMG_SIZE,IMG_SIZE,3)),
    
    layers.Conv2D(32,(3,3),activation="relu"),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(64,(3,3),activation = "relu"),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(128,(3,3),activation = "relu"),
    layers.MaxPooling2D((2,2)),
    
    layers.Flatten(),
    
    layers.Dense(128,activation='relu'),
    layers.Dropout(0.3),
    
    layers.Dense(15,activation = 'softmax')
])

model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

EPOCHS = 30

history = model.fit(
    train_dataset,
    validation_data = val_dataset,
    epochs = EPOCHS,
    callbacks=[early_stop]
)

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'],label='Validation Accuracy')
plt.title("Model Accuracy")
plt.xlabel('Epochs')
plt.legend(['Train','Validation'])
plt.show()


plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'],label='Validation Loss')
plt.title("Model Loss")
plt.xlabel('Epochs')
plt.legend(['Train','Validation'])
plt.show()

model.save("model/Plant_disease_model.h5")

