import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix, classification_report


# CONFIG

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5

TRAIN_PATH = 'dataset/train'
TEST_PATH = 'dataset/test'


# DATA LOADING

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_data = test_datagen.flow_from_directory(
    TEST_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)


# MODEL (TRANSFER LEARNING)

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()


# TRAIN OR LOAD MODEL

if os.path.exists("model.h5"):
    print("Model already trained. Loading model...")
    model = tf.keras.models.load_model("model.h5")
else:
    print("Training model...")
    history = model.fit(
        train_data,
        epochs=EPOCHS,
        validation_data=test_data
    )

    model.save("model.h5")

    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title("Accuracy Graph")
    plt.show()


# CONFUSION MATRIX

y_pred = model.predict(test_data)
y_pred = (y_pred > 0.5).astype(int)

cm = confusion_matrix(test_data.classes, y_pred)

sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.show()

print("\nClassification Report:\n")
print(classification_report(test_data.classes, y_pred))


# PREDICTION FUNCTION

def predict_image(img_path, model):
    img = cv2.imread(img_path)

    if img is None:
        print("❌ Image not found. Check path!")
        return

    img = cv2.resize(img, (224,224))
    img_array = np.expand_dims(img/255.0, axis=0)

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        print("Prediction: Pneumonia 🫁")
    else:
        print("Prediction: Normal ✅")


# GRAD-CAM FUNCTION

def generate_gradcam(img_path, model):
    img = cv2.imread(img_path)

    if img is None:
        print("❌ Image not found. Check path!")
        return

    img = cv2.resize(img, (224,224))
    img_array = np.expand_dims(img/255.0, axis=0)

    last_conv_layer = model.get_layer("conv5_block3_out")

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

    heatmap = cv2.resize(heatmap, (224,224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * 0.4 + img

    plt.imshow(superimposed_img.astype("uint8"))
    plt.axis('off')
    plt.title("Grad-CAM Output")
    plt.show()


# TEST NEW IMAGE


sample_image = "my_image.jpg"
predict_image(sample_image, model)
generate_gradcam(sample_image, model)