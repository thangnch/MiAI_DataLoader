import os
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


def get_model(input_size):
    # Su dung CGG16
    model_vgg16_conv = VGG16(weights='imagenet', include_top=False)

    # Dong bang cac layer
    for layer in model_vgg16_conv.layers:
        layer.trainable = False

    # Tao model
    input = Input(shape=input_size, name='image_input')
    output_vgg16_conv = model_vgg16_conv(input)

    # Them cac layer FC va Dropout
    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Dense(10, activation='softmax', name='predictions')(x)

    # Compile
    my_model = Model(inputs=input, outputs=x)
    my_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return my_model


w, h = 600, 600
# Load data
data_folder = 'dog_CLASS/'

data = []
label = []

for folder in os.listdir(data_folder):
    for file in os.listdir(os.path.join(data_folder, folder)):
        file_path = os.path.join(data_folder, folder, file)
        # Read
        image = cv2.imread(file_path)
        # Resize
        image = cv2.resize(image, dsize=(w, h))
        # Add to data
        data.append(image)
        label.append(folder)

# binary encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(label)
print(integer_encoded)

onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)
label = onehot_encoded

data = np.array(data)

model = get_model(input_size=(w, h, 3))

# Fit to model
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2)
hist = model.fit(X_train, y_train, epochs=2, validation_data=(X_test, y_test), verbose=1, batch_size=8)

# Save model
model.save("model.h5")
print("Finish model!")
