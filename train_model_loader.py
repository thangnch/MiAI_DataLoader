import os
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

w, h = 600, 600


# Dung de tao toan bo du lieu va load theo batch
class Dataset:
    def __init__(self, data, label, w, h):
        # the paths of images
        self.data = np.array(data)
        # the paths of segmentation images

        # binary encode
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(label)


        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)


        self.label = onehot_encoded
        self.w = w
        self.h = h

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        print("Build model")
        # read data
        image = cv2.imread(self.data[i])
        image = cv2.resize(image, (self.w, self.h))
        label = self.label[i]
        return image, label


class Dataloader(tf.keras.utils.Sequence):
    def __init__(self, dataset, batch_size, size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.size = size

    def __getitem__(self, i):
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])

        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        return tuple(batch)

    def __len__(self):
        return self.size // self.batch_size


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



# Load data
data_folder = 'dog_CLASS/'

data = []
label = []

for folder in os.listdir(data_folder):
    for file in os.listdir(os.path.join(data_folder, folder)):
        file_path = os.path.join(data_folder, folder, file)
        data.append(file_path)
        label.append(folder)

# Fit to model
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2)

# Build dataaset
train_dataset = Dataset(X_train, y_train, w, h)
test_dataset = Dataset(X_test, y_test, w, h)

# Loader

train_loader = Dataloader(train_dataset, 8, len(train_dataset))
test_loader = Dataloader(test_dataset, 8, len(test_dataset))


model = get_model(input_size=(w, h, 3))
hist = model.fit_generator(train_loader, validation_data=test_loader, epochs=2, verbose=1)

# Save model
model.save("model.h5")
print("Finish model!")
