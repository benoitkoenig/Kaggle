from IPython.display import display
import numpy as np

from tensorflow.data import Dataset
from tensorflow.image import decode_jpeg, resize_images
from tensorflow.io import read_file
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical, get_file
import pathlib
from PIL import Image
import random

image_size = 255
batch_size = 12
num_classes = 2

def handle_image_path(img_path):
    img = read_file(img_path)
    img = decode_jpeg(img, channels=3)
    img = resize_images(img, [image_size, image_size])
    img = img/255.0
    return img

def get_all_images(folder):
    data_root = pathlib.Path(folder)
    all_image_paths = list(data_root.glob('*/*'))
    all_image_paths = [str(path) for path in list(data_root.glob('*/*'))]
    all_images = [handle_image_path(img_path) for img_path in all_image_paths]

    label_names = sorted(item.name for item in data_root.glob('*/'))
    label_to_index = dict((name, index) for index,name in enumerate(label_names))
    all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]
    return all_images, all_image_labels

def load_and_format_images_for_fitting(folder):
    all_images, all_image_labels = get_all_images(folder)

    ds = Dataset.from_tensor_slices((all_images, all_image_labels))
    ds = ds.shuffle(buffer_size=len(all_images))
    ds = ds.batch(batch_size)
    return ds

def load_and_format_images_for_prediction(folder):
    all_images, all_image_labels = get_all_images(folder)

    image_ds = Dataset.from_tensor_slices(all_images)
    ds = image_ds.batch(20)
    return ds

train_data = load_and_format_images_for_fitting("./Pictures/train")

model = Sequential()
model.add(Conv2D(batch_size, kernel_size=(3, 3), activation='relu', input_shape=(image_size, image_size, 3)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(128, activation='softmax'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss="binary_crossentropy",
    optimizer='adam',
    metrics=['accuracy'])

# all_image_paths / batch_size must be higher than epochs * steps_per_epoch, or dataset should repeat
model.fit(train_data,
    epochs=8,
    steps_per_epoch=4)

prediction_data = load_and_format_images_for_prediction("./Pictures/validate")
preds = model.predict(prediction_data, steps=1)
print(">>>>>>>>>>>>> Predictions are:")
print(preds)
