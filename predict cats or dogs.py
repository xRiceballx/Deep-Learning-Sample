import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

imgdir = "prediction/catsanddogs"
classes = ["a dog", "a cat"]

model = load_model('trained models/catsanddogsmodel')

filepath = os.path.join(imgdir)

fig, axs = plt.subplots(1, len(os.listdir(filepath)))

for i, image_name in enumerate(os.listdir(filepath)):
    img_array = cv2.imread(os.path.join(filepath, image_name), cv2.IMREAD_GRAYSCALE) #画像読み込みとモノクロ化
    img_resize_array = cv2.resize(img_array, (128,128))
    img = np.expand_dims(img_resize_array, axis=0)
    data = img /255

    result = model.predict(data)
    result = round(result[0,0])

    axs[i].imshow(img_resize_array)
    axs[i].set_title(classes[result])
    axs[i].axis('off')

plt.show()
