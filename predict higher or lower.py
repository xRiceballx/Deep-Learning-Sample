import os
import cv2
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

imgdir = "prediction/higherorlowernew"
classes = ["lower", "higher"]

model = load_model('trained models/model05')

filepath = os.path.join(imgdir)

fig, axs = plt.subplots(1, len(os.listdir(filepath)), figsize = (20,2))
plt.subplots_adjust(top=1, bottom=0)

images = []

for i, image_name in enumerate(os.listdir(filepath)):
    img_array = cv2.imread(os.path.join(filepath, image_name), cv2.IMREAD_GRAYSCALE) #画像読み込みとモノクロ化
    img_resize_array = cv2.resize(img_array, (240,180))
    img = np.expand_dims(img_resize_array, axis=0)
    data = img /255

    result = model.predict(data)
    result = round(result[0,0])

    #images.append((img_resize_array, classes[result]))


    axs[i].imshow(img_resize_array)
    axs[i].set_title(classes[result])
    axs[i].axis('off')


#plt.show()
plt.savefig("result3.png")
