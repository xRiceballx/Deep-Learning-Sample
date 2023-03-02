import os
import cv2
import random
from icrawler.builtin import BingImageCrawler
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, optimizers, Sequential
from keras.layers import Flatten
import matplotlib.pyplot as plt
from keras.optimizers import SGD

# python3.8以降ではDLLsを探さなくなったため記述を追加
os.add_dll_directory(os.path.join(os.environ['CUDA_PATH'], 'bin'))

# download cats' and dogs' images
crawler = BingImageCrawler(storage={"root_dir": "cats"})
crawler.crawl(keyword="猫", max_num=100)

crawler = BingImageCrawler(storage={"root_dir": "dogs"})
crawler.crawl(keyword="犬", max_num=100)

# creating data set
categories = ["cats", "dogs"]
imagedir = "learning images/"
X_train = []
y_train = []
training_data = []


def create_training_data():
    for class_num, category in enumerate(categories):
        path = os.path.join(imagedir, category)
        for image_name in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, image_name), cv2.IMREAD_GRAYSCALE)  # 画像読み込み
                img_resize_array = cv2.resize(img_array, (80, 60))  # 画像のリサイズ
                training_data.append([img_resize_array, class_num])  # 画像データ、ラベル情報を追加
            except Exception as e:
                pass


create_training_data()


random.shuffle(training_data)  # データをシャッフル
X_train = []  # 画像データ
y_train = []  # ラベル情報
# データセット作成
for feature, label in training_data:
    X_train.append(feature)
    y_train.append(label)
# numpy配列に変換
X_train = np.array(X_train)
y_train = np.array(y_train)


# データセットの確認
for i in range(0, 4):
    print("学習データのラベル：", y_train[i])
    plt.subplot(2, 2, i+1)
    plt.axis('off')
    plt.title(label='higher' if y_train[i] == 0 else 'lower')
    plt.imshow(X_train[i], cmap='gray')

plt.show()


# save the dataset
"""
xy = (X_train, y_train)
np.save("datasets/histgrams.npy", xy)
"""

# create an empty model
model = tf.keras.Sequential(name="my_model")

# add layers to model
image_sizeY = 60
image_sizeX = 80
input_shape = (image_sizeY, image_sizeX, 1)


model.add(tf.keras.layers.Conv2D(128, kernel_size=5, strides=1, padding="same", activation="relu", input_shape=input_shape))
model.add(tf.keras.layers.MaxPooling2D(2, strides=2))
model.add(tf.keras.layers.Conv2D(256, kernel_size=5, strides=1, padding="same", activation="relu"))
model.add(tf.keras.layers.MaxPooling2D(2, strides=2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(300, activation="relu"))
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

# output summary of layers
model.summary()

# compile the model

sgd = optimizers.SGD(learning_rate=0.01,  # lr=0.01
                     decay=1e-6,
                     momentum=0.9,
                     nesterov=True)


model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

"""
opt = SGD(learning_rate=0.01)
model.compile(loss = "categorical_crossentropy", optimizer = opt)
"""

# training the model

tmp = model.fit(X_train, y_train, batch_size=32,
                validation_split=0.5, epochs=10)

# show graph of the trained model
acc = tmp.history['accuracy']
val_acc = tmp.history['val_accuracy']
loss = tmp.history['loss']
val_loss = tmp.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.show()
plt.savefig('accuracy')

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.show()
plt.savefig('loss')

# save the trained model

model.save('trained models/model02')
