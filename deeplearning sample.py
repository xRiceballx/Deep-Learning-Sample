import os
import cv2
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, optimizers, Sequential
from keras.layers import Flatten
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from keras.utils import np_utils


#以下の行はGPUを使用する際にコメントアウトを解除する
# python3.8以降ではDLLsを探さなくなったため記述を追加
os.add_dll_directory(os.path.join(os.environ['CUDA_PATH'], 'bin'))



# データセットを作成
categories = ["higher", "lower"]
imagedir = "learning images/"
X_train = []
y_train = []
training_data = []
number_of_classes = len(categories)


def create_training_data():
    for class_num, category in enumerate(categories):
        path = os.path.join(imagedir, category)
        for image_name in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, image_name), cv2.IMREAD_GRAYSCALE)  # 画像読み込み cv2.IMREAD_GRAYSCALE(グレースケールにするならこれを使う)
                img_resize_array = cv2.resize(img_array, (240, 180))  # 画像のリサイズ
                training_data.append([img_resize_array, class_num])  # 画像データ、ラベル情報を追加
            except Exception as e:
                pass


create_training_data()
random.shuffle(training_data)  # データをシャッフル

# データ一つ一つを配列に仕分け
for feature, label in training_data:
    X_train.append(feature)
    y_train.append(label)
    
# numpy配列に変換
X_train = np.array(X_train)
y_train = np.array(y_train)

X_train = X_train.astype('float32') / 255
y_train = np_utils.to_categorical(y_train, number_of_classes)

"""
# データセットの確認
for i in range(0, 4):
    print("学習データのラベル：", y_train[i])
    plt.subplot(2, 2, i+1)
    plt.axis('off')
    plt.title(label='higher' if y_train[i] == 0 else 'lower')
    plt.imshow(X_train[i], cmap='gray')

plt.show()

# データセットの保存
xy = (X_train, y_train)
np.save("datasets/histgrams.npy", xy)
"""

# 学習成果が出づらくなった場合に自動で学習を終える
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

# 空のモデルを作成
model = tf.keras.Sequential(name="my_model")

# モデルに層を追加する
image_sizeY = 180
image_sizeX = 240
input_shape = (image_sizeY, image_sizeX, 1)


model.add(tf.keras.layers.Conv2D(32,(3,3), padding='same',input_shape=input_shape))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Conv2D(32,(3,3)))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(64, (3,3), padding='same'))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Conv2D(64,(3,3)))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(2)) 
model.add(tf.keras.layers.Activation('sigmoid'))

# 層の概要（まとめ）をターミナル上に出力
model.summary()

# モデルの最適化とコンパイル
sgd = optimizers.SGD(learning_rate=0.0007, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

# モデルの学習を開始する

tmp = model.fit(X_train, y_train, batch_size=64, validation_split=0.3, epochs=100) #callbacks = [callback]

# 学習したモデルの精度、損失をグラフで表示
history = tmp.history

plt.plot(np.arange(len(history["accuracy"])), history["accuracy"], label="accuracy")
plt.plot(np.arange(len(history["val_accuracy"])), history["val_accuracy"], label="val_accuracy")
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy.png')
plt.show()

plt.plot(np.arange(len(history["loss"])),history["loss"], label="loss")
plt.plot(np.arange(len(history["val_loss"])),history["val_loss"], label="val_loss")
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss.png')
plt.show()

"""
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
"""

# モデルの保存

model.save('trained models/model05')
