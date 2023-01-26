import os
import cv2
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, optimizers
import matplotlib.pyplot as plt
from keras.optimizers import SGD

#creating data set
categories = ["higher","lower"]
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
                #img_resize_array = cv2.resize(img_array, (80, 60))  # 画像のリサイズ
                training_data.append([img_array, class_num])  # 画像データ、ラベル情報を追加
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
    plt.title(label = 'higher' if y_train[i] == 0 else 'lower')
    plt.imshow(X_train[i], cmap='gray')

plt.show()

#save the dataset
"""
xy = (X_train, y_train)
np.save("datasets/histgrams.npy", xy)
"""

#create an empty model
model = keras.Sequential(name="my_model")

#add layers to model

model.add( tf.keras.layers.Conv2D(16, (5, 5), padding="same", input_shape=(480, 640), activation="relu"))
model.add( tf.keras.layers.MaxPooling2D)
model.add( tf.keras.layers.Flatten )       
model.add( tf.keras.layers.Dense(128, activation=tf.nn.relu) )   
model.add( tf.keras.layers.Dropout(0.2) )                        
model.add( tf.keras.layers.Dense(2, activation=tf.nn.softmax) ) 

# output summary of layers
model.summary()

#compile the model

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
"""
opt = SGD(learning_rate=0.01)
model.compile(loss = "categorical_crossentropy", optimizer = opt)
"""

#training the model

tmp = model.fit(X_train, y_train, validation_split=0.2, epochs=20)

#show graph of the trained model
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

#save the trained model

model.save('trained models/model02')








