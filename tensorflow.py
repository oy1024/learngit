"""
    主题：tensorflow 搭建神经网络：对数据分类
    Fashion Mnist 数据集
    本次代码使用tf.keras建立和训练
    编译器：VS Code
"""

#导入tensorflow， keras
import tensorflow as tf
from tensorflow import keras

import numpy as np
import  matplotlib.pyplot as plt


# 绘制测试数据的图片
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


# 绘制分类结果：正确的预测标签是蓝色的，不正确的预测标签是红色的
def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


if __name__ == '__main__':
    fashion_mnist = keras.datasets.fashion_mnist

    # 从地址上下载数据
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # 创建一个结构作为服装分类的名字
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # 显示训练数据的前25个图像

    plt.figure(figsize=(10, 10))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 训练数据
    model.fit(train_images, train_labels, epochs=5)

    # 评估准确性
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)


    #预测数据
    img = test_images[0]
    print(img.shape)

    img = (np.expand_dims(img, 0))
    print(img.shape)

    predictions_single = model.predict(img)
    print(predictions_single)

    plot_value_array(0, predictions_single, test_labels)

    print(np.argmax(predictions_single[0]))
