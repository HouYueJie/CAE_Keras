#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author: YueJie time:2020/8/12

import numpy as np
from glob import glob
import imageio
import os
from skimage.transform import resize
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.constraints import max_norm
from tensorflow.keras.layers import Dropout, Conv2DTranspose
from keras import optimizers
from keras.layers.normalization import BatchNormalization


#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
np.set_printoptions(threshold=100000000)

# 读取图像数据
def read_image(data_):
    images = []
    for i in range(len(data_)):
        img = imageio.imread(data_[i])
        img = resize(img, output_shape = (224, 224))
        images.append(img)
    return images


data = glob('raw_contour_100x100/*')


# 转换数据格式
images = read_image(data)
images_arr = np.asarray(images)
images_arr = images_arr.astype('float32')

# 图像预处理
# 已知图像为，gray图像，像素值 0~255， 尺寸为224 * 224
images_arr = images_arr.reshape(-1, 224, 224, 1)  #  3200， 224， 224，1

# 使用图像最大像素，对图像数据就像重新的缩放
images_arr = images_arr / np.max(images_arr)

# 数据分块，便于后期模型的泛化以及防止过拟合，将数据分为： 训练集80% 和 验证集20%
train_X, valid_X, train_ground, valid_ground = train_test_split(images_arr,
                                                                images_arr,
                                                                test_size = 0.2,
                                                                random_state= 13)#random_state 确保每次都分割出同样的训练和验证集

# 构建 卷积自动编码器
batch_size = 16
epochs = 100 #数据训练循环周期
inChannel = 1
x, y = 224, 224
input_img = Input(shape = (x, y, inChannel))

#完美的复原 0.0011 -> 8192维度
# def autoencoder(input_img):
#     #input = 224x 224 x 1 (wide and thin)
#     conv1 = Conv2D(32, (3, 3), strides=(2, 2),activation='relu', padding='same')(input_img)   #28 x 28 x 32, kernel_initializer='random_uniform', kernel_regularizer=keras.regularizers.l2(0.001)
#     BatchNormalization()
#     conv2 = Conv2D(50, (3, 3), strides=(2, 2),activation='relu', padding='same')(conv1)  #14 x 14 x 64
#     BatchNormalization()
#     conv3 = Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(conv2)  #7 x 7 x 128 (small and thick)#     BatchNormalization()
#     conv3_1 = Conv2D(96, (3, 3), strides=(2, 2), activation='relu', padding='same')(conv3)  #7 x 7 x 128 (small and thick)
#     BatchNormalization()
#     conv3_2= Conv2D(128, (3, 3), strides=(2, 2), activation='relu', padding='same')(conv3_1)  #7 x 7 x 128 (small and thick)
#     BatchNormalization()
#
#
#     #decoder
#     conv4_2 = Conv2DTranspose(128, (3, 3), strides=(2, 2), activation='relu', padding='same')(conv3_2)  #7 x 7 x 128
#     BatchNormalization()
#     conv4_1 = Conv2DTranspose(96, (3, 3), strides=(2, 2), activation='relu', padding='same')(conv4_2)  #7 x 7 x 128
#     BatchNormalization()
#     conv4 = Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(conv4_1)  #7 x 7 x 128
#     BatchNormalization()
#     conv5 = Conv2DTranspose(50, (3, 3), strides=(2, 2), activation='relu', padding='same')(conv4)  # 14 x 14 x 64
#     BatchNormalization()
#     conv6 = Conv2DTranspose(32, (3, 3), strides=(2, 2), activation='relu', padding='same')(conv5)  # 14 x 14 x 64
#     BatchNormalization()
#
#     decoded = Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(conv6)  # 28 x 28 x 1
#     return decoded

#优化 
def autoencoder(input_img):
    #input = 28 x 28 x 1 (wide and thin)
    conv1 = Conv2D(16, (3, 3), strides=(2, 2), padding='same', kernel_regularizer=keras.regularizers.l2(0.001))(input_img)   #28 x 28 x 32, kernel_initializer='random_uniform', kernel_regularizer=keras.regularizers.l2(0.001)
    BatchNormalization()
    conv2 = Conv2D(32, (3, 3), strides=(2, 2), padding='same')(conv1)  #14 x 14 x 64
    BatchNormalization()
    conv3 = Conv2D(64, (3, 3), strides=(2, 2), padding='same')(conv2)  #7 x 7 x 128 (small and thick)
    BatchNormalization()
    conv3_1 = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(conv3)  #7 x 7 x 128 (small and thick)
    BatchNormalization()
    conv3_2_1= Conv2D(10, (3, 3), strides=(2, 2), padding='same')(conv3_1)  #7 x 7 x 128 (small and thick)
    BatchNormalization()
    # conv3_2= Conv2D(15, (3, 3), strides=(2, 2), padding='same')(conv3_2_1)  #7 x 7 x 128 (small and thick)
    # BatchNormalization()
    # conv3_3= Conv2D(60, (1, 1), activation='relu', padding='same')(conv3_2)  #7 x 7 x 128 (small and thick)
    # BatchNormalization()
    # conv3_4= Conv2D(30, (1, 1), activation='relu', padding='same')(conv3_3)  #7 x 7 x 128 (small and thick)
    # BatchNormalization()

    #decoder
    # conv4_4 = Conv2DTranspose(30, (3, 3), activation='relu', padding='same')(conv3_4)  #7 x 7 x 128
    # BatchNormalization()
    # conv4_3 = Conv2DTranspose(60, (3, 3), activation='relu', padding='same')(conv4_4)  #7 x 7 x 128
    # BatchNormalization()
    # conv4_2 = Conv2DTranspose(15, (3, 3), strides=(2, 2), padding='same')(conv3_2)  #7 x 7 x 128
    # BatchNormalization()
    conv4_2_1 = Conv2DTranspose(10, (3, 3), strides=(2, 2), padding='same')(conv3_2_1)  #7 x 7 x 128
    BatchNormalization()
    conv4_1 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(conv4_2_1)  #7 x 7 x 128
    BatchNormalization()
    conv4 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(conv4_1)  #7 x 7 x 128
    BatchNormalization()
    conv5 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(conv4)  # 14 x 14 x 64
    BatchNormalization()
    conv6 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(conv5)  # 14 x 14 x 64
    BatchNormalization()

    decoded = Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(conv6)  # 28 x 28 x 1
    return decoded
# 优化&搭建
autoencoder = Model(input_img, autoencoder(input_img))

adm = optimizers.Adam(lr=3e-4, beta_1=0.9, beta_2=0.999,epsilon=None, decay= 0.0 , amsgrad=False)
autoencoder.compile(loss = 'mean_squared_error', optimizer = adm)


# 使用 summary函数，打印coder_layer的图层信息。【图层中的weigth值和bias值】
autoencoder.summary()

# 训练模型
autoencoder_train = autoencoder.fit(train_X, train_ground, batch_size = batch_size, epochs=epochs,
                                    verbose = 1, validation_data = (valid_X, valid_ground))

# 绘制训练损失图
loss = autoencoder_train.history['loss']
val_loss = autoencoder_train.history['val_loss']
epochs = range(100)
#plt.figure()
#plt.plot(epochs, loss, 'bo', label = 'Training loss')
#plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
#plt.title('Training and validation loss')
#plt.legend()
#plt.show()


# 验证数据预测
pred = autoencoder.predict(valid_X)

#plt.figure(figsize = (20, 4))
#print("Test Image")
#for i in range(5):
#    plt.subplot(1, 5, i+1)
#    plt.imshow(valid_X[i, ..., 0], plt.cm.gray)
#plt.show()
#plt.figure(figsize = (20, 4))
#print("Reconstruction of Test Image")
#for i in range(5):
#    plt.subplot(1, 5, i+1)
#    plt.imshow(pred[i, ..., 0], plt.cm.gray)
#plt.show()


# 保存模型
autoencoder = autoencoder.save_weights('autoencoder.h5')  #权值保存
