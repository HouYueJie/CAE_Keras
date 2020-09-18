#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author: YueJie time:2020/8/12

import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from tensorflow.keras.layers import Conv2DTranspose
from keras import optimizers
from keras.layers.normalization import BatchNormalization




class CAE():
    def __init__(self,x,y,c):
        self.input_img = Input(shape = (x,y, c))
        self.autoencoder = Model(self.input_img, self.autoencoder(self.input_img))
        self.adm = optimizers.Adam(lr=3e-4, beta_1=0.9, beta_2=0.999,epsilon=None, decay= 0.0 , amsgrad=False)
        self.autoencoder.compile(loss = 'mean_squared_error', optimizer = self.adm)
        self.autoencoder.summary()


    def autoencoder(self,input_img):
        conv1 = Conv2D(32, (3, 3), strides=(2, 2), padding='same',\
            kernel_regularizer=keras.regularizers.l2(0.001))(input_img)   
        BatchNormalization()
        conv2 = Conv2D(16, (3, 3), strides=(2, 2), padding='same')(conv1)  
        BatchNormalization()
        conv3 = Conv2D(8, (3, 3), strides=(2, 2), padding='same')(conv2)  
        BatchNormalization()
        conv3_1 = Conv2D(2, (3, 3), strides=(2, 2), padding='same')(conv3)  
        BatchNormalization()

        conv4_1 = Conv2DTranspose(2, (3, 3), strides=(2, 2), padding='same')(conv3_1)  
        BatchNormalization()
        conv4 = Conv2DTranspose(8, (3, 3), strides=(2, 2), padding='same')(conv4_1)  
        BatchNormalization()
        conv5 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(conv4)  
        BatchNormalization()
        conv6 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(conv5) 
        BatchNormalization()
        decoded = Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(conv6)  
        return decoded


if __name__=="__main__":
    C=CAE(240,320,1)

