#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author: YueJie time:2020/8/12
import glob

import config as C
import model as M
import utils as U
import pickle
from keras.models import load_model
import sys

#保存输出数据
def save(data,file_name):
    with open(file_name,"wb")as f:
        pickle.dump(data,f)


if __name__=="__main__":

# 加载数据 
    raw_data_path_lst = glob.glob(C.raw_data+"/*")#'../raw_contour_100x100/*')
    batch_data_generator=U.generate_arrays_from_file(raw_data_path_lst,C.batch_size,0)

    cae=M.CAE(C.x,C.y,C.inChannel)
    
    #cae.autoencoder = load_model('autoencoder.h5')
    if sys.argv[1]=="train":
        if glob.glob("autoencoder.h5")!=[]:
            print("已经存在模型，继续训练！")
            cae.autoencoder.load_weights('autoencoder.h5')
        else:
            print("没有训练模型，重新训练！")

        autoencoder_train = cae.autoencoder.fit_generator(
                                              batch_data_generator,epochs=C.epochs,
                                              steps_per_epoch=int(len(raw_data_path_lst)/C.batch_size),)

        loss = autoencoder_train.history['loss']
        save(loss,"result/loss")
        img=next(batch_data_generator)
        raw=img[0]
        pred = cae.autoencoder.predict(raw)
        save(raw,"result/raw")
        save(pred,"result/pred")
        cae.autoencoder.save_weights('autoencoder.h5')  #权值保存
    if sys.argv[1]=="test":
        if glob.glob("autoencoder.h5")!=[]:
            print("已经存在模型，正在生成结果！")
            cae.autoencoder.load_weights('autoencoder.h5')
            img=next(batch_data_generator)
            raw=img[0]
            pred = cae.autoencoder.predict(raw)
            save(raw,"result/raw")
            save(pred,"result/pred")

        else:
            print("没有预先训练模型，请先开始训练使用python3 main.py train")
            exit()
            

