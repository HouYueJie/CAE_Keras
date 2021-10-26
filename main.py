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

    #load_data
    raw_data_path_lst = glob.glob(C.raw_data+"/*")
    batch_data_generator=U.generate_arrays_from_file(raw_data_path_lst,C.batch_size,0)
    
    #build_model
    cae=M.CAE(C.x,C.y,C.inChannel)
    check_path=C.ckpt_save_path
    if sys.argv[1]=="train":
        if glob.glob("autoencoder.h5")!=[]:
            new_train=input("是否删除旧的权重？(yes or no)")
        else:
            new_train='no'
        while new_train not in ['yes','no']:
            new_train=input("是否删除旧的权重？(yes or no)")
        if new_train=="yes" and glob.glob(check_path+'/*')!=[]
            os.system("rm %s/*"%check_path+"/*")
        checkpointer = ModelCheckpoint(os.path.join(check_path,"model_{epoch:03d}.h5"),\
                                       verbose=1,save_weights_only=False,period=C.train_config["save_step"])
        if glob.glob(check_path+'/*')!=[]:
            print("已经存在CAE模型，继续训练！")
            path = sorted(glob.glob(check_path+'/*'),key=lambda x: int(x.split("_")[-1].split(".")[0]))[-1]
            print(path)
            print("已经存在CAE模型%s，继续训练！"%(path.split("/")[-1]))
            CAE.cae.load_weights(path)
        else:
            print("没有训练CAE模型，重新训练！")


        autoencoder_train = cae.autoencoder.fit_generator(
                                              batch_data_generator,epochs=C.epochs,
                                              steps_per_epoch=int(len(raw_data_path_lst)/C.batch_size),callbacks=[checkpointer],shuffle=True)

        loss = autoencoder_train.history['loss']
        save(loss,"result/loss")
        img=next(batch_data_generator)
        raw=img[0]
        pred = cae.autoencoder.predict(raw)
        save(raw,"result/raw")
        save(pred,"result/pred")
        
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
            

