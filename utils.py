#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author: YueJie time:2020/8/12

import numpy as np
import PIL.Image as I
from glob import glob
import config as C

# 读取图像数据
def read_image(batch_img_path_lst,show=False):
    print(len(batch_img_path_lst))
    images = []
    for img_path in batch_img_path_lst:
        img = I.open(img_path)
        img=np.array(img)
        images.append(img)
    images_arr=np.array(images)
    images_arr = images_arr / np.max(images_arr)
    images_arr=np.expand_dims(images_arr,-1)
    if show:
        print("图像的格式是")
        print(images_arr.shape)
    return images_arr


#每个epoch后，重置指向，否则超出数据集范围
def transform(n,L):
    if n<L:
        return n
    else:
        ret=n-int(n/L)*L
        return ret  


#用于给训练模型喂数据[批量化处理,且使用generator,从而减少内存占有量]
def generate_arrays_from_file(img_path_lst,batch_size,which_seg):
    print("生成batch数据")
    L=int(len(img_path_lst)/batch_size)
    while True:#which_seg*batch_size <len(img_path_lst):
        #print(img_path_lst)
        which_seg=transform(which_seg,L)#which_seg/(len(img_path_lst)/batch_size)
        #print("----------------------")
        #print(which_seg)
        #print("----------------------")
        this_path_lst=img_path_lst[which_seg*batch_size:(which_seg+1)*batch_size]
        #print(this_path_lst)
        this_images=read_image(this_path_lst,show=False)
        which_seg +=1
        print("count:"+str(which_seg))
        yield this_images,this_images



#用于提取模型中间层数据
def generate_arrays_from_file_2(img_path_lst,batch_size,which_seg):
    print("生成batch数据")
    L=int(len(img_path_lst)/batch_size)
    while True:#which_seg*batch_size <len(img_path_lst):
        #print(img_path_lst)
        which_seg=transform(which_seg,L)#which_seg/(len(img_path_lst)/batch_size)
        #print("----------------------")
        #print(which_seg)
        #print("----------------------")
        this_path_lst=img_path_lst[which_seg*batch_size:(which_seg+1)*batch_size]
        #print(this_path_lst)
        this_images=read_image(this_path_lst,show=False)
        which_seg +=1
        print("count:"+str(which_seg))
        yield this_path_lst,this_images,this_images



if __name__=="__main__":
    #import sys
    #num,num2=sys.argv[1:]
    #print(transform(int(num),int(num2)))
    #exit()

    data = glob(C.raw_data+"/*")#'i../raw_contour_100x100/*')[:100]
    #ret=read_image(data[:2],show=True)
    #exit()
#    import time

    g=generate_arrays_from_file(data,30,0)
    for step in range(20):
        print("step",step)
        x=next(g)

