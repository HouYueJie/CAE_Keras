#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author: YueJie time:2020/9/9
from glob import glob
from matplotlib import pyplot as plt
import pickle


# 绘制训练损失图
with open("loss","rb") as f:
    loss =pickle.load(f)
print(len(loss))
print(loss)
epochs = range(len(loss))
plt.figure()
plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.title('Training loss')
plt.savefig("loss.png")


# 验证数据预测
with open("raw","rb") as f:
    raw =pickle.load(f)
with open("pred","rb") as f:
    pred =pickle.load(f)


plt.figure(figsize = (20, 4))
print("raw Image")
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(raw[i, ..., 0], plt.cm.gray)
    plt.title("raw Image_" + str(i))
plt.savefig("raw_image.png")


plt.figure(figsize = (20, 4))
print("pred Image")
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(pred[i, ..., 0], plt.cm.gray)
    plt.title("pred Image_" + str(i))
plt.savefig("rec_image.png")



