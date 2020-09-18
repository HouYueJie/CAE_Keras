# 项目简介

使用CAE压缩图片，得到压缩的特征。例如将400*300维度的图片压缩为600维的特征。



# 环境要求
tensorflow
keras




# 代码运行
- 将图片数据放在文件夹中，例如，raw_data，在config.py中配置相关参数


- 训练
 python3 main.py train

- 生成测试结果
 python3 main.py test
 在result中，生成 pred,raw 

- 生成中间结果
 python3 extract_dara.py

- 画图测试原图与重构图
 进入result，运行 python3 generate_image.py


# 其他
需要根据图片分辨率设置CAE格式
