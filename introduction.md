# 信息隐藏项目——基于降噪自编码器和CNN的抗摄屏水印提取

TERAinsert.py根据message向图片中插入水印；

Noise_Layer.py用于模拟光源失真和摩尔纹失真；

LSDmat.py根据message生成LSD的01矩阵；

all_possible_combination.py从嵌入水印的图片中获得所有消息组合；

generate_gif.py生成gif文件；

CNNextract可以提取出图片中的水印信息；

train_CNN.py训练二分类器；

SUNet/train.py用于训练SU-net；

SUNet/demo.py基于训练好的模型对图片降噪；

SUNet/process_images.py用于裁剪或拼接图片。