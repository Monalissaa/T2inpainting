import torchvision.transforms as transforms
import cv2 as cv
import os

mask_path = '/home/mona/codes/lama/datasets/afhq/test/training_mask_test/cat-all-large-mask'
mask_file = [x for x in os.listdir(mask_path) if 'mask' in x]
mask_ratio = []
transf = transforms.ToTensor()

for mask in mask_file:
    img = cv.imread(os.path.join(mask_path, mask))
    
    img_tensor = transf(img)
    mask_ratio.append(img_tensor.mean().item())

# print(mask_ratio)
# img = cv.imread('/home/mona/codes/lama/datasets/afhq/test/training_mask_test/cat-all/pixabay_cat_004833_crop000_mask000.jpg')
# print(img.shape)   # numpy数组格式为（H,W,C）

# transf = transforms.ToTensor()
# img_tensor = transf(img)  # tensor数据格式是torch(C,H,W)
# print(img_tensor.mean())


import matplotlib.pyplot as plt
import numpy as np
# import matplotlib

# 设置matplotlib正常显示中文和负号
# matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
# matplotlib.rcParams['axes.unicode_minus']=False     # 正常显示负号
# 随机生成（10000,）服从正态分布的数据
# data = np.random.randn(10000)
"""
绘制直方图
data:必选参数，绘图数据
bins:直方图的长条形数目，可选项，默认为10
normed:是否将得到的直方图向量归一化，可选项，默认为0，代表不归一化，显示频数。normed=1，表示归一化，显示频率。
facecolor:长条形的颜色
edgecolor:长条形边框的颜色
alpha:透明度
"""
plt.hist(mask_ratio, bins=10, facecolor="blue", edgecolor="black", alpha=0.7)
# 显示横轴标签
plt.xlabel("Masked Area")
# 显示纵轴标签
plt.ylabel("of samples out of 5000")
# 显示图标题
# plt.title("11")
# plt.show()
plt.savefig('/home/mona/codes/lama/datasets/afhq/test/training_mask_test/mask_test.png')