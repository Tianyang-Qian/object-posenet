'''
读取hdf5
其中RGB保存或者转换成BGR
instance 转换成 像素为 0 1 2 ..
还需要depth图片 不知道怎么处理   查看原图
'''

# 导入PIL和numpy模块
from PIL import Image
import numpy as np

# 读取一张灰度图片
img = Image.open("0000_0_depth.png").convert("L")

# 将图片转换为numpy数组
array = np.array(img)

# 获取图片的宽度和高度
width, height = array.shape

# 创建一个空集合，用于存储出现过的像素值
pixel_values = set()

# 遍历数组中的每个元素，即每个像素值
for i in range(height):
    for j in range(width):
        # 获取当前像素值
        pixel = array[j, i]
        # 将当前像素值添加到集合中
        pixel_values.add(pixel)

# 打印出现过的像素值
print(pixel_values)