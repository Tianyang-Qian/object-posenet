# 导入os模块
import os
import random

# 定义一个空列表，用来存储.png文件的名称
train_list = []
test_list = []
# 遍历当前目录下的所有文件和子目录
for root, dirs, files in os.walk("./datasets/VirtualData/data"):
    # 对每个文件进行判断
    for file in files:
        # 如果文件的扩展名是.png，就把它的名称加入到列表中
        if file.endswith(".pkl"):
            train_list.append(file)

for i in range(200):
    x = random.choice(train_list)
    train_list.remove(x)
    test_list.append(x)

# 打开一个txt文件，用写入模式
with open("./datasets/VirtualData/dataset_config/virtual_train_list.txt", "w") as f:
    # 对列表中的每个元素，写入一行到txt文件中
    for png in train_list:
        f.write(png[:6] + "\n")

# 关闭文件
f.close()


# 打开一个txt文件，用写入模式
with open("./datasets/VirtualData/dataset_config/virtual_test_list.txt", "w") as f:
    # 对列表中的每个元素，写入一行到txt文件中
    for png in test_list:
        f.write(png[:6] + "\n")

# 关闭文件
f.close()

