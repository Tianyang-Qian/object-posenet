import torch.utils.data as data
from PIL import Image
import os
import cv2
import os.path
import torch
import numpy as np
import torchvision.transforms as transforms
import argparse
import time
import random
import _pickle as cPickle
from lib.transformations import quaternion_from_euler, euler_matrix, random_quaternion, quaternion_matrix
import numpy.ma as ma
import copy
import scipy.misc
import scipy.io as scio

import imgaug.augmenters as iaa

import random
import numpy as np
import numpy.ma as ma
import yaml
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.utils.data as data


#init： ('train', opt.num_points, True, opt.dataset_root, opt.noise_trans)
class PoseDataset(data.Dataset):
    def __init__(self, mode, num, add_noise, root, noise_trans):
        # 场景  目前的情况是很多个场景 并且 instance id 也有很多个
        self.objlist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        self.mode = mode

        self.list_rgb = []  # 存放rgb图路径
        self.list_depth = []  # 存放深度图路径
        self.list_label = []  # 存放语义分割mask图路径

        self.list_obj = []  # 存放读取的类别编号 只有一类zigzag  先不处理  后续看怎么弄比较好
        self.list_rank = []  # 存放读取的图片编号

        self.meta = {}  # 存放每个类别的元数据信息
        self.pt = {}  # 存放每个类别的点云信息
        self.root = root  # 数据集根目录

        # 作者自己加的
        self.diameter = []

        item_count = 0

        # 存储了模型的一些参数 min xyz size xyz  但是只用了diameter
        dataset_config_dir = 'datasets/VirtualData/dataset_config'

        '''
        train  全部读取   eval 全部读取  test 只读取10行的倍数
        '''

        for item in self.objlist:
            if self.mode == 'train':  # 设定路径
                input_file = open('{0}/virtual_train_list.txt'.format(dataset_config_dir))
            else:  # test和eval都是用的test.txt!
                input_file = open('{0}/virtual_test_list.txt'.format(dataset_config_dir))

            while 1:  # 读取内容
                item_count += 1
                input_line = input_file.readline()
                if self.mode == 'test' and item_count % 10 != 0:  # 余0则直接下一次循环 不要了/？？
                    continue
                if not input_line:
                    break
                if input_line[-1:] == '\n':
                    input_line = input_line[:-1]  # 第0个--倒数第二个

                # 读取图片

                self.list_rgb.append('{0}/data/{1}_color.png'.format(self.root, input_line))
                self.list_depth.append('{0}/data/{1}_depth.png'.format(self.root, input_line))
                self.list_mask.append('{0}/data/{1}_mask.png'.format(self.root, input_line))
                with open('{0}/data/{1}_mask.png'.format(self.root, input_line), 'rb') as f:
                    label = cPickle.load(f)  # 字典
                self.list_label.append(label)


                # self.list_label.append('{0}/data/{1}_label.pkl'.format(self.root, input_line))

                # self.list_obj.append(item)  # 几号 obj(文件夹名字)
                self.list_rank.append(int(input_line))  # 读取了哪些图片（文件名字）

            # 真实信息

            # 这是点云吧
            self.pt = np.load('{0}/cad.npy'.format(dataset_config_dir))
            print('Object virtual buffer loaded')

            # 添加真实diameter信息  还没修改！！！
            self.diameter.append()
            obj_diameter = np.amax(np.linalg.norm(obj_cld, axis=1)) * 2
            self.diameter.append(obj_diameter)

        self.length = len(self.list_rgb)# 图片数

        # 相机内参
        self.cam_cx = 325.26110
        self.cam_cy = 242.04899
        self.cam_fx = 572.41140
        self.cam_fy = 573.57043

        # xmap和ymap通常是指输入图像的坐标系
        self.xmap = np.array([[i for i in range(640)] for j in range(480)])
        self.ymap = np.array([[j for i in range(640)] for j in range(480)])

        self.num = num # 点数
        # self.symmetry_obj_idx = [7, 8]  这里可能不要了
        self.num_pt_mesh = 500

        self.add_noise = add_noise
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)  # 颜色抖动的函数，它可以随机改变图像的亮度、对比度、饱和度和色调等属性
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
        self.noise_trans = noise_trans
        # 删掉了border_list  其他基本没变

    def __getitem__(self, index):
        img = Image.open(self.list_rgb[index])  # H,W,C
        depth = np.array(Image.open(self.list_depth[index]))
        mask = np.array(Image.open(self.list_mask[index]))  # label

        # obj = self.list_obj[index] 因为只有一种物品
        rank = self.list_rank[index]# 图片的名称

        label = self.list_label[index]

        # mask for both valid depth and foreground

        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
        # ma.masked_not_equal(depth, 0)返回一个掩码数组，其中所有非零元素都被标记为True，而所有零元素都被标记为False
        # ma.getmaskarray()返回一个与输入数组具有相同形状的布尔数组，其中True表示应该被掩盖的元素，False表示应该被保留的元素。


        mask_label = ma.getmaskarray(ma.masked_equal(mask, np.array([255, 255, 255])))[:, :, 0]
            # mask_label = ma.getmaskarray(ma.masked_equal(label, np.array(255)))[:, :, 0]

        mask = mask_label * mask_depth  # 最后取label和depth都为True的像素作为mask

        # 加噪声
        if self.add_noise:
            img = self.trancolor(img)
        img_masked = np.array(img)


        inst_id = label['instance_ids'][index]+1# 所有元素＋1   这是实例id
        rmin, rmax, cmin, cmax = get_bbox(label['bboxes'][inst_id])#第几个box  因为label里同时存在好几个instance
        # 加边框
        rmin, rmax, cmin, cmax = get_bbox(label['obj_bb'])  # 从gt.yml文件中，获取最标准的box
        img_masked = img_masked[rmin:rmax, cmin:cmax, :3]  # 截取 截取处包含了目标物体的图像

        cam_scale = 1000.0
        target_r = np.resize(np.array(label['cam_R_m2c']), (3, 3))
        target_t = np.array(label['cam_t_m2c']) / cam_scale

        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]  # 索引
        # return all zero vector if there is no valid point
        if len(choose) == 0:  # 若不存在box  则返回5个0
            cc = torch.LongTensor([0])
            return (cc, cc, cc, cc, cc, cc)
        # downsample points if there are too many
        if len(choose) > self.num:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        # repeat points if not enough 点云数目 不足500的话：
        else:
            choose = np.pad(choose, (0, self.num - len(choose)), 'wrap')

        depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)  # 二维数组
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        choose = np.array([choose])

        # point cloud
        pt2 = depth_masked / cam_scale
        pt0 = (xmap_masked - self.cam_cx) * pt2 / self.cam_fx
        pt1 = (ymap_masked - self.cam_cy) * pt2 / self.cam_fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)
        # 坐标

        # 加 干扰
        if self.add_noise:
            # shift
            add_t = np.random.uniform(-self.noise_trans, self.noise_trans, (1, 3))
            target_t = target_t + add_t
            # jittering
            add_t = add_t + np.clip(0.001 * np.random.randn(cloud.shape[0], 3), -0.005, 0.005)
            cloud = np.add(cloud, add_t)

        # position target
        gt_t = target_t
        target_t = target_t - cloud
        target_t = target_t / np.linalg.norm(target_t, axis=1)[:, None]

        # rotation target
        # model_points = self.pt[obj] / 1000.0
        model_points = self.pt / 1000.0
        dellist = [j for j in range(0, len(model_points))]
        dellist = random.sample(dellist, len(model_points) - self.num_pt_mesh)
        model_points = np.delete(model_points, dellist, axis=0)
        target_r = np.dot(model_points,
                          target_r.T)  # 将模型点从  原始坐标系  变换到  目标旋转矩阵所表示的坐标系  。这个变换可以理解为将模型点绕着某个轴旋转一定角度，使得它们与目标旋转矩阵对齐。

        return torch.from_numpy(cloud.astype(np.float32)), \
            torch.LongTensor(choose.astype(np.int32)), \
            self.transform(img_masked), \
            torch.from_numpy(target_t.astype(np.float32)), \
            torch.from_numpy(target_r.astype(np.float32)), \
            torch.from_numpy(model_points.astype(np.float32)), \
            torch.LongTensor([self.objlist.index(obj)]), \
            torch.from_numpy(gt_t.astype(np.float32))

    def __len__(self):
        return self.length

    def get_sym_list(self):
        return self.symmetry_obj_idx

    def get_num_points_mesh(self):
        return self.num_pt_mesh

    def get_diameter(self):
        return self.diameter

def get_bbox(bbox):
    border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
    bbx = [bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]]
    if bbx[0] < 0:
        bbx[0] = 0
    if bbx[1] >= 480:
        bbx[1] = 479
    if bbx[2] < 0:
        bbx[2] = 0
    if bbx[3] >= 640:
        bbx[3] = 639
    rmin, rmax, cmin, cmax = bbx[0], bbx[1], bbx[2], bbx[3]
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > 480:
        delt = rmax - 480
        rmax = 480
        rmin -= delt
    if cmax > 640:
        delt = cmax - 640
        cmax = 640
        cmin -= delt
    return rmin, rmax, cmin, cmax

##     #原dataset###########################################################

def ply_vtx(path):
    f = open(path)
    assert f.readline().strip() == "ply"
    f.readline()
    f.readline()
    N = int(f.readline().split()[-1])
    while f.readline().strip() != "end_header":
        continue
    pts = []
    for _ in range(N):
        pts.append(np.float32(f.readline().split()[:3]))
    return np.array(pts)


'''
就是取一个矩形
'''
def get_bbox(bbox):
    """ Compute square image crop window. """
    y1, x1, y2, x2 = bbox
    
    img_width = 1024
    img_length = 1280

    # 窗口大小
    window_size = (max(y2-y1, x2-x1) // 40 + 1) * 40
    window_size = min(window_size, 640)

    # 取中心
    center = [(y1 + y2) // 2, (x1 + x2) // 2]
    # row 极值 和 column列极值
    rmin = center[0] - int(window_size / 2)
    rmax = center[0] + int(window_size / 2)
    cmin = center[1] - int(window_size / 2)
    cmax = center[1] + int(window_size / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax


'''
对图像处理
计算范数
白点、随机处理
裁剪、采样等等
'''

class PoseDataset(data.Dataset):
    def __init__(self, mode, num_pt, add_noise, root, noise_trans):
        if mode == 'train':
            self.path = 'dataset/zigzag/dataset_config/st_real_train_list.txt'
        elif mode == 'test':
            self.path = 'dataset/zigzag/dataset_config/real_test_list.txt'
        
        self.mode = mode
        self.num_pt = num_pt
        self.root = root
        self.add_noise = add_noise
        self.noise_trans = noise_trans

        self.list = []
        # list存 real_test_list的内容
        input_file = open(self.path)
        while 1:
            input_line = input_file.readline()
            if not input_line:
                break
            if input_line[-1:] == '\n':
                input_line = input_line[:-1]
            self.list.append(input_line)
        input_file.close()

        self.length = len(self.list)

        self.cam_cx = 379.32687
        self.cam_cy = 509.43720
        self.cam_fx = 1083.09705
        self.cam_fy = 1083.09705
        
        self.cad_model = np.load('dataset/zigzag/dataset_config/cad.npy')

        self.diameter = []
        obj_center = (np.amin(self.cad_model, axis=0) + np.amax(self.cad_model, axis=0)) / 2.0
        obj_cld = self.cad_model - obj_center
        # 最大的范数平方 求多个行 向量的范数;
        obj_diameter = np.amax(np.linalg.norm(obj_cld, axis=1)) * 2
        self.diameter.append(obj_diameter)

        self.xmap = np.array([[i for i in range(1280)] for j in range(1024)])
        self.ymap = np.array([[j for i in range(1280)] for j in range(1024)])
        
        self.img_size = 192
        self.norm_scale = 1000.0
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.noise_img_loc = 0.0
        self.noise_img_scale = 7.0
        self.minimum_num_pt = 50
        self.norm = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.485, 0.485],
                                                            std=[0.229, 0.229, 0.229])])
        # self.symmetry_obj_idx = [0]
        self.symmetry_obj_idx = []
        self.num_pt_mesh = 1000
        print(len(self.list))
    
    def __getitem__(self, index):
        img = cv2.imread('{0}/{1}_color.png'.format(self.root, self.list[index]))[:, :, :3]# 只要0 1 2 三个通道  不需要透明度
        img = img[:, :, ::-1]# 倒序
        depth = np.array(cv2.imread('{0}/{1}_depth.png'.format(self.root, self.list[index]), -1))
        mask = np.array(cv2.imread('{0}/{1}_mask.png'.format(self.root, self.list[index]), -1))



        # # random dropout the pixel in mask
        # 遮盖处理   具体是黑白 有点忘了
        # 就是图上多了白点  mad写这么多
        if self.mode == 'train':
            drop_mask = np.ones(mask.shape, dtype=np.uint8) * 255  #全白
            # 将0%到5%的像素用原图大小10%到15%的黑色方块覆盖
            # 反正就是图片上多了一些黑色点
            aug = iaa.CoarseDropout((0.0, 0.05), size_percent=(0.1, 0.15))
            drop_mask = 255 - aug(images=drop_mask)  # 黑图里白点

            mask = drop_mask.astype(np.uint16) + mask.astype(np.uint16)
            mask[np.where(mask>255)] = 255
            mask = mask.astype(np.uint8)


        # 随机变化图片
        if self.add_noise:
            img = self.trancolor(Image.fromarray(np.uint8(img)))
            img = np.array(img)

        # 有些training set 数据是用.pkl 文件存储的。用module cPickle读取非常方便。
        with open('{0}/{1}_label.pkl'.format(self.root, self.list[index]), 'rb') as f:
            label = cPickle.load(f)# 字典

        # random select one object
        # train随机选择一张图片 test第一
        if self.mode == 'test':
            idx = 0
        else:        
            idx = random.randint(0, len(label['instance_ids']) - 1)

        # 正方形裁剪
        inst_id = label['instance_ids'][idx]+1# 所有元素＋1   这是个id还是  矩阵？
        rmin, rmax, cmin, cmax = get_bbox(label['bboxes'][idx])

        # sample points
        # 采样？？ 这部分没看懂
        # np.equal实现把label image每个像素的RGB值与某个class的RGB值进行比对，变成RGB bool值。
        mask = np.equal(mask, inst_id)
        # depth图中 大于0 的   和  mask  做 逻辑和
        mask = np.logical_and(mask, depth>0)

        target_r = label['rotations'][idx]
        target_t = label['translations'][idx]


        # choose是矩阵（2行）的第一组      [0]表示行   [1]表示列
        # https://blog.csdn.net/xiezhen_zheng/article/details/81326106?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522167697416516800213044030%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=167697416516800213044030&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~baidu_landing_v2~default-1-81326106-null-null.142^v73^insert_down3,201^v4^add_ask,239^v2^insert_chatgpt&utm_term=a.nonzero&spm=1018.2226.3001.4187
        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]

        # 这里不知道要干嘛  反正是扩充了choose到num_pt
        # 让前 num列为1 其他为0
        if len(choose) > self.num_pt:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num_pt] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        elif len(choose) > 0:
            # 后面填充   （前面的值）
            choose = np.pad(choose, (0, self.num_pt - len(choose)), 'wrap')
        else:
            choose = np.zeros(self.num_pt).astype(np.int32)
                
        depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]# 行变列
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]#相当于随机的取点
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
        pt2 = depth_masked / self.norm_scale
        pt0 = (xmap_masked - self.cam_cx) * pt2 / self.cam_fx
        pt1 = (ymap_masked - self.cam_cy) * pt2 / self.cam_fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)# 列拼接

        # cad model points
        model_points = self.cad_model

        # 矩阵相点积   旋转
        target = np.dot(model_points, target_r.T)
        target = np.add(target, target_t)



        # resize cropped image to standard size and adjust 'choose' accordingly
        img = img[rmin:rmax, cmin:cmax, :]
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        crop_w = rmax - rmin
        ratio = self.img_size / crop_w
        col_idx = choose % crop_w
        row_idx = choose // crop_w
        choose = (np.floor(row_idx * ratio) * self.img_size + np.floor(col_idx * ratio)).astype(np.int64)
        choose = np.array([choose])

        # data augmentation
        if self.mode == 'train':
            # point shift
            # 加上噪声
            add_t = np.array([random.uniform(-self.noise_trans, self.noise_trans) for i in range(3)])
            target_t = target_t + add_t

            # point jitter
            add_t = add_t + np.clip(0.001*np.random.randn(cloud.shape[0], 3), -0.005, 0.005)
            cloud = np.add(cloud, add_t)

        img = self.norm(img)
        cloud = cloud.astype(np.float32)

        # position target
        gt_t = target_t
        target_t = target_t - cloud
        target_t = target_t / np.linalg.norm(target_t, axis=1)[:, None]

        target_r = np.dot(model_points, target_r.T)

        return torch.from_numpy(cloud.astype(np.float32)), \
               torch.LongTensor(choose.astype(np.int32)), \
               img, \
               torch.from_numpy(target_t.astype(np.float32)), \
               torch.from_numpy(target_r.astype(np.float32)), \
               torch.from_numpy(model_points.astype(np.float32)), \
               torch.LongTensor([0]), \
               torch.from_numpy(gt_t.astype(np.float32))
    
    def __len__(self):
        return self.length
    
    def get_sym_list(self):
        return self.symmetry_obj_idx

    def get_num_points_mesh(self):
        return self.num_pt_mesh
    
    def get_diameter(self):
        return self.diameter


if __name__ == '__main__':
    dataset = PoseDataset('train', num_pt=1024, add_noise=True, root='./data_0201', noise_trans=0.01, refine=False)
    data = dataset.__getitem__(0)
    print(data[0].shape)
    print(data[1].shape)
    print(data[2].shape)
    print(data[3].shape)
    print(data[4].shape)
