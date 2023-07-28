import random
import numpy as np
import numpy.ma as ma
import yaml
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.utils.data as data

# ('train', opt.num_points, True, opt.dataset_root, opt.noise_trans)
class PoseDataset(data.Dataset):
    def __init__(self, mode, num, add_noise, root, noise_trans):
        self.objlist = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
        self.mode = mode

        self.list_rgb = [] #存放rgb图路径
        self.list_depth = [] #存放深度图路径
        self.list_label = [] #存放语义分割mask图路径

        self.list_obj = [] #存放读取的类别编号
        self.list_rank = [] #存放读取的图片编号

        self.meta = {} #存放每个类别的元数据信息
        self.pt = {} #存放每个类别的点云信息
        self.root = root #数据集根目录

        # 作者自己加的
        self.diameter = []

        item_count = 0

        # 存储了模型的一些参数 min xyz size xyz
        dataset_config_dir = 'datasets/linemod/dataset_config'



        '''
        train  全部读取   eval 全部读取  test 只读取10行的倍数
        '''
        for item in self.objlist:
            if self.mode == 'train':#设定路径
                input_file = open('{0}/data/{1}/train.txt'.format(self.root, '%02d' % item))
            else:# test和eval都是用的test.txt!
                input_file = open('{0}/data/{1}/test.txt'.format(self.root, '%02d' % item))
            while 1:# 读取内容
                item_count += 1
                input_line = input_file.readline()
                if self.mode == 'test' and item_count % 10 != 0:# 余0则直接下一次循环 不要了/？？
                    continue
                if not input_line:
                    break
                if input_line[-1:] == '\n':
                    input_line = input_line[:-1]# 第0个--倒数第二个

                # 读取图片
                self.list_rgb.append('{0}/data/{1}/rgb/{2}.png'.format(self.root, '%02d' % item, input_line))
                self.list_depth.append('{0}/data/{1}/depth/{2}.png'.format(self.root, '%02d' % item, input_line))

                if self.mode == 'eval':
                    # eval使用的是分割后的图片 帮助训练
                    self.list_label.append('{0}/segnet_results/{1}_label/{2}_label.png'.format(self.root, '%02d' % item, input_line))
                else:
                    self.list_label.append('{0}/data/{1}/mask/{2}.png'.format(self.root, '%02d' % item, input_line))

                self.list_obj.append(item)# 几号 obj(文件夹名字)
                self.list_rank.append(int(input_line))# 读取了哪些图片（文件名字）


            # 真实信息，
            meta_file = open('{0}/data/{1}/gt.yml'.format(self.root, '%02d' % item), 'r')
            self.meta[item] = yaml.load(meta_file, Loader=yaml.FullLoader)# , Loader=yaml.FullLoader
            self.pt[item] = ply_vtx('{0}/models/obj_{1}.ply'.format(self.root, '%02d' % item))
            print('Object {0} buffer loaded'.format(item))
            meta_file.close()

            # 添加真实diameter信息
            meta_file = open('{0}/models_info.yml'.format(dataset_config_dir), 'r')
            model_info = yaml.load(meta_file, Loader=yaml.FullLoader)# 加了, Loader=yaml.FullLoader
            self.diameter.append(model_info[item]['diameter'] / 1000.0)

        self.length = len(self.list_rgb)

        #相机内参
        self.cam_cx = 325.26110
        self.cam_cy = 242.04899
        self.cam_fx = 572.41140
        self.cam_fy = 573.57043

        # xmap和ymap通常是指输入图像的坐标系
        self.xmap = np.array([[i for i in range(640)] for j in range(480)])
        self.ymap = np.array([[j for i in range(640)] for j in range(480)])
        
        self.num = num
        self.symmetry_obj_idx = [7, 8]
        self.num_pt_mesh = 500

        self.add_noise = add_noise
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)# 颜色抖动的函数，它可以随机改变图像的亮度、对比度、饱和度和色调等属性
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
        self.noise_trans = noise_trans
        # 删掉了border_list  其他基本没变

    def __getitem__(self, index):
        img = Image.open(self.list_rgb[index])# H,W,C
        depth = np.array(Image.open(self.list_depth[index]))
        label = np.array(Image.open(self.list_label[index]))# label
        obj = self.list_obj[index]
        rank = self.list_rank[index]


        if obj == 2:
            for i in range(0, len(self.meta[obj][rank])):
                if self.meta[obj][rank][i]['obj_id'] == 2:
                    meta = self.meta[obj][rank][i]
                    break
        else:
            meta = self.meta[obj][rank][0]

        # mask for both valid depth and foreground

        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
        # ma.masked_not_equal(depth, 0)返回一个掩码数组，其中所有非零元素都被标记为True，而所有零元素都被标记为False
        # ma.getmaskarray()返回一个与输入数组具有相同形状的布尔数组，其中True表示应该被掩盖的元素，False表示应该被保留的元素。

        if self.mode == 'eval':# 所以区别就是eval是单通道
            mask_label = ma.getmaskarray(ma.masked_equal(label, np.array(255)))

        else:
            mask_label = ma.getmaskarray(ma.masked_equal(label, np.array([255, 255, 255])))[:, :, 0]
            # mask_label = ma.getmaskarray(ma.masked_equal(label, np.array(255)))[:, :, 0]

        mask = mask_label * mask_depth# 最后取label和depth都为True的像素作为mask

        # 加噪声
        if self.add_noise:
            img = self.trancolor(img)
        img_masked = np.array(img)

        # 加边框
        rmin, rmax, cmin, cmax = get_bbox(meta['obj_bb'])# 从gt.yml文件中，获取最标准的box
        img_masked = img_masked[rmin:rmax, cmin:cmax, :3]# 截取 截取处包含了目标物体的图像

        cam_scale = 1000.0
        target_r = np.resize(np.array(meta['cam_R_m2c']), (3, 3))
        target_t = np.array(meta['cam_t_m2c']) / cam_scale


        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]# 索引
        # return all zero vector if there is no valid point
        if len(choose) == 0:# 若不存在box  则返回5个0
            cc = torch.LongTensor([0])
            return(cc, cc, cc, cc, cc, cc)
        # downsample points if there are too many
        if len(choose) > self.num:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        # repeat points if not enough 点云数目 不足500的话：
        else:
            choose = np.pad(choose, (0, self.num - len(choose)), 'wrap')

        depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)#二维数组
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
            add_t = add_t + np.clip(0.001*np.random.randn(cloud.shape[0], 3), -0.005, 0.005)
            cloud = np.add(cloud, add_t)


        # position target
        gt_t = target_t
        target_t = target_t - cloud
        target_t = target_t / np.linalg.norm(target_t, axis=1)[:, None]

        # rotation target
        model_points = self.pt[obj] / 1000.0
        dellist = [j for j in range(0, len(model_points))]
        dellist = random.sample(dellist, len(model_points) - self.num_pt_mesh)
        model_points = np.delete(model_points, dellist, axis=0)
        target_r = np.dot(model_points, target_r.T)# 将模型点从  原始坐标系  变换到  目标旋转矩阵所表示的坐标系  。这个变换可以理解为将模型点绕着某个轴旋转一定角度，使得它们与目标旋转矩阵对齐。

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
