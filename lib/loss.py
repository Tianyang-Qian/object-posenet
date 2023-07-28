import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from lib.knn.__init__ import KNearestNeighbor


def loss_calculation(pred_r, pred_t, pred_c, target_r, target_t, model_points, idx, obj_diameter, rot_anchors, sym_list):
    """
    Args:
        pred_t: bs x num_p x 3
        target_t: bs x num_p x 3
        pred_r: bs x num_rot x 4
        pred_c: bs x num_rot
        target_r: bs x num_point_mesh x 3  有问题
        rot_anchors: num_rot x 4# 数组就包含了60个旋转四元数，它们表示了正二十面体的对称群。
        model_points: bs x num_point_mesh x 3
        idx: bs x 1, index of object in object class list
    Return:
        loss:
    """
    # knn = KNearestNeighbor(1)
    bs, num_p, _ = pred_t.size()
    num_rot = pred_r.size()[1]
    num_point_mesh = model_points.size()[1]

    # regularization loss
    rot_anchors = torch.from_numpy(rot_anchors).float().cuda()# 60个旋转四元数
    rot_anchors = rot_anchors.unsqueeze(0).repeat(bs, 1, 1).permute(0, 2, 1)# bs,4,num_rot   下一步左乘bs x num_rot x 4
    cos_dist = torch.bmm(pred_r, rot_anchors)   # bs x num_rot x num_rot 预测的四元数与锚点四元数 批量相乘 余弦距离越接近1两个向量越相似
    loss_reg = F.threshold((torch.max(cos_dist, 2)[0] - torch.diagonal(cos_dist, dim1=1, dim2=2)), 0.001, 0)# 将余弦距离与0.001进行比较，并取较大值
    '''
    重点解释一下上面这行：
    •  首先，使用torch.max函数沿着第三个维度（num_rot）求余弦距离（cos_dist）的最大值，得到一个形状为(bs, num_rot)的张量。这个张量表示每个样本的每个预测旋转与最相似的旋转锚点之间的余弦距离。

    •  然后，使用torch.diagonal函数沿着第二个和第三个维度（num_rot）求余弦距离（cos_dist）的  对角线元素  ，得到一个形状为(bs, num_rot)的张量。这个张量表示每个样本的每个预测旋转与对应的旋转锚点之间的余弦距离。

    •  然后，将前两步得到的两个张量相减，得到一个形状为(bs, num_rot)的张量。这个张量表示每个样本的每个预测旋转与最相似的旋转锚点之间的余弦距离与对应的旋转锚点之间的余弦距离的差值。

    •  最后，使用阈值函数（F.threshold）将差值与0.001进行比较，并取较大值。如果差值小于0.001，则将其置为0，表示没有损失；如果差值大于0.001，则保留其原值，表示有损失。这样得到一个形状为(bs, num_rot)的张量，表示每个样本的每个预测旋转与最相似的旋转锚点之间的差异（loss_reg）。这个差异越小，表示预测的旋转越接近旋转锚点。
    
    '''
    loss_reg = torch.mean(loss_reg)# 差异或者说损失 损失越小 表示 预测旋转越接近旋转锚点

    # rotation loss 旋转矩阵R
    rotations = torch.cat(((1.0 - 2.0*(pred_r[:, :, 2]**2 + pred_r[:, :, 3]**2)).view(bs, num_rot, 1),\
                           (2.0*pred_r[:, :, 1]*pred_r[:, :, 2] - 2.0*pred_r[:, :, 0]*pred_r[:, :, 3]).view(bs, num_rot, 1), \
                           (2.0*pred_r[:, :, 0]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 1]*pred_r[:, :, 3]).view(bs, num_rot, 1), \
                           (2.0*pred_r[:, :, 1]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 3]*pred_r[:, :, 0]).view(bs, num_rot, 1), \
                           (1.0 - 2.0*(pred_r[:, :, 1]**2 + pred_r[:, :, 3]**2)).view(bs, num_rot, 1), \
                           (-2.0*pred_r[:, :, 0]*pred_r[:, :, 1] + 2.0*pred_r[:, :, 2]*pred_r[:, :, 3]).view(bs, num_rot, 1), \
                           (-2.0*pred_r[:, :, 0]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 1]*pred_r[:, :, 3]).view(bs, num_rot, 1), \
                           (2.0*pred_r[:, :, 0]*pred_r[:, :, 1] + 2.0*pred_r[:, :, 2]*pred_r[:, :, 3]).view(bs, num_rot, 1), \
                           (1.0 - 2.0*(pred_r[:, :, 1]**2 + pred_r[:, :, 2]**2)).view(bs, num_rot, 1)), dim=2).contiguous().view(bs*num_rot, 3, 3)

    rotations = rotations.contiguous().transpose(2, 1).contiguous()
    model_points = model_points.view(bs, 1, num_point_mesh, 3).repeat(1, num_rot, 1, 1).view(bs*num_rot, num_point_mesh, 3)
    pred_r = torch.bmm(model_points, rotations)


    '''计算ADD'''
    if idx[0].item() in sym_list:
        # 将target[0]的后两维交换顺序，并展平成一个3行n列的张量，其中n是target[0]的元素个数。这样做是为了将target[0]的每个点看作一个三维坐标

        target_r = target_r[0].transpose(1, 0).contiguous().view(3, -1)# [500,500,3]-->[3,500]
        pred_r = pred_r.permute(2, 0, 1).contiguous().view(3, -1)#[500, 500, 3]-> [3,250000]
        # inds = knn.apply(target_r.unsqueeze(0), pred_r.unsqueeze(0))
        # 挑出预测点中500个点最近的   然后赋值给target
        inds = KNearestNeighbor.apply(target_r.unsqueeze(0), pred_r.unsqueeze(0))# pred中每个点在target中最近的点的索引 [1，250000]
        target_r = torch.index_select(target_r, 1, inds.view(-1).detach() - 1)# [3,250000]还是target中的值 只不过筛选了一下
        # 回复形状
        target_r = target_r.view(3, bs*num_rot, num_point_mesh).permute(1, 2, 0).contiguous()# [500, 500, 3]
        pred_r = pred_r.view(3, bs*num_rot, num_point_mesh).permute(1, 2, 0).contiguous()# [500, 500, 3]

    dis = torch.mean(torch.norm((pred_r - target_r), dim=2), dim=1)#计算范数并取均值 是一个数
    dis = dis / obj_diameter   # normalize by diameter 归一化
    pred_c = pred_c.contiguous().view(bs*num_rot)
    loss_r = torch.mean(dis / pred_c + torch.log(pred_c), dim=0)


    # translation loss
    loss_t = F.smooth_l1_loss(pred_t, target_t, reduction='mean')
    # total loss
    loss = loss_r + 2.0 * loss_reg + 5.0 * loss_t
    # del knn
    return loss, loss_r, loss_t, loss_reg


class Loss(_Loss):
    def __init__(self, sym_list, rot_anchors):
        super(Loss, self).__init__(True)
        # super(Loss, self).__init__(reduction='mean')
        self.sym_list = sym_list
        self.rot_anchors = rot_anchors

    def forward(self, pred_r, pred_t, pred_c, target_r, target_t, model_points, idx, obj_diameter):
        """
        """
        return loss_calculation(pred_r, pred_t, pred_c, target_r, target_t, model_points, idx, obj_diameter, self.rot_anchors, self.sym_list)