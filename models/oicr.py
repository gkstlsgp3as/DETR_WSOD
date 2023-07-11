# OICR output module
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from pdb import set_trace as pause

from torchvision.ops.boxes import box_iou

def OICR(boxes, cls_prob, im_labels, lambda_gt=0.5): ## aux_loss 넣으면 안돌아감..! 
    # boxes = (2, 100, 4)
    # cls_prob: mil_score = (2, 100, 92)
    # im_labels = (2, 92)

    im_labels    = im_labels.long()

    cls_prob     = cls_prob.clone().detach()

    # if cls_prob have the background dimenssion, we cut it out
    if cls_prob.shape[1] != im_labels.shape[1]:
        cls_prob = cls_prob[:, 1:]
    

    #  avoiding NaNs.
    eps = 1e-9
    cls_prob = cls_prob.clamp(eps, 1 - eps)

    num_images, num_classes = im_labels.shape
    
    # 2, 100 각 클래스 별로 제일 높은 score를 가지는 proposal 반환 

    gt_assignment = {}
    labels = {}
    cls_loss_weights = {}
    
    for i in range(cls_prob.shape[0]):
        if len(im_labels[i, im_labels[i]==1])==0: # when image label exists
            gt_assignment[i] = torch.zeros(100).cuda()
            labels[i] = torch.zeros(100).cuda()
            cls_loss_weights[i] = torch.zeros(100).cuda()
            continue

        max_values, max_indexes = cls_prob[i, ...].max(dim=0) # select the proposal with highest score from (k-1)th OICR
        # 92개 클래스에 대해 maximum score를 가지는 proposals 택 
        gt_boxes = boxes[i, cls_prob[i, :, im_labels[i]==1].max(dim=0)[1], :] # n, 4 <- img_label에 해당하는 클래스 중 
                                                                              # class score가 가장 큰 proposals들 filter    
        gt_classes = torch.arange(num_classes).cuda()
        gt_classes = gt_classes[im_labels[i]==1].view(-1,1) + 1 # n, 1
        gt_scores  = max_values[im_labels[i]==1].view(-1,1) # n, 1

        overlaps = box_iou(boxes[i,...], gt_boxes)  # calculate IOU -> 100, 4 vs n, 4 => 100, n
        max_overlaps, gt_assignment[i] = overlaps.max(dim=1)  # max IOU and Index => val: 100 / ind: 100

        labels[i] = gt_classes[gt_assignment[i], 0] # 100
        cls_loss_weights[i] = gt_scores[gt_assignment[i], 0] # 100
        
        bg_inds = torch.where(max_overlaps < lambda_gt)[0]
        labels[i][bg_inds] = 0
        gt_assignment[i][bg_inds] = -1

    labels = torch.stack([labels[i] for i in list(labels.keys())])
    cls_loss_weights = torch.stack([cls_loss_weights[i] for i in list(cls_loss_weights.keys())])
    gt_assignment = torch.stack([gt_assignment[i] for i in list(gt_assignment.keys())])

    return {'labels' : labels,#.reshape(1, -1),
            'cls_loss_weights' : cls_loss_weights,#.reshape(1, -1),
            'gt_assignment' : gt_assignment,#.reshape(1, -1),
            'im_labels_real' : torch.cat((torch.tensor([[1],[1]]).cuda(), im_labels), dim=1)} # 2, 93
