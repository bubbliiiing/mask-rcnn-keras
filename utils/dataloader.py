import math
import os.path as osp

import numpy as np
from keras.utils import Sequence
from PIL import Image

from utils.utils import (_resize, compose_image_meta, cvtColor,
                         letterbox_image, letterbox_mask, preprocess_input)


def minimize_mask(bbox, mask, mini_shape):
    #------------------------------#
    #   对mask再次进行缩小
    #   进行裁剪且resize
    #------------------------------#
    mini_mask = np.zeros(mini_shape + (mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        y1, x1, y2, x2  = bbox[i][:4]
        
        m               = mask[:, :, i].astype(bool)
        m               = m[y1:y2, x1:x2]
        if m.size == 0:
            raise Exception("Invalid bounding box with area of zero")
        
        #------------------------------#
        #   Resize
        #------------------------------#
        m = _resize(m, mini_shape)
        mini_mask[:, :, i] = np.around(m).astype(np.bool)
    return mini_mask

#----------------------------------------------------------#
#   损失函数公式们
#----------------------------------------------------------#
def load_image_gt(image, mask, boxes, class_ids, image_id, config, use_mini_mask=False):
    #------------------------------#
    #   原始shape
    #------------------------------#
    original_shape = image.shape
    #------------------------------#
    #   对图片和mask进行填充
    #------------------------------#
    image, window, scale, padding, crop = letterbox_image(image, max_dim=config.IMAGE_MAX_DIM)
    mask                                = letterbox_mask(mask, scale, padding, crop)
    
    #------------------------------#
    #   检漏，防止resize后目标消失
    #------------------------------#
    _idx = np.sum(mask, axis=(0, 1)) > 0
    mask        = mask[:, :, _idx]
    boxes       = boxes[_idx]
    class_ids   = class_ids[_idx]
    
    #------------------------------#
    #   对mask再次进行缩小
    #------------------------------#
    if use_mini_mask:
        mask = minimize_mask(boxes, mask, config.MINI_MASK_SHAPE)

    #------------------------------#
    #   生成Image_meta
    #------------------------------#
    active_class_ids = np.zeros(config.NUM_CLASSES)
    active_class_ids[0] = 1
    for id in class_ids:
        active_class_ids[id] = 1
    image_meta = compose_image_meta(image_id, original_shape, image.shape,
                                    window, scale, active_class_ids)

    return image, image_meta, class_ids, boxes, mask

def compute_iou(box, boxes, box_area, boxes_area):
    #------------------------------#
    #   单独的重合度计算
    #------------------------------#
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou

def compute_overlaps(boxes1, boxes2):
    #------------------------------#
    #   重合度计算
    #------------------------------#
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps

def build_rpn_targets(image_shape, anchors, gt_class_ids, gt_boxes, config):
    #------------------------------#
    #   rpn_match中
    #   1代表正样本、-1代表负样本
    #   0代表忽略
    #------------------------------#
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    #-----------------------------------------------#
    #   创建该部分内容利用先验框和真实框进行编码
    #-----------------------------------------------#
    rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))

    '''
    iscrowd=0的时候，表示这是一个单独的物体，轮廓用Polygon(多边形的点)表示，
    iscrowd=1的时候表示两个没有分开的物体，轮廓用RLE编码表示，比如说一张图片里面有三个人，
    一个人单独站一边，另外两个搂在一起（标注的时候距离太近分不开了），这个时候，
    单独的那个人的注释里面的iscrowing=0,segmentation用Polygon表示，
    而另外两个用放在同一个anatation的数组里面用一个segmention的RLE编码形式表示
    '''
    crowd_ix = np.where(gt_class_ids < 0)[0]
    if crowd_ix.shape[0] > 0:
        non_crowd_ix    = np.where(gt_class_ids > 0)[0]
        crowd_boxes     = gt_boxes[crowd_ix]
        gt_class_ids    = gt_class_ids[non_crowd_ix]
        gt_boxes        = gt_boxes[non_crowd_ix]
        crowd_overlaps  = compute_overlaps(anchors, crowd_boxes)
        crowd_iou_max   = np.amax(crowd_overlaps, axis=1)
        no_crowd_bool   = (crowd_iou_max < 0.001)
    else:
        no_crowd_bool   = np.ones([anchors.shape[0]], dtype=bool)

    #-----------------------------------------------#
    #   计算先验框和真实框的重合程度 
    #   [num_anchors, num_gt_boxes]
    #-----------------------------------------------#
    overlaps = compute_overlaps(anchors, gt_boxes)

    #-----------------------------------------------#
    #   1. 重合程度小于0.3则代表为负样本
    #-----------------------------------------------#
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = -1
    #-----------------------------------------------#
    #   2. 每个真实框重合度最大的先验框是正样本
    #-----------------------------------------------#
    gt_iou_argmax = np.argwhere(overlaps == np.max(overlaps, axis=0))[:,0]
    rpn_match[gt_iou_argmax] = 1
    #-----------------------------------------------#
    #   3. 重合度大于0.7则代表为正样本
    #-----------------------------------------------#
    rpn_match[anchor_iou_max >= 0.7] = 1

    #-----------------------------------------------#
    #   正负样本平衡
    #   找到正样本的索引
    #-----------------------------------------------#
    ids = np.where(rpn_match == 1)[0]
    
    #-----------------------------------------------#
    #   如果大于(config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)则删掉一些
    #-----------------------------------------------#
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
    if extra > 0:
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
        
    #-----------------------------------------------#
    #   找到负样本的索引
    #-----------------------------------------------#
    ids = np.where(rpn_match == -1)[0]
    
    #-----------------------------------------------#
    #   使得总数为config.RPN_TRAIN_ANCHORS_PER_IMAGE
    #-----------------------------------------------#
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE -
                        np.sum(rpn_match == 1))
    if extra > 0:
        # Rest the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0

    #-----------------------------------------------#
    #   找到内部真实存在物体的先验框，进行编码
    #-----------------------------------------------#
    ids = np.where(rpn_match == 1)[0]
    ix = 0 
    for i, a in zip(ids, anchors[ids]):
        gt = gt_boxes[anchor_iou_argmax[i]]
        #-----------------------------------------------#
        #   计算真实框的中心，高宽
        #-----------------------------------------------#
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w
        #-----------------------------------------------#
        #   计算先验框中心，高宽
        #-----------------------------------------------#
        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w
        #-----------------------------------------------#
        #   编码运算
        #-----------------------------------------------#
        rpn_bbox[ix] = [
            (gt_center_y - a_center_y) / np.maximum(a_h, 1),
            (gt_center_x - a_center_x) / np.maximum(a_w, 1),
            np.log(np.maximum(gt_h / np.maximum(a_h, 1), 1e-5)),
            np.log(np.maximum(gt_w / np.maximum(a_w, 1), 1e-5)),
        ]
        #-----------------------------------------------#
        #   改变数量级
        #-----------------------------------------------#
        rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV
        ix += 1
    return rpn_match, rpn_bbox

class COCODetection(Sequence):
    def __init__(self, image_path, coco, num_classes, anchors, batch_size, config, COCO_LABEL_MAP={}, augmentation=None):
        self.image_path     = image_path

        self.coco           = coco
        self.ids            = list(self.coco.imgToAnns.keys())

        self.num_classes    = num_classes
        self.anchors        = anchors
        self.batch_size     = batch_size
        self.config         = config

        self.augmentation   = augmentation

        self.label_map      = COCO_LABEL_MAP
        self.length         = len(self.ids)

    def __getitem__(self, index):
        for i, global_index in enumerate(range(index * self.batch_size, (index + 1) * self.batch_size)):  
            global_index = global_index % self.length

            image, boxes, mask_gt, num_crowds, image_id = self.pull_item(global_index)
            #------------------------------#
            #   获得种类
            #------------------------------#
            class_ids   = boxes[:,  -1]
            #------------------------------#
            #   获得框的坐标
            #------------------------------#
            boxes       = boxes[:, :-1]

            image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
            load_image_gt(image, mask_gt, boxes, class_ids, image_id, self.config, use_mini_mask=self.config.USE_MINI_MASK)

            if not np.any(gt_class_ids > 0):
                continue
        
            # RPN Targets
            rpn_match, rpn_bbox = build_rpn_targets(image.shape, self.anchors, gt_class_ids, gt_boxes, self.config)
        
            #-----------------------------------------------------------------------#
            #   如果某张图片里面物体的数量大于最大值的话，则进行筛选，防止过大
            #-----------------------------------------------------------------------#
            if gt_boxes.shape[0] > self.config.MAX_GT_INSTANCES:
                ids = np.random.choice(
                    np.arange(gt_boxes.shape[0]), self.config.MAX_GT_INSTANCES, replace=False)
                gt_class_ids = gt_class_ids[ids]
                gt_boxes = gt_boxes[ids]
                gt_masks = gt_masks[:, :, ids]

            #------------------------------#
            #   初始化用于训练的内容
            #------------------------------#
            if i == 0:
                batch_image_meta    = np.zeros((self.batch_size,) + image_meta.shape, dtype=image_meta.dtype)
                batch_rpn_match     = np.zeros([self.batch_size, self.anchors.shape[0], 1], dtype=rpn_match.dtype)
                batch_rpn_bbox      = np.zeros([self.batch_size, self.config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4], dtype=rpn_bbox.dtype)
                batch_images        = np.zeros((self.batch_size,) + image.shape, dtype=np.float32)
                batch_gt_class_ids  = np.zeros((self.batch_size, self.config.MAX_GT_INSTANCES), dtype=np.int32)
                batch_gt_boxes      = np.zeros((self.batch_size, self.config.MAX_GT_INSTANCES, 4), dtype=np.int32)
                batch_gt_masks      = np.zeros((self.batch_size, gt_masks.shape[0], gt_masks.shape[1], self.config.MAX_GT_INSTANCES), dtype=gt_masks.dtype)

            #------------------------------#
            #   将当前信息加载进batch
            #------------------------------#
            batch_image_meta[i]                             = image_meta
            batch_rpn_match[i]                              = rpn_match[:, np.newaxis]
            batch_rpn_bbox[i]                               = rpn_bbox
            batch_images[i]                                 = preprocess_input(image.astype(np.float32))
            batch_gt_class_ids[i, :gt_class_ids.shape[0]]   = gt_class_ids
            batch_gt_boxes[i, :gt_boxes.shape[0]]           = gt_boxes
            batch_gt_masks[i, :, :, :gt_masks.shape[-1]]    = gt_masks
        return [batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox, batch_gt_class_ids, batch_gt_boxes, batch_gt_masks], \
               [np.zeros(self.batch_size), np.zeros(self.batch_size), np.zeros(self.batch_size), np.zeros(self.batch_size), np.zeros(self.batch_size)]

    def __len__(self):
        return math.ceil(len(self.ids) / float(self.batch_size))

    def pull_item(self, index):
        #------------------------------#
        #   载入coco序号
        #   根据coco序号载入目标信息
        #------------------------------#
        image_id    = self.ids[index]
        target      = self.coco.loadAnns(self.coco.getAnnIds(imgIds = image_id))

        #------------------------------#
        #   根据目标信息判断是否为
        #   iscrowd
        #------------------------------#
        target      = [x for x in target if not ('iscrowd' in x and x['iscrowd'])]
        crowd       = [x for x in target if ('iscrowd' in x and x['iscrowd'])]
        num_crowds  = len(crowd)
        #------------------------------#
        #   将不是iscrowd的目标
        #       是iscrowd的目标进行堆叠
        #------------------------------#
        target      += crowd

        image_path  = osp.join(self.image_path, self.coco.loadImgs(image_id)[0]['file_name'])
        image       = Image.open(image_path)
        image       = cvtColor(image)
        image       = np.array(image, np.float32)
        height, width, _ = image.shape

        if len(target) > 0:
            masks = np.array([self.coco.annToMask(obj).reshape(-1) for obj in target], np.float32)
            masks = masks.reshape((-1, height, width)) 

            boxes_classes = []
            for obj in target:
                bbox        = obj['bbox']
                final_box   = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3], self.label_map[obj['category_id']]]
                boxes_classes.append(final_box)
            boxes_classes = np.array(boxes_classes, np.float32)
            boxes_classes[:, [0, 2]] /= width
            boxes_classes[:, [1, 3]] /= height

        if self.augmentation is not None:
            if len(boxes_classes) > 0:
                image, masks, boxes, labels = self.augmentation(image, masks, boxes_classes[:, :4], {'num_crowds': num_crowds, 'labels': boxes_classes[:, 4]})
                num_crowds  = labels['num_crowds']
                labels      = labels['labels']
                if num_crowds > 0:
                    labels[-num_crowds:] = -1
                boxes       = np.concatenate([boxes, np.expand_dims(labels, axis=1)], -1)
        
        masks               = np.transpose(masks, [1, 2, 0])
        outboxes            = np.zeros_like(boxes)
        outboxes[:, [0, 2]] = boxes[:, [1, 3]] * self.config.IMAGE_SHAPE[0]
        outboxes[:, [1, 3]] = boxes[:, [0, 2]] * self.config.IMAGE_SHAPE[1]
        outboxes[:, -1]     = boxes[:, -1]
        outboxes            = np.array(outboxes, np.int)
        return image, outboxes, masks, num_crowds, image_id
