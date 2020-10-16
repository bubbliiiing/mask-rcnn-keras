import tensorflow as tf
import keras.backend as K
import random
import numpy as np
import logging
from utils import utils
from utils.anchors import compute_backbone_shapes,generate_pyramid_anchors

#----------------------------------------------------------#
#   损失函数公式们
#----------------------------------------------------------#
def batch_pack_graph(x, counts, num_rows):
    outputs = []
    for i in range(num_rows):
        outputs.append(x[i, :counts[i]])
    return tf.concat(outputs, axis=0)

def smooth_l1_loss(y_true, y_pred):
    """
    smmoth_l1 损失函数
    """
    diff = K.abs(y_true - y_pred)
    less_than_one = K.cast(K.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
    return loss

def rpn_class_loss_graph(rpn_match, rpn_class_logits):
    """
    建议框分类损失函数
    """
    # 在最后一维度添加一维度
    rpn_match = tf.squeeze(rpn_match, -1)
    # 获得正样本
    anchor_class = K.cast(K.equal(rpn_match, 1), tf.int32)
    # 获得未被忽略的样本
    indices = tf.where(K.not_equal(rpn_match, 0))
    # 获得预测结果和实际结果
    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
    anchor_class = tf.gather_nd(anchor_class, indices)
    # 计算二者之间的交叉熵
    loss = K.sparse_categorical_crossentropy(target=anchor_class,
                                             output=rpn_class_logits,
                                             from_logits=True)
    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    loss = K.switch(tf.math.is_nan(loss), tf.constant([0.0]), loss)
    return loss

def rpn_bbox_loss_graph(config, target_bbox, rpn_match, rpn_bbox):
    """
    建议框回归损失
    """
    # 在最后一维度添加一维度
    rpn_match = K.squeeze(rpn_match, -1)

    # 获得正样本
    indices = tf.where(K.equal(rpn_match, 1))
    # 获得预测结果与实际结果
    rpn_bbox = tf.gather_nd(rpn_bbox, indices)
    # 将目标边界框修剪为与rpn_bbox相同的长度。
    batch_counts = K.sum(K.cast(K.equal(rpn_match, 1), tf.int32), axis=1)
    target_bbox = batch_pack_graph(target_bbox, batch_counts,
                                   config.IMAGES_PER_GPU)
    # 计算smooth_l1损失函数
    loss = smooth_l1_loss(target_bbox, rpn_bbox)
    
    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    loss = K.switch(tf.math.is_nan(loss), tf.constant([0.0]), loss)
    return loss

def mrcnn_class_loss_graph(target_class_ids, pred_class_logits,
                           active_class_ids):
    """
    classifier的分类损失函数
    """
    # 目标信息
    target_class_ids = tf.cast(target_class_ids, 'int64')
    # 预测信息
    pred_class_ids = tf.argmax(pred_class_logits, axis=2)
    pred_active = tf.gather(active_class_ids[0], pred_class_ids)
    # 求二者交叉熵损失
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_class_ids, logits=pred_class_logits)

    # 去除无用的损失
    loss = loss * pred_active

    # 求平均
    loss = tf.reduce_sum(loss) / tf.maximum(tf.reduce_sum(pred_active), 1)
    return loss

def mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox):
    """
    classifier的回归损失函数
    """
    # Reshape
    target_class_ids = K.reshape(target_class_ids, (-1,))
    target_bbox = K.reshape(target_bbox, (-1, 4))
    pred_bbox = K.reshape(pred_bbox, (-1, K.int_shape(pred_bbox)[2], 4))

    # 只有属于正样本的建议框用于训练
    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_roi_class_ids = tf.cast(tf.gather(target_class_ids, positive_roi_ix), tf.int64)
    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

    # 获得对应预测结果与实际结果
    target_bbox = tf.gather(target_bbox, positive_roi_ix)
    pred_bbox = tf.gather_nd(pred_bbox, indices)

    # Smooth-L1 Loss
    loss = K.switch(tf.size(target_bbox) > 0,
                    smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox),
                    tf.constant(0.0))
    loss = K.mean(loss)
    return loss

def mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks):
    """
    交叉熵损失
    """
    target_class_ids = K.reshape(target_class_ids, (-1,))
    # 实际结果
    mask_shape = tf.shape(target_masks)
    target_masks = K.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))

    # 预测结果
    pred_shape = tf.shape(pred_masks)
    pred_masks = K.reshape(pred_masks, (-1, pred_shape[2], pred_shape[3], pred_shape[4]))

    # 进行维度变换 [N, num_classes, height, width]
    pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])

    # 只有正样本有效
    positive_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_class_ids = tf.cast(tf.gather(target_class_ids, positive_ix), tf.int64)
    indices = tf.stack([positive_ix, positive_class_ids], axis=1)

    # 获得实际结果与预测结果
    y_true = tf.gather(target_masks, positive_ix)
    y_pred = tf.gather_nd(pred_masks, indices)

    # shape: [batch, roi, num_classes]
    loss = K.switch(tf.size(y_true) > 0,
                    K.binary_crossentropy(target=y_true, output=y_pred),
                    tf.constant(0.0))
    loss = K.mean(loss)
    return loss

#----------------------------------------------------------#
#   损失函数公式们
#----------------------------------------------------------#
def load_image_gt(dataset, config, image_id, augment=False, augmentation=None,
                  use_mini_mask=False):
    # 载入图片和语义分割效果
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    # print("\nbefore:",image_id,np.shape(mask),np.shape(class_ids))
    # 原始shape
    original_shape = image.shape
    # 获得新图片，原图片在新图片中的位置，变化的尺度，填充的情况等
    image, window, scale, padding, crop = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)
    mask = utils.resize_mask(mask, scale, padding, crop)
    # print("\nafter:",np.shape(mask),np.shape(class_ids))
    # print(np.shape(image),np.shape(mask))
    # 可以把图片进行翻转
    if augment:
        logging.warning("'augment' is deprecated. Use 'augmentation' instead.")
        if random.randint(0, 1):
            image = np.fliplr(image)
            mask = np.fliplr(mask)

    if augmentation:
        import imgaug
        # 可用于图像增强
        MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                           "Fliplr", "Flipud", "CropAndPad",
                           "Affine", "PiecewiseAffine"]

        def hook(images, augmenter, parents, default):
            """Determines which augmenters to apply to masks."""
            return augmenter.__class__.__name__ in MASK_AUGMENTERS

        image_shape = image.shape
        mask_shape = mask.shape
        det = augmentation.to_deterministic()
        image = det.augment_image(image)
        mask = det.augment_image(mask.astype(np.uint8),
                                 hooks=imgaug.HooksImages(activator=hook))
        assert image.shape == image_shape, "Augmentation shouldn't change image size"
        assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"
        mask = mask.astype(np.bool)
    # 检漏，防止某些层内部实际上不存在语义分割情况
    _idx = np.sum(mask, axis=(0, 1)) > 0
    
    # print("\nafterer:",np.shape(mask),np.shape(_idx))
    mask = mask[:, :, _idx]
    class_ids = class_ids[_idx]
    # 找到mask对应的box
    bbox = utils.extract_bboxes(mask)

    active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
    source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
    active_class_ids[source_class_ids] = 1

    if use_mini_mask:
        mask = utils.minimize_mask(bbox, mask, config.MINI_MASK_SHAPE)

    # 生成Image_meta
    image_meta = utils.compose_image_meta(image_id, original_shape, image.shape,
                                    window, scale, active_class_ids)

    return image, image_meta, class_ids, bbox, mask



def build_rpn_targets(image_shape, anchors, gt_class_ids, gt_boxes, config):
    # 1代表正样本
    # -1代表负样本
    # 0代表忽略
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    # 创建该部分内容利用先验框和真实框进行编码
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
        non_crowd_ix = np.where(gt_class_ids > 0)[0]
        crowd_boxes = gt_boxes[crowd_ix]
        gt_class_ids = gt_class_ids[non_crowd_ix]
        gt_boxes = gt_boxes[non_crowd_ix]
        crowd_overlaps = utils.compute_overlaps(anchors, crowd_boxes)
        crowd_iou_max = np.amax(crowd_overlaps, axis=1)
        no_crowd_bool = (crowd_iou_max < 0.001)
    else:
        no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)

    # 计算先验框和真实框的重合程度 [num_anchors, num_gt_boxes]
    overlaps = utils.compute_overlaps(anchors, gt_boxes)

    # 1. 重合程度小于0.3则代表为负样本
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = -1
    # 2. 每个真实框重合度最大的先验框是正样本
    gt_iou_argmax = np.argwhere(overlaps == np.max(overlaps, axis=0))[:,0]
    rpn_match[gt_iou_argmax] = 1
    # 3. 重合度大于0.7则代表为正样本
    rpn_match[anchor_iou_max >= 0.7] = 1

    # 正负样本平衡
    # 找到正样本的索引
    ids = np.where(rpn_match == 1)[0]
    # 如果大于(config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)则删掉一些
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
    if extra > 0:
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    # 找到负样本的索引
    ids = np.where(rpn_match == -1)[0]
    # 使得总数为config.RPN_TRAIN_ANCHORS_PER_IMAGE
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE -
                        np.sum(rpn_match == 1))
    if extra > 0:
        # Rest the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0

    # 找到内部真实存在物体的先验框，进行编码
    ids = np.where(rpn_match == 1)[0]
    ix = 0 
    for i, a in zip(ids, anchors[ids]):
        gt = gt_boxes[anchor_iou_argmax[i]]
        # 计算真实框的中心，高宽
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w
        # 计算先验框中心，高宽
        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w
        # 编码运算
        rpn_bbox[ix] = [
            (gt_center_y - a_center_y) / np.maximum(a_h, 1),
            (gt_center_x - a_center_x) / np.maximum(a_w, 1),
            np.log(np.maximum(gt_h / np.maximum(a_h, 1), 1e-5)),
            np.log(np.maximum(gt_w / np.maximum(a_w, 1), 1e-5)),
        ]
        # 改变数量级
        rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV
        ix += 1
    return rpn_match, rpn_bbox




def data_generator(dataset, config, shuffle=True, augment=False, augmentation=None,
                   batch_size=1, detection_targets=False,
                   no_augmentation_sources=None):
    """
    网络输入清单
    - images: [batch, H, W, C]
    - image_meta: [batch, (meta data)] 图像详细信息。
    - rpn_match: [batch, N] 代表建议框的匹配情况 (1=正样本, -1=负样本, 0=中性)
    - rpn_bbox: [batch, N, (dy, dx, log(dh), log(dw))] 建议框网络应该有的预测结果.
    - gt_class_ids: [batch, MAX_GT_INSTANCES] 种类ID
    - gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
    - gt_masks: [batch, height, width, MAX_GT_INSTANCES].

    网络输出清单:
        在常规训练中通常是空的。
    """
    b = 0  # batch item index
    image_index = -1
    image_ids = np.copy(dataset.image_ids)
    no_augmentation_sources = no_augmentation_sources or []

    # [anchor_count, (y1, x1, y2, x2)]
    # 计算获得先验框
    backbone_shapes = compute_backbone_shapes(config, config.IMAGE_SHAPE)
    anchors = generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                             config.RPN_ANCHOR_RATIOS,
                                             backbone_shapes,
                                             config.BACKBONE_STRIDES,
                                             config.RPN_ANCHOR_STRIDE)

    while True:

        image_index = (image_index + 1) % len(image_ids)
        if shuffle and image_index == 0:
            np.random.shuffle(image_ids)

        # 获得id
        image_id = image_ids[image_index]

        # 获得图片，真实框，语义分割结果等
        if dataset.image_info[image_id]['source'] in no_augmentation_sources:
            image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
            load_image_gt(dataset, config, image_id, augment=augment,
                            augmentation=None,
                            use_mini_mask=config.USE_MINI_MASK)
        else:
            image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
                load_image_gt(dataset, config, image_id, augment=augment,
                            augmentation=augmentation,
                            use_mini_mask=config.USE_MINI_MASK)

        if not np.any(gt_class_ids > 0):
            continue

        # RPN Targets
        rpn_match, rpn_bbox = build_rpn_targets(image.shape, anchors,
                                                gt_class_ids, gt_boxes, config)

        # 如果某张图片里面物体的数量大于最大值的话，则进行筛选，防止过大
        if gt_boxes.shape[0] > config.MAX_GT_INSTANCES:
            ids = np.random.choice(
                np.arange(gt_boxes.shape[0]), config.MAX_GT_INSTANCES, replace=False)
            gt_class_ids = gt_class_ids[ids]
            gt_boxes = gt_boxes[ids]
            gt_masks = gt_masks[:, :, ids]

        # 初始化用于训练的内容
        if b == 0:
            batch_image_meta = np.zeros(
                (batch_size,) + image_meta.shape, dtype=image_meta.dtype)
            batch_rpn_match = np.zeros(
                [batch_size, anchors.shape[0], 1], dtype=rpn_match.dtype)
            batch_rpn_bbox = np.zeros(
                [batch_size, config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4], dtype=rpn_bbox.dtype)
            batch_images = np.zeros(
                (batch_size,) + image.shape, dtype=np.float32)
            batch_gt_class_ids = np.zeros(
                (batch_size, config.MAX_GT_INSTANCES), dtype=np.int32)
            batch_gt_boxes = np.zeros(
                (batch_size, config.MAX_GT_INSTANCES, 4), dtype=np.int32)
            batch_gt_masks = np.zeros(
                (batch_size, gt_masks.shape[0], gt_masks.shape[1],
                    config.MAX_GT_INSTANCES), dtype=gt_masks.dtype)
        
        # 将当前信息加载进batch
        batch_image_meta[b] = image_meta
        batch_rpn_match[b] = rpn_match[:, np.newaxis]
        batch_rpn_bbox[b] = rpn_bbox
        batch_images[b] = utils.mold_image(image.astype(np.float32), config)
        batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids
        batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
        batch_gt_masks[b, :, :, :gt_masks.shape[-1]] = gt_masks

        b += 1
        
        # 判断是否已经将batch_size全部载入
        if b >= batch_size:
            inputs = [batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox,
                        batch_gt_class_ids, batch_gt_boxes, batch_gt_masks]
            outputs = []

            yield inputs, outputs
            # 开始一个新的batch_size
            b = 0
            

