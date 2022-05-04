import logging
import math
import random
from functools import partial

import keras.backend as K
import numpy as np
import tensorflow as tf
from utils import utils
from utils.anchors import compute_backbone_shapes, generate_pyramid_anchors


#----------------------------------------------------------#
#   损失函数公式们
#----------------------------------------------------------#
def batch_pack_graph(x, counts, num_rows):
    outputs = []
    for i in range(num_rows):
        outputs.append(x[i, :counts[i]])
    return tf.concat(outputs, axis=0)

#----------------------------------------------------------#
#   回归损失函数smooth_l1_loss
#----------------------------------------------------------#
def smooth_l1_loss(y_true, y_pred):
    diff = K.abs(y_true - y_pred)
    less_than_one = K.cast(K.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
    return loss

def rpn_class_loss_graph(rpn_match, rpn_class_logits):
    #----------------------------------------------------------#
    #   在最后一维度添加一维度
    #----------------------------------------------------------#
    rpn_match = tf.squeeze(rpn_match, -1)
    
    #----------------------------------------------------------#
    #   获得正样本与尚未忽略的负样本
    #----------------------------------------------------------#
    anchor_class = K.cast(K.equal(rpn_match, 1), tf.int32)
    indices = tf.where(K.not_equal(rpn_match, 0))
    #----------------------------------------------------------#
    #   获得预测结果和实际结果
    #----------------------------------------------------------#
    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
    anchor_class = tf.gather_nd(anchor_class, indices)
    #----------------------------------------------------------#
    #   计算二者之间的交叉熵
    #----------------------------------------------------------#
    loss = K.sparse_categorical_crossentropy(target=anchor_class, output=rpn_class_logits, from_logits=True)
    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    loss = K.switch(tf.math.is_nan(loss), tf.constant([0.0]), loss)
    loss = K.mean(loss)
    return loss

def rpn_bbox_loss_graph(config, target_bbox, rpn_match, rpn_bbox):
    #----------------------------------------------------------#
    #   在最后一维度添加一维度
    #----------------------------------------------------------#
    rpn_match = K.squeeze(rpn_match, -1)

    #----------------------------------------------------------#
    #   获得正样本用于计算回归loss
    #----------------------------------------------------------#
    indices = tf.where(K.equal(rpn_match, 1))
    #----------------------------------------------------------#
    #   获得预测结果与实际结果
    #----------------------------------------------------------#
    rpn_bbox = tf.gather_nd(rpn_bbox, indices)
    batch_counts = K.sum(K.cast(K.equal(rpn_match, 1), tf.int32), axis=1)
    target_bbox = batch_pack_graph(target_bbox, batch_counts, config.IMAGES_PER_GPU)
    #----------------------------------------------------------#
    #   计算smooth_l1损失函数
    #----------------------------------------------------------#
    loss = smooth_l1_loss(target_bbox, rpn_bbox)
    
    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    loss = K.switch(tf.math.is_nan(loss), tf.constant([0.0]), loss)
    loss = K.mean(loss)
    return loss

def mrcnn_class_loss_graph(target_class_ids, pred_class_logits,
                           active_class_ids):
    #----------------------------------------------------------#
    #   classifier的分类损失函数
    #----------------------------------------------------------#
    target_class_ids = tf.cast(target_class_ids, 'int64')
    pred_class_ids = tf.argmax(pred_class_logits, axis=2)
    pred_active = tf.gather(active_class_ids[0], pred_class_ids)

    # target_class_ids = tf.Print(target_class_ids, [target_class_ids], summarize=100)
    #----------------------------------------------------------#
    #   求二者交叉熵损失
    #----------------------------------------------------------#
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_class_ids, logits=pred_class_logits)
    loss = loss * pred_active

    loss = tf.reduce_sum(loss) / tf.maximum(tf.reduce_sum(pred_active), 1)
    return loss

def mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox):
    #----------------------------------------------------------#
    #   进行Reshape方便下一步处理
    #----------------------------------------------------------#
    target_class_ids = K.reshape(target_class_ids, (-1,))
    target_bbox = K.reshape(target_bbox, (-1, 4))
    pred_bbox = K.reshape(pred_bbox, (-1, K.int_shape(pred_bbox)[2], 4))

    #----------------------------------------------------------#
    #   只有属于正样本的建议框用于训练
    #----------------------------------------------------------#
    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_roi_class_ids = tf.cast(tf.gather(target_class_ids, positive_roi_ix), tf.int64)
    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

    #----------------------------------------------------------#
    #   获得对应预测结果与实际结果
    #----------------------------------------------------------#
    target_bbox = tf.gather(target_bbox, positive_roi_ix)
    pred_bbox = tf.gather_nd(pred_bbox, indices)

    #----------------------------------------------------------#
    #   计算二者的Smooth-L1 Loss
    #----------------------------------------------------------#
    loss = K.switch(tf.size(target_bbox) > 0,
                    smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox),
                    tf.constant(0.0))
    loss = K.mean(loss)
    return loss

def mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks):
    target_class_ids = K.reshape(target_class_ids, (-1,))
    mask_shape = tf.shape(target_masks)
    target_masks = K.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))

    pred_shape = tf.shape(pred_masks)
    pred_masks = K.reshape(pred_masks, (-1, pred_shape[2], pred_shape[3], pred_shape[4]))

    #----------------------------------------------------------#
    #   进行维度变换，方便后续利用tf.gather_nd获得y_pred
    #   pred_masks  : N, num_classes, height, width
    #----------------------------------------------------------#
    pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])

    #----------------------------------------------------------#
    #   只有正样本有效
    #----------------------------------------------------------#
    positive_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_class_ids = tf.cast(tf.gather(target_class_ids, positive_ix), tf.int64)
    indices = tf.stack([positive_ix, positive_class_ids], axis=1)

    #----------------------------------------------------------#
    #   获得实际结果与预测结果
    #----------------------------------------------------------#
    y_true = tf.gather(target_masks, positive_ix)
    y_pred = tf.gather_nd(pred_masks, indices)

    #----------------------------------------------------------#
    #   计算交叉熵损失的平均值
    #----------------------------------------------------------#
    loss = K.switch(tf.size(y_true) > 0, K.binary_crossentropy(target=y_true, output=y_pred), tf.constant(0.0))
    loss = K.mean(loss)
    return loss

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2
            ) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0
                + math.cos(
                    math.pi
                    * (iters - warmup_total_iters)
                    / (total_iters - warmup_total_iters - no_aug_iter)
                )
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

