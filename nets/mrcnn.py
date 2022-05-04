import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.layers import (Activation, Add, BatchNormalization, Concatenate,
                          Conv2D, Conv2DTranspose, Dense, Input, Lambda,
                          MaxPooling2D, Reshape, TimeDistributed, UpSampling2D,
                          ZeroPadding2D)
from keras.models import Model
from utils.anchors import get_anchors

from nets.layers import (DetectionLayer, DetectionTargetLayer, ProposalLayer,
                         PyramidROIAlign, norm_boxes_graph, parse_image_meta_graph)
from nets.mrcnn_training import *
from nets.resnet import get_resnet


#------------------------------------#
#   五个不同大小的特征层会传入到
#   RPN当中，获得建议框
#------------------------------------#
def rpn_graph(feature_map, anchors_per_location):
    #------------------------------------#
    #   利用一个3x3卷积进行特征整合
    #------------------------------------#
    shared = Conv2D(512, (3, 3), padding='same', activation='relu',
                       name='rpn_conv_shared')(feature_map)
    
    #------------------------------------#
    #   batch_size, num_anchors, 2
    #   代表这个先验框是否包含物体
    #------------------------------------#
    x = Conv2D(anchors_per_location * 2, (1, 1), padding='valid', activation='linear', name='rpn_class_raw')(shared)
    rpn_class_logits = Reshape([-1,2])(x)
    rpn_probs = Activation("softmax", name="rpn_class_xxx")(rpn_class_logits)
    
    #------------------------------------#
    #   batch_size, num_anchors, 4
    #   这个先验框的调整参数
    #------------------------------------#
    x = Conv2D(anchors_per_location * 4, (1, 1), padding="valid", activation='linear', name='rpn_bbox_pred')(shared)
    rpn_bbox = Reshape([-1, 4])(x)

    return [rpn_class_logits, rpn_probs, rpn_bbox]

#------------------------------------#
#   建立建议框网络模型
#   RPN模型
#------------------------------------#
def build_rpn_model(anchors_per_location, depth):
    input_feature_map = Input(shape=[None, None, depth], name="input_rpn_feature_map")
    outputs = rpn_graph(input_feature_map, anchors_per_location)
    return Model([input_feature_map], outputs, name="rpn_model")

#------------------------------------#
#   建立classifier模型
#   这个模型的预测结果会调整建议框
#   获得最终的预测框
#------------------------------------#
def fpn_classifier_graph(rois, feature_maps, image_meta,
                         pool_size, num_classes, train_bn=True,
                         fc_layers_size=1024):
    #---------------------------------------------------------------#
    #   ROI Pooling，利用建议框在特征层上进行截取
    #   x   : [batch, num_rois, POOL_SIZE, POOL_SIZE, channels]
    #---------------------------------------------------------------#
    x = PyramidROIAlign([pool_size, pool_size], name="roi_align_classifier")([rois, image_meta] + feature_maps)

    #------------------------------------------------------------------#
    #   利用卷积进行特征整合
    #   x   : [batch, num_rois, 1, 1, fc_layers_size]
    #------------------------------------------------------------------#
    x = TimeDistributed(Conv2D(fc_layers_size, (pool_size, pool_size), padding="valid"),  name="mrcnn_class_conv1")(x)
    x = TimeDistributed(BatchNormalization(), name='mrcnn_class_bn1')(x, training=train_bn)
    x = Activation('relu')(x)
    #------------------------------------------------------------------#
    #   x   : [batch, num_rois, 1, 1, fc_layers_size]
    #------------------------------------------------------------------#
    x = TimeDistributed(Conv2D(fc_layers_size, (1, 1)), name="mrcnn_class_conv2")(x)
    x = TimeDistributed(BatchNormalization(), name='mrcnn_class_bn2')(x, training=train_bn)
    x = Activation('relu')(x)

    #------------------------------------------------------------------#
    #   x   : [batch, num_rois, fc_layers_size]
    #------------------------------------------------------------------#
    shared = Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2),  name="pool_squeeze")(x)

    #------------------------------------------------------------------#
    #   Classifier head
    #   这个的预测结果代表这个先验框内部的物体的种类
    #   mrcnn_probs   : [batch, num_rois, num_classes]
    #------------------------------------------------------------------#
    mrcnn_class_logits = TimeDistributed(Dense(num_classes), name='mrcnn_class_logits')(shared)
    mrcnn_probs = TimeDistributed(Activation("softmax"), name="mrcnn_class")(mrcnn_class_logits)

    #------------------------------------------------------------------#
    #   BBox head
    #   这个的预测结果会对先验框进行调整
    #   mrcnn_bbox : [batch, num_rois, num_classes, 4]
    #------------------------------------------------------------------#
    x = TimeDistributed(Dense(num_classes * 4, activation='linear'), name='mrcnn_bbox_fc')(shared)
    mrcnn_bbox = Reshape((-1, num_classes, 4), name="mrcnn_bbox")(x)

    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox


#----------------------------------------------#
#   建立mask模型
#   这个模型会利用预测框对特征层进行ROIAlign
#   根据截取下来的特征层进行语义分割
#----------------------------------------------#
def build_fpn_mask_graph(rois, feature_maps, image_meta,
                         pool_size, num_classes, train_bn=True):
    #--------------------------------------------------------------------#
    #   ROI Pooling，利用预测框在特征层上进行截取
    #   x   : batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels
    #--------------------------------------------------------------------#
    x = PyramidROIAlign([pool_size, pool_size], name="roi_align_mask")([rois, image_meta] + feature_maps)

    #--------------------------------------------------------------------#
    #   x   : batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, 256
    #--------------------------------------------------------------------#
    x = TimeDistributed(Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv1")(x)
    x = TimeDistributed(BatchNormalization(), name='mrcnn_mask_bn1')(x, training=train_bn)
    x = Activation('relu')(x)

    #--------------------------------------------------------------------#
    #   x   : batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, 256
    #--------------------------------------------------------------------#
    x = TimeDistributed(Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv2")(x)
    x = TimeDistributed(BatchNormalization(), name='mrcnn_mask_bn2')(x, training=train_bn)
    x = Activation('relu')(x)

    #--------------------------------------------------------------------#
    #   x   : batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, 256
    #--------------------------------------------------------------------#
    x = TimeDistributed(Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv3")(x)
    x = TimeDistributed(BatchNormalization(), name='mrcnn_mask_bn3')(x, training=train_bn)
    x = Activation('relu')(x)

    #--------------------------------------------------------------------#
    #   x   : batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, 256
    #--------------------------------------------------------------------#
    x = TimeDistributed(Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv4")(x)
    x = TimeDistributed(BatchNormalization(), name='mrcnn_mask_bn4')(x, training=train_bn)
    x = Activation('relu')(x)

    #--------------------------------------------------------------------#
    #   x   : batch, num_rois, 2xMASK_POOL_SIZE, 2xMASK_POOL_SIZE, 256
    #--------------------------------------------------------------------#
    x = TimeDistributed(Conv2DTranspose(256, (2, 2), strides=2, activation="relu"), name="mrcnn_mask_deconv")(x)
    #--------------------------------------------------------------------#
    #   反卷积后再次进行一个1x1卷积调整通道，
    #   使其最终数量为numclasses，代表分的类
    #   x   : batch, num_rois, 2xMASK_POOL_SIZE, 2xMASK_POOL_SIZE, numclasses
    #--------------------------------------------------------------------#
    x = TimeDistributed(Conv2D(num_classes, (1, 1), strides=1, activation="sigmoid"), name="mrcnn_mask")(x)
    return x


def get_predict_model(config):
    h, w = config.IMAGE_SHAPE[:2]
    #----------------------------------------------#
    #   输入进来的图片必须是2的6次方以上的倍数
    #----------------------------------------------#
    if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
        raise Exception("Image size must be dividable by 2 at least 6 times "
                        "to avoid fractions when downscaling and upscaling."
                        "For example, use 256, 320, 384, 448, 512, ... etc. ")
    
    input_image         = Input(shape=[None, None, config.IMAGE_SHAPE[2]], name="input_image")
    #----------------------------------------------#
    #   meta包含了一些必要信息
    #----------------------------------------------#
    input_image_meta    = Input(shape=[config.IMAGE_META_SIZE],name="input_image_meta")
    #----------------------------------------------#
    #   输入进来的先验框
    #----------------------------------------------#
    input_anchors       = Input(shape=[None, 4], name="input_anchors")

    #----------------------------------------------#
    #   获得四个有效特征层
    #   当输入进来的图片是1024,1024,3的时候
    #   C2为256,256,256
    #   C3为128,128,512
    #   C4为64,64,1024
    #   C5为32,32,2048
    #----------------------------------------------#
    _, C2, C3, C4, C5 = get_resnet(input_image, train_bn=config.TRAIN_BN)

    #----------------------------------------------#
    #   组合成特征金字塔的结构
    #   P5长宽共压缩了5次
    #   P5为32,32,256
    #----------------------------------------------#
    P5 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')(C5)
    #----------------------------------------------#
    #   将P5上采样和P4进行相加
    #   P4长宽共压缩了4次
    #   P4为64,64,256
    #----------------------------------------------#
    P4 = Add(name="fpn_p4add")([
        UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
        Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4')(C4)])
    #----------------------------------------------#
    #   将P4上采样和P3进行相加
    #   P3长宽共压缩了3次
    #   P3为128,128,256
    #----------------------------------------------#
    P3 = Add(name="fpn_p3add")([
        UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
        Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')(C3)])
    #----------------------------------------------#
    #   将P3上采样和P2进行相加
    #   P2长宽共压缩了2次
    #   P2为256,256,256
    #----------------------------------------------#
    P2 = Add(name="fpn_p2add")([
        UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
        Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2')(C2)])
        
    #-----------------------------------------------------------#
    #   各自进行一次256通道的卷积，此时P2、P3、P4、P5通道数相同
    #   P2为256,256,256
    #   P3为128,128,256
    #   P4为64,64,256
    #   P5为32,32,256
    #-----------------------------------------------------------#
    P2 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p2")(P2)
    P3 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p3")(P3)
    P4 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p4")(P4)
    P5 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p5")(P5)
    #----------------------------------------------#
    #   在建议框网络里面还有一个P6用于获取建议框
    #   P5为16,16,256
    #----------------------------------------------#
    P6 = MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)

    #----------------------------------------------#
    #   P2, P3, P4, P5, P6可以用于获取建议框
    #----------------------------------------------#
    rpn_feature_maps    = [P2, P3, P4, P5, P6]
    #----------------------------------------------#
    #   P2, P3, P4, P5用于获取mask信息
    #----------------------------------------------#
    mrcnn_feature_maps  = [P2, P3, P4, P5]

    #----------------------------------------------#
    #   建立RPN模型
    #----------------------------------------------#
    anchors = input_anchors
    rpn     = build_rpn_model(len(config.RPN_ANCHOR_RATIOS), config.TOP_DOWN_PYRAMID_SIZE)

    #------------------------------------------------------------------#
    #   获得RPN网络的预测结果，进行格式调整，把五个特征层的结果进行堆叠
    #------------------------------------------------------------------#
    rpn_class_logits, rpn_class, rpn_bbox = [],[],[]
    for p in rpn_feature_maps:
        logits,classes,bbox = rpn([p])
        rpn_class_logits.append(logits)
        rpn_class.append(classes)
        rpn_bbox.append(bbox)
    #------------------------------------------------------------------#
    #   此时获得的rpn_class_logits、rpn_class、rpn_bbox的维度是
    #   rpn_class_logits    : Batch_size, num_anchors, 2
    #   rpn_class           : Batch_size, num_anchors, 2
    #   rpn_bbox            : Batch_size, num_anchors, 4
    #------------------------------------------------------------------#
    rpn_class_logits = Concatenate(axis=1,name="rpn_class_logits")(rpn_class_logits)
    rpn_class = Concatenate(axis=1,name="rpn_class")(rpn_class)
    rpn_bbox = Concatenate(axis=1,name="rpn_bbox")(rpn_bbox)

    #------------------------------------------------------------------#
    #   对先验框进行解码，获得先验框解码后的建议框的坐标
    #   rpn_rois            : Batch_size, proposal_count, 4
    #------------------------------------------------------------------#
    proposal_count = config.POST_NMS_ROIS_INFERENCE 
    rpn_rois = ProposalLayer(proposal_count=proposal_count, nms_threshold=config.RPN_NMS_THRESHOLD, 
                    name="ROI", config=config)([rpn_class, rpn_bbox, anchors])

    #------------------------------------------------------------------#
    #   获得classifier的结果
    #   mrcnn_class_logits  : Batch_size, num_rois, num_classes
    #   mrcnn_class         : Batch_size, num_rois, num_classes
    #   mrcnn_bbox          : Batch_size, num_rois, num_classes, 4
    #------------------------------------------------------------------#
    mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
        fpn_classifier_graph(rpn_rois, mrcnn_feature_maps, input_image_meta,
                                config.POOL_SIZE, config.NUM_CLASSES,
                                train_bn=config.TRAIN_BN,
                                fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)
    
    #------------------------------------------------------------#
    #   detections          : Batch_size, num_detections, 6
    #   detection_boxes     : Batch_size, num_detections, 4
    #------------------------------------------------------------#
    detections = DetectionLayer(config, name="mrcnn_detection")([rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta])
    detection_boxes = Lambda(lambda x: x[..., :4])(detections)
    
    #-------------------------------------------------------------------------------------#
    #   获得mask的结果
    #   mrcnn_mask   : batch, num_rois, 2xMASK_POOL_SIZE, 2xMASK_POOL_SIZE, numclasses
    #-------------------------------------------------------------------------------------#
    mrcnn_mask = build_fpn_mask_graph(detection_boxes, mrcnn_feature_maps,
                                    input_image_meta,
                                    config.MASK_POOL_SIZE,
                                    config.NUM_CLASSES,
                                    train_bn=config.TRAIN_BN)

    #------------------------------------------------------------#
    #   获得整个mrcnn的模型
    #------------------------------------------------------------#
    model = Model([input_image, input_image_meta, input_anchors],
                        [detections, mrcnn_class, mrcnn_bbox,
                            mrcnn_mask, rpn_rois, rpn_class, rpn_bbox],
                        name='mask_rcnn')
    return model

def get_train_model(config):
    h, w = config.IMAGE_SHAPE[:2]
    #----------------------------------------------#
    #   输入进来的图片必须是2的6次方以上的倍数
    #----------------------------------------------#
    if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
        raise Exception("Image size must be dividable by 2 at least 6 times "
                        "to avoid fractions when downscaling and upscaling."
                        "For example, use 256, 320, 384, 448, 512, ... etc. ")

    input_image = Input(shape=[None, None, config.IMAGE_SHAPE[2]], name="input_image")
    #----------------------------------------------#
    #   meta包含了一些必要信息
    #----------------------------------------------#
    input_image_meta = Input(shape=[config.IMAGE_META_SIZE],name="input_image_meta")

    #----------------------------------------------#
    #   RPN建议框网络的真实框信息
    #----------------------------------------------#
    input_rpn_match = Input(shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
    input_rpn_bbox = Input(shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)

    #----------------------------------------------#
    #   种类信息与框的位置信息
    #----------------------------------------------#
    input_gt_class_ids = Input(shape=[None], name="input_gt_class_ids", dtype=tf.int32)
    input_gt_boxes = Input(shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)

    #----------------------------------------------#
    #   将输入进来的框的坐标标准化到0-1之间
    #----------------------------------------------#
    gt_boxes = Lambda(lambda x: norm_boxes_graph(x, K.shape(input_image)[1:3]))(input_gt_boxes)

    #---------------------------------------------------------------------#
    #   mask语义分割信息
    #   batch, MINI_MASK_SHAPE[0], MINI_MASK_SHAPE[1], MAX_GT_INSTANCES
    #---------------------------------------------------------------------#
    if config.USE_MINI_MASK:
        input_gt_masks = Input(shape=[config.MINI_MASK_SHAPE[0], config.MINI_MASK_SHAPE[1], None],name="input_gt_masks", dtype=bool)
    else:
        input_gt_masks = Input(shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None],name="input_gt_masks", dtype=bool)

    #----------------------------------------------#
    #   获得四个有效特征层
    #   当输入进来的图片是1024,1024,3的时候
    #   C2为256,256,256
    #   C3为128,128,512
    #   C4为64,64,1024
    #   C5为32,32,2048
    #----------------------------------------------#
    _, C2, C3, C4, C5 = get_resnet(input_image, train_bn=config.TRAIN_BN)

    #----------------------------------------------#
    #   组合成特征金字塔的结构
    #   P5长宽共压缩了5次
    #   P5为32,32,256
    #----------------------------------------------#
    P5 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')(C5)
    #----------------------------------------------#
    #   将P5上采样和P4进行相加
    #   P4长宽共压缩了4次
    #   P4为64,64,256
    #----------------------------------------------#
    P4 = Add(name="fpn_p4add")([
        UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
        Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4')(C4)])
    #----------------------------------------------#
    #   将P4上采样和P3进行相加
    #   P3长宽共压缩了3次
    #   P3为128,128,256
    #----------------------------------------------#
    P3 = Add(name="fpn_p3add")([
        UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
        Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')(C3)])
    #----------------------------------------------#
    #   将P3上采样和P2进行相加
    #   P2长宽共压缩了2次
    #   P2为256,256,256
    #----------------------------------------------#
    P2 = Add(name="fpn_p2add")([
        UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
        Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2')(C2)])
        
    #-----------------------------------------------------------#
    #   各自进行一次256通道的卷积，此时P2、P3、P4、P5通道数相同
    #   P2为256,256,256
    #   P3为128,128,256
    #   P4为64,64,256
    #   P5为32,32,256
    #-----------------------------------------------------------#
    P2 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p2")(P2)
    P3 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p3")(P3)
    P4 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p4")(P4)
    P5 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p5")(P5)
    #----------------------------------------------#
    #   在建议框网络里面还有一个P6用于获取建议框
    #   P5为16,16,256
    #----------------------------------------------#
    P6 = MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)

    #----------------------------------------------#
    #   P2, P3, P4, P5, P6可以用于获取建议框
    #----------------------------------------------#
    rpn_feature_maps = [P2, P3, P4, P5, P6]
    #----------------------------------------------#
    #   P2, P3, P4, P5用于获取mask信息
    #----------------------------------------------#
    mrcnn_feature_maps = [P2, P3, P4, P5]

    #----------------------------------------------#
    #   将anchors转化成tensor的形式
    #----------------------------------------------#
    anchors = get_anchors(config, config.IMAGE_SHAPE)
    anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)
    anchors = Lambda(lambda x: tf.Variable(anchors), name="anchors")(input_image)
    #----------------------------------------------#
    #   建立RPN模型
    #----------------------------------------------#
    rpn = build_rpn_model(len(config.RPN_ANCHOR_RATIOS), config.TOP_DOWN_PYRAMID_SIZE)

    rpn_class_logits, rpn_class, rpn_bbox = [],[],[]

    #------------------------------------------------------------------#
    #   获得RPN网络的预测结果，进行格式调整，把五个特征层的结果进行堆叠
    #------------------------------------------------------------------#
    rpn_class_logits, rpn_class, rpn_bbox = [],[],[]
    for p in rpn_feature_maps:
        logits,classes,bbox = rpn([p])
        rpn_class_logits.append(logits)
        rpn_class.append(classes)
        rpn_bbox.append(bbox)
        
    #------------------------------------------------------------------#
    #   此时获得的rpn_class_logits、rpn_class、rpn_bbox的维度是
    #   rpn_class_logits    : Batch_size, num_anchors, 2
    #   rpn_class           : Batch_size, num_anchors, 2
    #   rpn_bbox            : Batch_size, num_anchors, 4
    #------------------------------------------------------------------#
    for p in rpn_feature_maps:
        logits,classes,bbox = rpn([p])
        rpn_class_logits.append(logits)
        rpn_class.append(classes)
        rpn_bbox.append(bbox)
    rpn_class_logits = Concatenate(axis=1,name="rpn_class_logits")(rpn_class_logits)
    rpn_class = Concatenate(axis=1,name="rpn_class")(rpn_class)
    rpn_bbox = Concatenate(axis=1,name="rpn_bbox")(rpn_bbox)

    #------------------------------------------------------------------#
    #   对先验框进行解码，获得先验框解码后的建议框的坐标
    #   rpn_rois            : Batch_size, proposal_count, 4
    #------------------------------------------------------------------#
    proposal_count = config.POST_NMS_ROIS_TRAINING
    rpn_rois = ProposalLayer(proposal_count=proposal_count, nms_threshold=config.RPN_NMS_THRESHOLD,
                        name="ROI", config=config)([rpn_class, rpn_bbox, anchors])

    active_class_ids = Lambda(lambda x: parse_image_meta_graph(x)["active_class_ids"])(input_image_meta)

    #--------------------------------------# 
    #   利用预测到的建议框进行下一步的操作
    #--------------------------------------#
    target_rois = rpn_rois

    """
    找到建议框的ground_truth
    Inputs:
    proposals       : [batch, N, (y1, x1, y2, x2)]                                          建议框
    gt_class_ids    : [batch, MAX_GT_INSTANCES]                                             每个真实框对应的类
    gt_boxes        : [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]                           真实框的位置
    gt_masks        : [batch, MINI_MASK_SHAPE[0], MINI_MASK_SHAPE[1], MAX_GT_INSTANCES]     真实框的语义分割情况

    Returns: 
    rois            : [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)]                       内部真实存在目标的建议框
    target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]                                         每个建议框对应的类
    target_bbox     : [batch, TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw)]              每个建议框应该有的调整参数
    target_mask     : [batch, TRAIN_ROIS_PER_IMAGE, 2xMASK_POOL_SIZE, 2xMASK_POOL_SIZE]                          每个建议框语义分割情况
    """
    rois, target_class_ids, target_bbox, target_mask =\
        DetectionTargetLayer(config, name="proposal_targets")([
            target_rois, input_gt_class_ids, gt_boxes, input_gt_masks])

    #------------------------------------------------------------------#
    #   获得classifier的结果
    #   mrcnn_class_logits  : batch, num_rois, num_classes
    #   mrcnn_class         : batch, num_rois, num_classes
    #   mrcnn_bbox          : batch, num_rois, num_classes, 4
    #------------------------------------------------------------------#
    mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
        fpn_classifier_graph(rois, mrcnn_feature_maps, input_image_meta,
                                config.POOL_SIZE, config.NUM_CLASSES,
                                train_bn=config.TRAIN_BN,
                                fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)
    #-------------------------------------------------------------------------------------#
    #   获得mask的结果
    #   mrcnn_mask   : batch, num_rois, 2xMASK_POOL_SIZE, 2xMASK_POOL_SIZE, numclasses
    #-------------------------------------------------------------------------------------#
    mrcnn_mask = build_fpn_mask_graph(rois, mrcnn_feature_maps,
                                        input_image_meta,
                                        config.MASK_POOL_SIZE,
                                        config.NUM_CLASSES,
                                        train_bn=config.TRAIN_BN)

    # Losses
    rpn_class_loss = Lambda(lambda x: rpn_class_loss_graph(*x), name="rpn_class_loss")(
        [input_rpn_match, rpn_class_logits])
    rpn_bbox_loss = Lambda(lambda x: rpn_bbox_loss_graph(config, *x), name="rpn_bbox_loss")(
        [input_rpn_bbox, input_rpn_match, rpn_bbox])
    class_loss = Lambda(lambda x: mrcnn_class_loss_graph(*x), name="mrcnn_class_loss")(
        [target_class_ids, mrcnn_class_logits, active_class_ids])
    bbox_loss = Lambda(lambda x: mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss")(
        [target_bbox, target_class_ids, mrcnn_bbox])
    mask_loss = Lambda(lambda x: mrcnn_mask_loss_graph(*x), name="mrcnn_mask_loss")(
        [target_mask, target_class_ids, mrcnn_mask])

    # Model
    inputs = [input_image, input_image_meta,
                input_rpn_match, input_rpn_bbox, input_gt_class_ids, input_gt_boxes, input_gt_masks]
                
    outputs = [rpn_class_logits, rpn_class, rpn_bbox,
                mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask,
                rpn_rois, 
                rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss]
    model = Model(inputs, outputs, name='mask_rcnn')
    return model

class ParallelModel(Model):
    """Subclasses the standard Keras Model and adds multi-GPU support.
    It works by creating a copy of the model on each GPU. Then it slices
    the inputs and sends a slice to each copy of the model, and then
    merges the outputs together and applies the loss on the combined
    outputs.
    """

    def __init__(self, keras_model, gpu_count):
        """Class constructor.
        keras_model: The Keras model to parallelize
        gpu_count: Number of GPUs. Must be > 1
        """
        super(ParallelModel, self).__init__()
        self.inner_model = keras_model
        self.gpu_count = gpu_count
        merged_outputs = self.make_parallel()
        super(ParallelModel, self).__init__(inputs=self.inner_model.inputs,
                                            outputs=merged_outputs)

    def __getattribute__(self, attrname):
        """Redirect loading and saving methods to the inner model. That's where
        the weights are stored."""
        if 'load' in attrname or 'save' in attrname:
            return getattr(self.inner_model, attrname)
        return super(ParallelModel, self).__getattribute__(attrname)

    def summary(self, *args, **kwargs):
        """Override summary() to display summaries of both, the wrapper
        and inner models."""
        super(ParallelModel, self).summary(*args, **kwargs)
        self.inner_model.summary(*args, **kwargs)

    def make_parallel(self):
        """Creates a new wrapper model that consists of multiple replicas of
        the original model placed on different GPUs.
        """
        # Slice inputs. Slice inputs on the CPU to avoid sending a copy
        # of the full inputs to all GPUs. Saves on bandwidth and memory.
        input_slices = {name: tf.split(x, self.gpu_count)
                        for name, x in zip(self.inner_model.input_names,
                                           self.inner_model.inputs)}

        output_names = self.inner_model.output_names
        outputs_all = []
        for i in range(len(self.inner_model.outputs)):
            outputs_all.append([])

        # Run the model call() on each GPU to place the ops there
        for i in range(self.gpu_count):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('tower_%d' % i):
                    # Run a slice of inputs through this replica
                    zipped_inputs = zip(self.inner_model.input_names,
                                        self.inner_model.inputs)
                    inputs = [
                        Lambda(lambda s: input_slices[name][i],
                                  output_shape=lambda s: (None,) + s[1:])(tensor)
                        for name, tensor in zipped_inputs]
                    # Create the model replica and get the outputs
                    outputs = self.inner_model(inputs)
                    if not isinstance(outputs, list):
                        outputs = [outputs]
                    # Save the outputs for merging back together later
                    for l, o in enumerate(outputs):
                        outputs_all[l].append(o)

        # Merge outputs on CPU
        with tf.device('/cpu:0'):
            merged = []
            for outputs, name in zip(outputs_all, output_names):
                # Concatenate or average outputs?
                # Outputs usually have a batch dimension and we concatenate
                # across it. If they don't, then the output is likely a loss
                # or a metric value that gets averaged across the batch.
                # Keras expects losses and metrics to be scalars.
                if K.int_shape(outputs[0]) == ():
                    # Average
                    m = Lambda(lambda o: tf.add_n(o) / len(outputs), name=name)(outputs)
                else:
                    # Concatenate
                    m = Concatenate(axis=0, name=name)(outputs)
                merged.append(m)
        return merged
