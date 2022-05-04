import datetime
import os

import tensorflow as tf
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard)
from keras.layers import Conv2D, Dense, DepthwiseConv2D, PReLU
from keras.optimizers import SGD, Adam
from keras.regularizers import l2
from pycocotools.coco import COCO

from nets.mrcnn import ParallelModel, get_train_model
from nets.mrcnn_training import get_lr_scheduler
from utils.anchors import compute_backbone_shapes, generate_pyramid_anchors
from utils.augmentations import Augmentation
from utils.callbacks import (ExponentDecayScheduler, LossHistory,
                             ParallelModelCheckpoint)
from utils.config import Config
from utils.dataloader import COCODetection
from utils.utils import get_classes, get_coco_label_map

tf.logging.set_verbosity(tf.logging.ERROR)

if __name__ == "__main__":
    #---------------------------------------------------------------------#
    #   train_gpu       训练用到的GPU
    #                   默认为第一张卡、双卡为[0, 1]、三卡为[0, 1, 2]
    #                   在使用多GPU时，每个卡上的batch为总batch除以卡的数量。
    #---------------------------------------------------------------------#
    train_gpu       = [0,]
    #---------------------------------------------------------------------#
    #   classes_path    指向model_data下的txt，与自己训练的数据集相关 
    #                   训练前一定要修改classes_path，使其对应自己的数据集
    #---------------------------------------------------------------------#
    classes_path    = 'model_data/shape_classes.txt'   
    #----------------------------------------------------------------------------------------------------------------------------#
    #   权值文件的下载请看README，可以通过网盘下载。模型的 预训练权重 对不同数据集是通用的，因为特征是通用的。
    #   模型的 预训练权重 比较重要的部分是 主干特征提取网络的权值部分，用于进行特征提取。
    #   预训练权重对于99%的情况都必须要用，不用的话主干部分的权值太过随机，特征提取效果不明显，网络训练的结果也不会好
    #
    #   如果训练过程中存在中断训练的操作，可以将model_path设置成logs文件夹下的权值文件，将已经训练了一部分的权值再次载入。
    #   同时修改下方的 冻结阶段 或者 解冻阶段 的参数，来保证模型epoch的连续性。
    #   
    #   当model_path = ''的时候不加载整个模型的权值。
    #
    #   此处使用的是整个模型的权重，因此是在train.py进行加载的。
    #   如果想要让模型从主干的预训练权值开始训练，则设置model_path为主干网络的权值，此时仅加载主干。
    #   如果想要让模型从0开始训练，则设置model_path = ''，Freeze_Train = Fasle，此时从0开始训练，且没有冻结主干的过程。
    #   一般来讲，从0开始训练效果会很差，因为权值太过随机，特征提取效果不明显。
    #
    #   网络一般不从0开始训练，至少会使用主干部分的权值，有些论文提到可以不用预训练，主要原因是他们 数据集较大 且 调参能力优秀。
    #   如果一定要训练网络的主干部分，可以了解imagenet数据集，首先训练分类模型，分类模型的 主干部分 和该模型通用，基于此进行训练。
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path      = "model_data/mask_rcnn_coco.h5"
    #---------------------------------------------------------------------#
    #   输入的shape大小
    #   算法会填充输入图片到[IMAGE_MAX_DIM, IMAGE_MAX_DIM]的大小
    #---------------------------------------------------------------------#
    IMAGE_MAX_DIM       = 512
    #---------------------------------------------------------------------#
    #   用于设定先验框大小，默认的先验框大多数情况下是通用的，可以不修改。
    #   在目标较小时可以设置较小的先验框如[16, 32, 64, 128, 256]
    #---------------------------------------------------------------------#
    RPN_ANCHOR_SCALES   = [32, 64, 128, 256, 512]

    #----------------------------------------------------------------------------------------------------------------------------#
    #   在此提供若干参数设置建议，各位训练者根据自己的需求进行灵活调整：
    #   （一）从预训练权重开始训练：
    #       Adam：
    #           Init_Epoch = 0，Epoch = 100，optimizer_type = 'adam'，Init_lr = 1e-4，weight_decay = 0。
    #       SGD：
    #           Init_Epoch = 0，Epoch = 100，optimizer_type = 'sgd'，Init_lr = 1e-2，weight_decay = 5e-4。
    #       其中：UnFreeze_Epoch可以在100-300之间调整。
    #   （二）batch_size的设置：
    #       在显卡能够接受的范围内，以大为好。显存不足与数据集大小无关，提示显存不足（OOM或者CUDA out of memory）请调小batch_size。
    #----------------------------------------------------------------------------------------------------------------------------#
    #------------------------------------------------------#
    #   训练参数
    #   Init_Epoch      模型当前开始的训练世代
    #   Epoch           模型总共训练的epoch
    #   batch_size      每次输入的图片数量
    #------------------------------------------------------#
    Init_Epoch      = 0
    Epoch           = 100
    batch_size      = 2

    #------------------------------------------------------------------#
    #   其它训练参数：学习率、优化器、学习率下降有关
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   Init_lr         模型的最大学习率
    #   Min_lr          模型的最小学习率，默认为最大学习率的0.01
    #------------------------------------------------------------------#
    Init_lr             = 1e-4
    Min_lr              = Init_lr * 0.01
    #------------------------------------------------------------------#
    #   optimizer_type  使用到的优化器种类，可选的有adam、sgd
    #                   当使用Adam优化器时建议设置  Init_lr=1e-4
    #                   当使用SGD优化器时建议设置   Init_lr=1e-2
    #   momentum        优化器内部使用到的momentum参数
    #   weight_decay    权值衰减，可防止过拟合
    #                   adam会导致weight_decay错误，使用adam时建议设置为0。
    #------------------------------------------------------------------#
    optimizer_type      = "adam"
    momentum            = 0.9
    weight_decay        = 0
    #------------------------------------------------------------------#
    #   lr_decay_type   使用到的学习率下降方式，可选的有step、cos
    #------------------------------------------------------------------#
    lr_decay_type       = "cos"
    #------------------------------------------------------------------#
    #   save_period     多少个epoch保存一次权值
    #------------------------------------------------------------------#
    save_period         = 5
    #------------------------------------------------------------------#
    #   save_dir        权值与日志文件保存的文件夹
    #------------------------------------------------------------------#
    save_dir            = 'logs'
    #------------------------------------------------------------------#
    #   用于设置是否使用多线程读取数据
    #   开启后会加快数据读取速度，但是会占用更多内存
    #   内存较小的电脑可以设置为2或者1 
    #------------------------------------------------------------------#
    num_workers         = 1
    
    #----------------------------------------------------#
    #   获得图片路径和标签
    #   默认指向根目录下面的datasets/coco文件夹
    #----------------------------------------------------#
    train_image_path        = "datasets/coco/JPEGImages"
    train_annotation_path   = "datasets/coco/Jsons/train_annotations.json"
    val_image_path          = "datasets/coco/JPEGImages"
    val_annotation_path     = "datasets/coco/Jsons/val_annotations.json"

    #------------------------------------------------------#
    #   设置用到的显卡
    #------------------------------------------------------#
    os.environ["CUDA_VISIBLE_DEVICES"]  = ','.join(str(x) for x in train_gpu)
    ngpus_per_node                      = len(train_gpu)
    print('Number of devices: {}'.format(ngpus_per_node))

    #----------------------------------------------------#
    #   获取classes和anchor
    #----------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)
    num_classes = num_classes + 1
    
    class TrainConfig(Config):
        GPU_COUNT                   = ngpus_per_node
        IMAGES_PER_GPU              = batch_size // ngpus_per_node
        NUM_CLASSES                 = num_classes
        
        RPN_ANCHOR_SCALES           = RPN_ANCHOR_SCALES
        IMAGE_MAX_DIM               = IMAGE_MAX_DIM

    config = TrainConfig()
    config.display()

    backbone_shapes = compute_backbone_shapes(config, config.IMAGE_SHAPE)
    anchors         = generate_pyramid_anchors(config.RPN_ANCHOR_SCALES, config.RPN_ANCHOR_RATIOS, backbone_shapes, config.BACKBONE_STRIDES, config.RPN_ANCHOR_STRIDE)
    
    model_body  = get_train_model(config)
    if model_path != "":
        #------------------------------------------------------#
        #   载入预训练权重
        #------------------------------------------------------#
        print('Load weights {}.'.format(model_path))
        model_body.load_weights(model_path, by_name=True, skip_mismatch=True)

    if ngpus_per_node > 1:
        model   = ParallelModel(model_body, ngpus_per_node)
    else:
        model   = model_body
        
    #---------------------------#
    #   读取数据集对应的txt
    #---------------------------#
    train_coco  = COCO(train_annotation_path)
    val_coco    = COCO(val_annotation_path)
    num_train   = len(list(train_coco.imgToAnns.keys()))
    num_val     = len(list(val_coco.imgToAnns.keys()))

    #---------------------------------------------------------#
    #   总训练世代指的是遍历全部数据的总次数
    #   总训练步长指的是梯度下降的总次数 
    #   每个训练世代包含若干训练步长，每个训练步长进行一次梯度下降。
    #   此处仅建议最低训练世代，上不封顶，计算时只考虑了解冻部分
    #----------------------------------------------------------#
    wanted_step = 1.5e4 if optimizer_type == "sgd" else 0.5e4
    total_step  = num_train // batch_size * Epoch
    if total_step <= wanted_step:
        wanted_epoch = wanted_step // (num_train // batch_size) + 1
        print("\n\033[1;33;44m[Warning] 使用%s优化器时，建议将训练总步长设置到%d以上。\033[0m"%(optimizer_type, wanted_step))
        print("\033[1;33;44m[Warning] 本次运行的总训练数据量为%d，Unfreeze_batch_size为%d，共训练%d个Epoch，计算出总训练步长为%d。\033[0m"%(num_train, batch_size, Epoch, total_step))
        print("\033[1;33;44m[Warning] 由于总训练步长为%d，小于建议总步长%d，建议设置总世代为%d。\033[0m"%(total_step, wanted_step, wanted_epoch))

    for layer in model.layers:
        if isinstance(layer, DepthwiseConv2D):
            layer.add_loss(l2(weight_decay)(layer.depthwise_kernel))
        elif isinstance(layer, Conv2D) or isinstance(layer, Dense):
            layer.add_loss(l2(weight_decay)(layer.kernel))
        elif isinstance(layer, PReLU):
            layer.add_loss(l2(weight_decay)(layer.alpha))

    COCO_LABEL_MAP  = get_coco_label_map(train_coco, class_names)

    if True:
        #-------------------------------------------------------------------#
        #   判断当前batch_size，自适应调整学习率
        #-------------------------------------------------------------------#
        nbs             = 16
        lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 1e-1
        lr_limit_min    = 3e-5 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        optimizer = {
            'adam'  : Adam(lr = Init_lr_fit, beta_1 = momentum),
            'sgd'   : SGD(lr = Init_lr_fit, momentum = momentum, nesterov=True)
        }[optimizer_type]

        #---------------------------------------#
        #   进行编译
        #---------------------------------------#
        model.compile(optimizer=optimizer, loss={
            'rpn_class_loss'    : lambda y_true, y_pred: y_pred,
            'rpn_bbox_loss'     : lambda y_true, y_pred: y_pred,
            'mrcnn_class_loss'  : lambda y_true, y_pred: y_pred,
            'mrcnn_bbox_loss'   : lambda y_true, y_pred: y_pred,
            'mrcnn_mask_loss'   : lambda y_true, y_pred: y_pred
        })

        #---------------------------------------#
        #   获得学习率下降的公式
        #---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, Epoch)
        
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size

        train_dataloader    = COCODetection(train_image_path, train_coco, num_classes, anchors, batch_size, config, COCO_LABEL_MAP, Augmentation(config.IMAGE_SHAPE))
        val_dataloader      = COCODetection(val_image_path, train_coco, num_classes, anchors, batch_size, config, COCO_LABEL_MAP, Augmentation(config.IMAGE_SHAPE))

        #-------------------------------------------------------------------------------#
        #   训练参数的设置
        #   logging         用于设置tensorboard的保存地址
        #   checkpoint      用于设置权值保存的细节，period用于修改多少epoch保存一次
        #   lr_scheduler       用于设置学习率下降的方式
        #   early_stopping  用于设定早停，val_loss多次不下降自动结束训练，表示模型基本收敛
        #-------------------------------------------------------------------------------#
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        logging         = TensorBoard(log_dir)
        loss_history    = LossHistory(log_dir)
        if ngpus_per_node > 1:
            checkpoint      = ParallelModelCheckpoint(model_body, os.path.join(save_dir, "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5"), 
                                    monitor = 'val_loss', save_weights_only = True, save_best_only = False, period = save_period)
            checkpoint_last = ParallelModelCheckpoint(model_body, os.path.join(save_dir, "last_epoch_weights.h5"), 
                                    monitor = 'val_loss', save_weights_only = True, save_best_only = False, period = 1)
            checkpoint_best = ParallelModelCheckpoint(model_body, os.path.join(save_dir, "best_epoch_weights.h5"), 
                                    monitor = 'val_loss', save_weights_only = True, save_best_only = True, period = 1)
        else:
            checkpoint      = ModelCheckpoint(os.path.join(save_dir, "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5"), 
                                    monitor = 'val_loss', save_weights_only = True, save_best_only = False, period = save_period)
            checkpoint_last = ModelCheckpoint(os.path.join(save_dir, "last_epoch_weights.h5"), 
                                    monitor = 'val_loss', save_weights_only = True, save_best_only = False, period = 1)
            checkpoint_best = ModelCheckpoint(os.path.join(save_dir, "best_epoch_weights.h5"), 
                                    monitor = 'val_loss', save_weights_only = True, save_best_only = True, period = 1)
        early_stopping  = EarlyStopping(monitor='val_loss', min_delta = 0, patience = 10, verbose = 1)
        lr_scheduler    = LearningRateScheduler(lr_scheduler_func, verbose = 1)
        callbacks       = [logging, loss_history, checkpoint, checkpoint_last, checkpoint_best, lr_scheduler]

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(
            generator           = train_dataloader,
            steps_per_epoch     = epoch_step,
            validation_data     = val_dataloader,
            validation_steps    = epoch_step_val,
            epochs              = Epoch,
            initial_epoch       = Init_Epoch,
            use_multiprocessing = True if num_workers > 1 else False,
            workers             = num_workers,
            callbacks           = callbacks
        )
