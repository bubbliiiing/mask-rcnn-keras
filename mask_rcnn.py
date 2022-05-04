import colorsys
import os
import time

import cv2
import numpy as np
from PIL import Image

from nets.mrcnn import get_predict_model
from utils.anchors import get_anchors
from utils.config import Config
from utils.utils import cvtColor, get_classes, resize_image
from utils.utils_bbox import postprocess


class MASK_RCNN(object):
    _defaults = {
        #--------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        #--------------------------------------------------------------------------#
        "model_path"        : 'model_data/mask_rcnn_coco.h5',
        "classes_path"      : 'model_data/coco_classes.txt',
        #---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        #---------------------------------------------------------------------#
        "confidence"        : 0.7,
        #---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        #---------------------------------------------------------------------#
        "nms_iou"           : 0.3,
        #----------------------------------------------------------------------#
        #   输入的shape大小
        #   算法会填充输入图片到[IMAGE_MAX_DIM, IMAGE_MAX_DIM]的大小
        #----------------------------------------------------------------------#
        "IMAGE_MAX_DIM"     : 512,
        #----------------------------------------------------------------------#
        #   用于设定先验框大小，默认的先验框大多数情况下是通用的，可以不修改。
        #   需要和训练设置的RPN_ANCHOR_SCALES一致。
        #----------------------------------------------------------------------#
        "RPN_ANCHOR_SCALES" : [32, 64, 128, 256, 512]
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化Mask-Rcnn
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        #---------------------------------------------------#
        #   计算总的类的数量
        #---------------------------------------------------#
        self.class_names, self.num_classes  = get_classes(self.classes_path)
        self.num_classes                    += 1

        #---------------------------------------------------#
        #   画框设置不同的颜色
        #---------------------------------------------------#
        if self.num_classes <= 81:
            self.colors = np.array([[0, 0, 0], [244, 67, 54], [233, 30, 99], [156, 39, 176], [103, 58, 183], 
                                    [100, 30, 60], [63, 81, 181], [33, 150, 243], [3, 169, 244], [0, 188, 212], 
                                    [20, 55, 200], [0, 150, 136], [76, 175, 80], [139, 195, 74], [205, 220, 57], 
                                    [70, 25, 100], [255, 235, 59], [255, 193, 7], [255, 152, 0], [255, 87, 34], 
                                    [90, 155, 50], [121, 85, 72], [158, 158, 158], [96, 125, 139], [15, 67, 34], 
                                    [98, 55, 20], [21, 82, 172], [58, 128, 255], [196, 125, 39], [75, 27, 134], 
                                    [90, 125, 120], [121, 82, 7], [158, 58, 8], [96, 25, 9], [115, 7, 234], 
                                    [8, 155, 220], [221, 25, 72], [188, 58, 158], [56, 175, 19], [215, 67, 64], 
                                    [198, 75, 20], [62, 185, 22], [108, 70, 58], [160, 225, 39], [95, 60, 144], 
                                    [78, 155, 120], [101, 25, 142], [48, 198, 28], [96, 225, 200], [150, 167, 134], 
                                    [18, 185, 90], [21, 145, 172], [98, 68, 78], [196, 105, 19], [215, 67, 84], 
                                    [130, 115, 170], [255, 0, 255], [255, 255, 0], [196, 185, 10], [95, 167, 234], 
                                    [18, 25, 190], [0, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255], 
                                    [155, 0, 0], [0, 155, 0], [0, 0, 155], [46, 22, 130], [255, 0, 155], 
                                    [155, 0, 255], [255, 155, 0], [155, 255, 0], [0, 155, 255], [0, 255, 155], 
                                    [18, 5, 40], [120, 120, 255], [255, 58, 30], [60, 45, 60], [75, 27, 244], [128, 25, 70]], dtype='uint8')
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        class InferenceConfig(Config):
            GPU_COUNT                   = 1
            IMAGES_PER_GPU              = 1
            NUM_CLASSES                 = self.num_classes
            DETECTION_MIN_CONFIDENCE    = self.confidence
            DETECTION_NMS_THRESHOLD     = self.nms_iou
            
            RPN_ANCHOR_SCALES           = self.RPN_ANCHOR_SCALES
            IMAGE_MAX_DIM               = self.IMAGE_MAX_DIM

        self.config = InferenceConfig()
        self.config.display()
        self.generate()

    #---------------------------------------------------#
    #   生成模型
    #---------------------------------------------------#
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        
        #-----------------------#
        #   载入模型
        #-----------------------#
        self.model = get_predict_model(self.config)
        self.model.load_weights(self.model_path, by_name=True)
    
    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image):
        image_shape     = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image           = cvtColor(image)
        image_origin    = np.array(image, np.uint8)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data, image_metas, windows = resize_image([np.array(image)], self.config)
        #---------------------------------------------------------#
        #   根据当前输入图像的大小，生成先验框
        #---------------------------------------------------------#
        anchors = np.expand_dims(get_anchors(self.config, image_data[0].shape), 0)
        #---------------------------------------------------------#
        #   将图像输入网络当中进行预测！
        #---------------------------------------------------------#
        detections, _, _, mrcnn_mask, _, _, _ = self.model.predict([image_data, image_metas, anchors], verbose=0)

        #---------------------------------------------------#
        #   上面获得的预测结果是相对于padding后的图片的
        #   我们需要将预测结果转换到原图上
        #---------------------------------------------------#
        box_thre, class_thre, class_ids, masks_arg, masks_sigmoid = postprocess(
            detections[0], mrcnn_mask[0], image_shape, image_data[0].shape, windows[0]
        )

        if box_thre is None:
            return image

        #----------------------------------------------------------------------#
        #   masks_class [image_shape[0], image_shape[1]]
        #   根据每个像素点所属的实例和是否满足门限需求，判断每个像素点的种类
        #----------------------------------------------------------------------#
        masks_class     = masks_sigmoid * (class_ids[None, None, :] + 1) 
        masks_class     = np.reshape(masks_class, [-1, np.shape(masks_sigmoid)[-1]])
        masks_class     = np.reshape(masks_class[np.arange(np.shape(masks_class)[0]), np.reshape(masks_arg, [-1])], [image_shape[0], image_shape[1]])
        
        #---------------------------------------------------------#
        #   设置字体与边框厚度
        #---------------------------------------------------------#
        scale       = 0.6
        thickness   = int(max((image.size[0] + image.size[1]) // self.IMAGE_MAX_DIM, 1))
        font        = cv2.FONT_HERSHEY_DUPLEX
        color_masks = self.colors[masks_class].astype('uint8')
        image_fused = cv2.addWeighted(color_masks, 0.4, image_origin, 0.6, gamma=0)
        for i in range(np.shape(class_ids)[0]):
            top, left, bottom, right = np.array(box_thre[i, :], np.int32)

            #---------------------------------------------------------#
            #   获取颜色并绘制预测框
            #---------------------------------------------------------#
            color = self.colors[class_ids[i] + 1].tolist()
            cv2.rectangle(image_fused, (left, top), (right, bottom), color, thickness)

            #---------------------------------------------------------#
            #   获得这个框的种类并写在图片上
            #---------------------------------------------------------#
            class_name  = self.class_names[class_ids[i]]
            print(class_name, top, left, bottom, right)
            text_str    = f'{class_name}: {class_thre[i]:.2f}'
            text_w, text_h = cv2.getTextSize(text_str, font, scale, 1)[0]
            cv2.rectangle(image_fused, (left, top), (left + text_w, top + text_h + 5), color, -1)
            cv2.putText(image_fused, text_str, (left, top + 15), font, scale, (255, 255, 255), 1, cv2.LINE_AA)

        image = Image.fromarray(np.uint8(image_fused))
        return image
        
    def get_FPS(self, image, test_interval):
        image_shape     = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image           = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data, image_metas, windows = resize_image([np.array(image)], self.config)
        #---------------------------------------------------------#
        #   根据当前输入图像的大小，生成先验框
        #---------------------------------------------------------#
        anchors = np.expand_dims(get_anchors(self.config, image_data[0].shape), 0)
        #---------------------------------------------------------#
        #   将图像输入网络当中进行预测！
        #---------------------------------------------------------#
        detections, _, _, mrcnn_mask, _, _, _ = self.model.predict([image_data, image_metas, anchors], verbose=0)

        #---------------------------------------------------#
        #   上面获得的预测结果是相对于padding后的图片的
        #   我们需要将预测结果转换到原图上
        #---------------------------------------------------#
        box_thre, class_thre, class_ids, masks_arg, masks_sigmoid = postprocess(
            detections[0], mrcnn_mask[0], image_shape, image_data[0].shape, windows[0]
        )

        t1 = time.time()
        for _ in range(test_interval):
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            detections, _, _, mrcnn_mask, _, _, _ = self.model.predict([image_data, image_metas, anchors], verbose=0)

            #---------------------------------------------------#
            #   上面获得的预测结果是相对于padding后的图片的
            #   我们需要将预测结果转换到原图上
            #---------------------------------------------------#
            box_thre, class_thre, class_ids, masks_arg, masks_sigmoid = postprocess(
                detections[0], mrcnn_mask[0], image_shape, image_data[0].shape, windows[0]
            )
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def get_map_out(self, image):
        image_shape     = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image           = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data, image_metas, windows = resize_image([np.array(image)], self.config)
        #---------------------------------------------------------#
        #   根据当前输入图像的大小，生成先验框
        #---------------------------------------------------------#
        anchors = np.expand_dims(get_anchors(self.config, image_data[0].shape), 0)
        #---------------------------------------------------------#
        #   将图像输入网络当中进行预测！
        #---------------------------------------------------------#
        detections, _, _, mrcnn_mask, _, _, _ = self.model.predict([image_data, image_metas, anchors], verbose=0)

        #---------------------------------------------------#
        #   上面获得的预测结果是相对于padding后的图片的
        #   我们需要将预测结果转换到原图上
        #---------------------------------------------------#
        box_thre, class_thre, class_ids, masks_arg, masks_sigmoid = postprocess(
            detections[0], mrcnn_mask[0], image_shape, image_data[0].shape, windows[0]
        )

        outboxes = None
        if box_thre is not None:
            outboxes            = np.zeros_like(box_thre)
            outboxes[:, [0, 2]] = box_thre[:, [1, 3]]
            outboxes[:, [1, 3]] = box_thre[:, [0, 2]]
        return outboxes, class_thre, class_ids, masks_arg, masks_sigmoid