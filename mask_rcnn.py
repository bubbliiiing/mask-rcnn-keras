import os
import sys
import random
import math
import numpy as np
import skimage.io
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from nets.mrcnn import get_predict_model
from utils.config import Config
from utils.anchors import get_anchors
from utils.utils import mold_inputs,unmold_detections
from utils import visualize
import keras.backend as K
class MASK_RCNN(object):
    _defaults = {
        "model_path": 'model_data/mask_rcnn_coco.h5',
        "classes_path": 'model_data/coco_classes.txt',
        "confidence": 0.7,

        # 使用coco数据集检测的时候，IMAGE_MIN_DIM=1024，IMAGE_MAX_DIM=1024, RPN_ANCHOR_SCALES=(32, 64, 128, 256, 512)
        "RPN_ANCHOR_SCALES": (32, 64, 128, 256, 512),
        "IMAGE_MIN_DIM": 1024,
        "IMAGE_MAX_DIM": 1024,
        
        # 在使用自己的数据集进行训练的时候，如果显存不足要调小图片大小
        # 同时要调小anchors
        #"IMAGE_MIN_DIM": 512,
        #"IMAGE_MAX_DIM": 512,
        #"RPN_ANCHOR_SCALES": (16, 32, 64, 128, 256)
    }

    @classmethod
    def get_defaults(cls, n):
        """
        Return the default value for the field.

        Args:
            cls: (callable): write your description
            n: (dict): write your description
        """
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化Mask-Rcnn
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        """
        Initialize the configuration.

        Args:
            self: (todo): write your description
        """
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.sess = K.get_session()
        self.config = self._get_config()
        self.generate()
    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def _get_class(self):
        """
        Get class_path and class_path.

        Args:
            self: (todo): write your description
        """
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        class_names.insert(0,"BG")
        return class_names

    def _get_config(self):
        """
        Get a dictionary of this class configurations.

        Args:
            self: (todo): write your description
        """
        class InferenceConfig(Config):
            NUM_CLASSES = len(self.class_names)
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = self.confidence
            
            NAME = "shapes"
            RPN_ANCHOR_SCALES = self.RPN_ANCHOR_SCALES
            IMAGE_MIN_DIM = self.IMAGE_MIN_DIM
            IMAGE_MAX_DIM = self.IMAGE_MAX_DIM

        config = InferenceConfig()
        config.display()
        return config

    #---------------------------------------------------#
    #   生成模型
    #---------------------------------------------------#
    def generate(self):
        """
        Generate the weights.

        Args:
            self: (todo): write your description
        """
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        
        # 计算总的种类
        self.num_classes = len(self.class_names)

        # 载入模型，如果原来的模型里已经包括了模型结构则直接载入。
        # 否则先构建模型再载入
        self.model = get_predict_model(self.config)
        self.model.load_weights(self.model_path,by_name=True)
    
    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image):
        """
        Detect the image on the image

        Args:
            self: (todo): write your description
            image: (array): write your description
        """
        image = [np.array(image)]
        molded_images, image_metas, windows = mold_inputs(self.config,image)

        image_shape = molded_images[0].shape
        anchors = get_anchors(self.config,image_shape)
        anchors = np.broadcast_to(anchors, (1,) + anchors.shape)

        detections, _, _, mrcnn_mask, _, _, _ =\
            self.model.predict([molded_images, image_metas, anchors], verbose=0)

        final_rois, final_class_ids, final_scores, final_masks =\
            unmold_detections(detections[0], mrcnn_mask[0],
                                    image[0].shape, molded_images[0].shape,
                                    windows[0])

        r = {
            "rois": final_rois,
            "class_ids": final_class_ids,
            "scores": final_scores,
            "masks": final_masks,
        }

        # 想要保存处理后的图片请查询plt保存图片的方法。
        drawed_image = visualize.display_instances(image[0], r['rois'], r['masks'], r['class_ids'], 
                                    self.class_names, r['scores'])
        return drawed_image
        
    def close_session(self):
        """
        Closes the session.

        Args:
            self: (todo): write your description
        """
        self.sess.close()