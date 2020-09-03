
import numpy as np

class Config(object):
    """
    基本配置类。对于自定义配置，请创建
    继承自该类并重写属性的子类
    """
    # 名称
    NAME = None 

    # GPU数量
    GPU_COUNT = 1

    # 每个GPU的图片数量
    IMAGES_PER_GPU = 2

    # 每个世代的步长
    STEPS_PER_EPOCH = 1000

    # 验证集长度
    VALIDATION_STEPS = 50

    COMPUTE_BACKBONE_SHAPE = None

    # 特征金字塔的步长
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]

    # 分类图中完全连接层的大小
    FPN_CLASSIF_FC_LAYERS_SIZE = 1024

    # 用于构建特征金字塔的自上而下层的大小
    TOP_DOWN_PYRAMID_SIZE = 256

    # 分类类别数（包括背景）
    NUM_CLASSES = 1 

    # 建议框的先验框的长度（像素）
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    # 先验框的变化比率
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]

    # 建议框步长
    RPN_ANCHOR_STRIDE = 1

    # 建议框的非极大抑制的值
    RPN_NMS_THRESHOLD = 0.7

    # 每个图像有多少先验框用于RPN培训
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256

    # 非极大抑制前的框的数量
    PRE_NMS_LIMIT = 6000

    # 非最大抑制后保持的ROI（训练和推理）
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000

    # 是否使用Mini Mask
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width)

    BACKBONE = "resnet101"
    # 可选择的
    # square: 调整大小并用零填充以获得大小的方形图像 [max_dim, max_dim].
    # pad64:  如果IMAGE_MIN_DIM或IMAGE_MIN_SCALE 不是“无”，则在填充之前它会先放大。
    #         在此中忽略图像最大亮度模式需要64的倍数，以确保在FPN金字塔的6个级别上下平滑地缩放特征地图（2**6=64）。
    # crop:   从图像中随机选取作物。首先，根据图像亮度和图像灰度对图像进行缩放，
    #         然后随机选取一个大小为image_MIN_DIM x image_MIN_DIM的裁剪。只能用于培训。此模式下不使用图像最大亮度。
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024
    # 最小比例。在IMAGE_MIN_DIM后检查，可以强制进一步放大。例如，如果设置为2，
    # 则图像将缩放为宽度和高度的两倍或更多，即使MIN_IMAGE_DIM不需要它。然而，在“正方形”模式下，它可能会被图像_MAX_DIM否决。
    IMAGE_MIN_SCALE = 0
    # RGB = 3, grayscale = 1, RGB-D = 4
    IMAGE_CHANNEL_COUNT = 3

    # Image mean (RGB)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])

    # 训练用的ROIS数量
    TRAIN_ROIS_PER_IMAGE = 200

    # 正样本比例
    ROI_POSITIVE_RATIO = 0.33

    # 池化方式
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14

    # Mask
    MASK_SHAPE = [28, 28]

    MAX_GT_INSTANCES = 100

    # 标准化比率
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

    DETECTION_MAX_INSTANCES = 100

    # 置信度
    DETECTION_MIN_CONFIDENCE = 0.7

    # 非极大抑制
    DETECTION_NMS_THRESHOLD = 0.3

    WEIGHT_DECAY = 0.0001

    # 损失的比重
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }


    USE_RPN_ROIS = True

    # 是否冻结BN层
    TRAIN_BN = False 

    GRADIENT_CLIP_NORM = 5.0

    def __init__(self):
        # 计算BATCH
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

        if self.IMAGE_RESIZE_MODE == "crop":
            self.IMAGE_SHAPE = np.array([self.IMAGE_MIN_DIM, self.IMAGE_MIN_DIM,
                self.IMAGE_CHANNEL_COUNT])
        else:
            self.IMAGE_SHAPE = np.array([self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM,
                self.IMAGE_CHANNEL_COUNT])

        self.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES

    def display(self):
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
