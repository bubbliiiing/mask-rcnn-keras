import numpy as np
import math
from utils.utils import norm_boxes
from utils.config import Config
import matplotlib.pyplot as plt

#----------------------------------------------------------#
#  Anchors
#----------------------------------------------------------#
def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    # 获得所有框的长度和比例的组合
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # 生成网格中心
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # 获得先验框的中心和宽高
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # 更变格式
    box_centers = np.stack(
        [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # 计算出(y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ylim(-300,1200)
    plt.xlim(-300,1200)
    # plt.ylim(0,600)
    # plt.xlim(0,600)
    plt.scatter(shifts_x,shifts_y)
    box_heights = boxes[:,2]-boxes[:,0]
    box_widths = boxes[:,3]-boxes[:,1]
    
    initial = 0
    for i in [initial+0,initial+1,initial+2]:
        rect = plt.Rectangle([boxes[i, 1],boxes[i, 0]],box_widths[i],box_heights[i],color="r",fill=False)
        ax.add_patch(rect)

    plt.show()
    # print(len(boxes))
    return boxes

def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides,
                             anchor_stride):
    """
    生成不同特征层的anchors，并利用concatenate进行堆叠
    """
    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    # P2对应的scale是32
    # P3对应的scale是64
    # P4对应的scale是128
    # P5对应的scale是256
    # P6对应的scale是512
    anchors = []
    for i in range(len(scales)):
        anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i],
                                        feature_strides[i], anchor_stride))
        
    return np.concatenate(anchors, axis=0)

def compute_backbone_shapes(config, image_shape):
    # 用于计算主干特征提取网络的shape
    if callable(config.BACKBONE):
        return config.COMPUTE_BACKBONE_SHAPE(image_shape)
    # 其实就是计算P2、P3、P4、P5、P6这些特征层的宽和高
    assert config.BACKBONE in ["resnet50", "resnet101"]
    return np.array(
        [[int(math.ceil(image_shape[0] / stride)),
            int(math.ceil(image_shape[1] / stride))]
            for stride in config.BACKBONE_STRIDES])

def get_anchors(config, image_shape):
    backbone_shapes = compute_backbone_shapes(config, image_shape)
    # print(backbone_shapes)
    anchor_cache = {}
    if not tuple(image_shape) in anchor_cache:
        a = generate_pyramid_anchors(
            config.RPN_ANCHOR_SCALES,
            config.RPN_ANCHOR_RATIOS,
            backbone_shapes,
            config.BACKBONE_STRIDES,
            config.RPN_ANCHOR_STRIDE)
        anchor_cache[tuple(image_shape)] = norm_boxes(a, image_shape[:2])
    
    return anchor_cache[tuple(image_shape)]

image_shape = [1024,1024]
config = Config()
anchors = get_anchors(config,image_shape)
print(len(anchors))