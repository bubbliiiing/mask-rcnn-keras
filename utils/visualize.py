import os
import sys
import random
import itertools
import colorsys
import numpy as np

from skimage.measure import find_contours
from PIL import Image
import cv2
ROOT_DIR = os.path.abspath("../")

sys.path.append(ROOT_DIR)

#---------------------------------------------------------#
#  Visualization
#---------------------------------------------------------#
def random_colors(N, bright=True):
    """
    生成随机颜色
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """
    打上mask图标
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def display_instances(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), 
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):
    # instance的数量
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]
    colors = colors or random_colors(N)

    # 当masked_image为原图时是在原图上绘制
    # 如果不想在原图上绘制，可以把masked_image设置成等大小的全0矩阵
    masked_image = np.array(image,np.uint8)
    for i in range(N):
        color = colors[i]

        # 该部分用于显示bbox
        if not np.any(boxes[i]):
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            cv2.rectangle(masked_image, (x1, y1), (x2, y2), (color[0] * 255,color[1] * 255,color[2] * 255), 2)

        # 该部分用于显示文字与置信度
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(masked_image, caption, (x1, y1 + 8), font, 1, (255, 255, 255), 2)

        # 该部分用于显示语义分割part
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # 画出语义分割的范围
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            verts = np.fliplr(verts) - 1
            cv2.polylines(masked_image, [np.array([verts],np.int)], 1, (color[0] * 255,color[1] * 255,color[2] * 255), 2)

    img = Image.fromarray(np.uint8(masked_image))
    return img