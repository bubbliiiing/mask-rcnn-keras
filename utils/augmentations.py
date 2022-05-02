import cv2
import numpy as np
from numpy import random

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, masks=None, boxes=None, labels=None):
        for t in self.transforms:
            img, masks, boxes, labels = t(img, masks, boxes, labels)
        return img, masks, boxes, labels

#---------------------------------------------------------#
#   将图像转换成np.float32
#---------------------------------------------------------#
class ConvertFromInts(object):
    def __call__(self, image, masks=None, boxes=None, labels=None):
        return image.astype(np.float32), masks, boxes, labels

#---------------------------------------------------------#
#   将框的坐标进行调整，调整成相对于原图大小的
#---------------------------------------------------------#
class ToAbsoluteCoords(object):
    def __call__(self, image, masks=None, boxes=None, labels=None):
        height, width, _ = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height
        return image, masks, boxes, labels

#---------------------------------------------------------#
#   调整随机亮度
#---------------------------------------------------------#
class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, masks=None, boxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, masks, boxes, labels

#---------------------------------------------------------#
#   调整随机对比度
#---------------------------------------------------------#
class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, masks=None, boxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, masks, boxes, labels

#---------------------------------------------------------#
#   将RGB转成HSV，方便下一步的处理
#---------------------------------------------------------#
class ConvertColor(object):
    def __init__(self, current='RGB', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, masks=None, boxes=None, labels=None):
        if self.current == 'RGB' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif self.current == 'HSV' and self.transform == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        else:
            raise NotImplementedError
        return image, masks, boxes, labels

#---------------------------------------------------------#
#   调整随机饱和度
#---------------------------------------------------------#
class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, masks=None, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, masks, boxes, labels

#---------------------------------------------------------#
#   调整随机色调
#---------------------------------------------------------#
class RandomHue(object):
    def __init__(self, delta=18.0):
        assert 0.0 <= delta <= 360.0
        self.delta = delta

    def __call__(self, image, masks=None, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, masks, boxes, labels

#---------------------------------------------------------#
#   随机数据增强，色域变换
#---------------------------------------------------------#
class PhotometricDistort(object):
    def __init__(self):
        self.rand_brightness = RandomBrightness()
        self.pd = [
            RandomContrast(),
            ConvertColor(current='RGB', transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='RGB'),
            RandomContrast()
        ]
        
    def __call__(self, image, masks, boxes, labels):
        im = image.copy()
        im, masks, boxes, labels = self.rand_brightness(im, masks, boxes, labels)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, masks, boxes, labels = distort(im, masks, boxes, labels)
        return im, masks, boxes, labels

#---------------------------------------------------------#
#   设计一个较大的图，将原图放上去
#   此时原图相对于大图较小，有利于小目标
#---------------------------------------------------------#
class Expand(object):
    def __init__(self):
        pass

    def __call__(self, image, masks, boxes, labels):
        if random.randint(2):
            return image, masks, boxes, labels

        height, width, depth = image.shape
        ratio   = random.uniform(1, 4)
        left    = random.uniform(0, width * ratio - width)
        top     = random.uniform(0, height * ratio - height)

        expand_image            = np.zeros((int(height * ratio), int(width * ratio), depth), dtype=image.dtype)
        expand_image[:, :, :]   = [128, 128, 128]
        expand_image[int(top):int(top + height), int(left):int(left + width)] = image
        image = expand_image

        expand_masks            = np.zeros((masks.shape[0], int(height * ratio), int(width * ratio)), dtype=masks.dtype)
        expand_masks[:, int(top):int(top + height), int(left):int(left + width)] = masks
        masks = expand_masks

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))

        return image, masks, boxes, labels

#---------------------------------------------------------#
#   对输入进来的图片进行随机裁剪
#---------------------------------------------------------#
class RandomSampleCrop(object):
    def __init__(self):
        self.sample_options = (
            None,
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            (None, None))

    def intersect(self, box_a, box_b):
        max_xy = np.minimum(box_a[:, 2:], box_b[2:])
        min_xy = np.maximum(box_a[:, :2], box_b[:2])
        inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
        return inter[:, 0] * inter[:, 1]

    def jaccard_numpy(self, box_a, box_b):
        inter = self.intersect(box_a, box_b)
        area_a = ((box_a[:, 2] - box_a[:, 0]) *
                (box_a[:, 3] - box_a[:, 1]))  # [A,B]
        area_b = ((box_b[2] - box_b[0]) *
                (box_b[3] - box_b[1]))  # [A,B]
        union = area_a + area_b - inter

        return inter / union  # [A,B]

    def __call__(self, image, masks, boxes=None, labels=None):
        height, width, _ = image.shape
        while True:
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, masks, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            #---------------------------------------------------------#
            #   最多尝试五十次
            #---------------------------------------------------------#
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                #---------------------------------------------------------#
                #   如果高宽比例严重失调，则跳过
                #---------------------------------------------------------#
                if h / w < 0.5 or h / w > 2:
                    continue

                #---------------------------------------------------------#
                #   获得crop位置的左上角
                #---------------------------------------------------------#
                left    = random.uniform(width - w)
                top     = random.uniform(height - h)

                #---------------------------------------------------------#
                #   计算当前截取画面和框的重合程度
                #---------------------------------------------------------#
                left    = random.uniform(width - w)
                rect    = np.array([int(left), int(top), int(left + w), int(top + h)])
                overlap = self.jaccard_numpy(boxes, rect)

                #---------------------------------------------------------#
                #   如果重合程度小于一定条件，则跳过。
                #---------------------------------------------------------#
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                #---------------------------------------------------------#
                #   对原图进行截取
                #---------------------------------------------------------#
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2], :]

                #---------------------------------------------------------#
                #   判断框的中心是否在图像中
                #---------------------------------------------------------#
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                #---------------------------------------------------------#
                #   是的话则保留
                #---------------------------------------------------------#
                mask = m1 * m2

                num_crowds = labels['num_crowds']
                crowd_mask = np.zeros(mask.shape, dtype=np.int32)

                if num_crowds > 0:
                    crowd_mask[-num_crowds:] = 1

                #---------------------------------------------------------#
                #   如果没有框的话，则重新选取图片
                #---------------------------------------------------------#
                if not mask.any() or np.sum(1 - crowd_mask[mask]) == 0:
                    continue

                #---------------------------------------------------------#
                #   根据筛选结果，将参数进行更新
                #---------------------------------------------------------#
                current_masks       = masks[mask, :, :].copy()
                current_boxes       = boxes[mask, :].copy()
                labels['labels']    = labels['labels'][mask]
                if num_crowds > 0:
                    labels['num_crowds'] = np.sum(crowd_mask[mask])
                current_labels      = labels

                #---------------------------------------------------------#
                #   根据筛选结果，将框的坐标进行调整
                #---------------------------------------------------------#
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2], rect[:2])
                current_boxes[:, :2] -= rect[:2]
                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:], rect[2:])
                current_boxes[:, 2:] -= rect[:2]

                current_masks = current_masks[:, rect[1]:rect[3], rect[0]:rect[2]]

                return current_image, current_masks, current_boxes, current_labels

#---------------------------------------------------------#
#   是否进行镜像翻转
#---------------------------------------------------------#
class RandomMirror(object):
    def __call__(self, image, masks, boxes, labels):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            masks = masks[:, :, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, masks, boxes, labels

#---------------------------------------------------------#
#   进行图像大小调整
#---------------------------------------------------------#
class Resize(object):
    def __init__(self, input_shape, resize_gt=True):
        self.resize_gt      = resize_gt
        self.input_shape    = input_shape

    def __call__(self, image, masks, boxes, labels=None):
        image_h, image_w, _ = image.shape
        width, height       = self.input_shape[1], self.input_shape[0]
        image               = cv2.resize(image, (width, height))

        if self.resize_gt:
            masks = masks.transpose((1, 2, 0))
            masks = cv2.resize(masks, (width, height))

            if len(masks.shape) == 2:
                masks = np.expand_dims(masks, 0)
            else:
                masks = masks.transpose((2, 0, 1))

            boxes[:, [0, 2]] *= (width / image_w)
            boxes[:, [1, 3]] *= (height / image_h)

        return image, masks, boxes, labels

class Pad(object):
    def __init__(self, input_shape, pad_gt=True):
        self.height = input_shape[0]
        self.width  = input_shape[1]
        self.pad_gt = pad_gt

    def __call__(self, image, masks, boxes=None, labels=None):
        im_h, im_w, depth = image.shape
        expand_image                = np.zeros((self.height, self.width, depth), dtype=image.dtype)
        expand_image[:, :, :]       = [128, 128, 128]
        expand_image[:im_h, :im_w]  = image
        
        if self.pad_gt:
            expand_masks = np.zeros((masks.shape[0], self.height, self.width), dtype=masks.dtype)
            expand_masks[:, :im_h, :im_w] = masks
            masks = expand_masks
        return expand_image, masks, boxes, labels

#---------------------------------------------------------#
#   框的归一化
#---------------------------------------------------------#
class ToPercentCoords(object):
    def __call__(self, image, masks=None, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, masks, boxes, labels

#---------------------------------------------------------#
#   图像的归一化
#---------------------------------------------------------#
class BackboneTransform(object):
    def __init__(self, in_channel_order):
        self.mean   = np.array((103.94, 116.78, 123.68), dtype=np.float32)
        self.std    = np.array((57.38, 57.12, 58.40), dtype=np.float32)

        self.channel_map            = {c: idx for idx, c in enumerate(in_channel_order)}
        self.channel_permutation    = [self.channel_map[c] for c in "RGB"]

    def __call__(self, img, masks=None, boxes=None, labels=None):

        img = img.astype(np.float32)
        img = (img - self.mean) / self.std
        img = img[:, :, self.channel_permutation]

        return img.astype(np.float32), masks, boxes, labels

class BaseTransform(object):
    def __init__(self, input_shape):
        self.augment = Compose(
            [
                ConvertFromInts(), 
                Resize(input_shape), 
            ]
        )

    def __call__(self, img, masks=None, boxes=None, labels=None):
        return self.augment(img, masks, boxes, labels)

class Augmentation(object):
    def __init__(self, input_shape):
        self.augment = Compose(
            [
                ConvertFromInts(),
                ToAbsoluteCoords(),
                PhotometricDistort(),
                Expand(),
                RandomSampleCrop(),
                RandomMirror(),
                Resize(input_shape),
                ToPercentCoords(),
            ]
        )

    def __call__(self, img, masks, boxes, labels):
        return self.augment(img, masks, boxes, labels)
