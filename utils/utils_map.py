import json
import os.path as osp

import numpy as np
import pycocotools


class Make_json:
    def __init__(self, map_out_path, coco_label_map):
        self.map_out_path = map_out_path
        self.bbox_data = []
        self.mask_data = []
        self.coco_cats = {}

        for coco_id, real_id in coco_label_map.items():
            class_id = real_id - 1
            self.coco_cats[class_id] = coco_id

    def add_bbox(self, image_id: int, category_id: int, bbox: list, score: float):
        bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]

        bbox = [round(float(x) * 10) / 10 for x in bbox]

        self.bbox_data.append(
            {
                'image_id'      : int(image_id),
                'category_id'   : self.coco_cats[int(category_id)],
                'bbox'          : bbox,
                'score'         : float(score)
            }
        )

    def add_mask(self, image_id: int, category_id: int, segmentation: np.ndarray, score: float):
        rle = pycocotools.mask.encode(np.asfortranarray(segmentation.astype(np.uint8)))
        rle['counts'] = rle['counts'].decode('ascii')

        self.mask_data.append(
            {
                'image_id'      : int(image_id),
                'category_id'   : self.coco_cats[int(category_id)],
                'segmentation'  : rle,
                'score'         : float(score)
            }
        )

    def dump(self):
        dump_arguments = [
            (self.bbox_data, osp.join(self.map_out_path, "bbox_detections.json")),
            (self.mask_data, osp.join(self.map_out_path, "mask_detections.json"))
        ]

        for data, path in dump_arguments:
            with open(path, 'w') as f:
                json.dump(data, f)

def prep_metrics(pred_boxes, pred_confs, pred_classes, pred_masks, image_id, make_json):
    pred_classes    = list(np.array(pred_classes, np.int32))
    pred_confs      = list(np.array(pred_confs, np.float32))
    for i in range(pred_boxes.shape[0]):
        if (pred_boxes[i, 3] - pred_boxes[i, 1]) * (pred_boxes[i, 2] - pred_boxes[i, 0]) > 0:
            make_json.add_bbox(image_id, pred_classes[i], pred_boxes[i, :], pred_confs[i])
            make_json.add_mask(image_id, pred_classes[i], pred_masks[:, :, i], pred_confs[i])
