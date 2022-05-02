import collections
import datetime
import glob
import json
import os.path as osp

import labelme
import numpy as np
import PIL.Image
import pycocotools.mask

from utils.utils import get_classes
'''
标注自己的数据集需要注意以下几点：
1、我使用的labelme版本是3.16.7，建议使用该版本的labelme，
2、标注的数据集存放在datasets/before里面。
   jpg结尾的为图片文件
   json结尾的为标签文件
   图片文件和标签文件相对应
3、在标注目标时需要注意，同一种类的不同目标需要使用 _ 来隔开。   
   比如想要训练网络检测三角形和正方形，当一幅图片存在两个三角形时，一个标记为：   
   triangle_1
   另一个为：  
   triangle_2
'''
if __name__ == '__main__':
    #------------------------------------#
    #   训练自己的数据集必须要修改
    #   所需要区分的类别对应的txt文件
    #------------------------------------#
    classes_path    = "model_data/shape_classes.txt"
    #------------------------------------#
    #   labelme标注数据保存的位置
    #------------------------------------#
    input_dir       = "datasets/before"
    #------------------------------------#
    #   输出的图片文件保存的位置
    #------------------------------------#
    Img_output_dir  = "datasets/coco/JPEGImages"
    #------------------------------------#
    #   输出的json文件保存的位置
    #------------------------------------#
    Json_output_dir = "datasets/coco/Jsons"
    #--------------------------------------------------------------------------------------------------------------------------------#
    #   trainval_percent用于指定(训练集+验证集)与测试集的比例，默认情况下 (训练集+验证集):测试集 = 9:1
    #   train_percent用于指定(训练集+验证集)中训练集与验证集的比例，默认情况下 训练集:验证集 = 9:1
    #--------------------------------------------------------------------------------------------------------------------------------#
    trainval_percent    = 0.9
    train_percent       = 0.9

    #------------------------------------#
    #   获取当前时间
    #------------------------------------#
    now = datetime.datetime.now()
    #------------------------------------#
    #   找到所有标注好的json文件
    #------------------------------------#
    label_files     = glob.glob(osp.join(input_dir, '*.json'))
    #------------------------------------#
    #   对数据集进行打乱，并进行训练集、
    #   验证集和测试集的划分。
    #------------------------------------#
    np.random.seed(10101)
    np.random.shuffle(label_files)
    np.random.seed(None)
    num_train_val       = int(trainval_percent * len(label_files))
    num_train           = int(train_percent * num_train_val)

    train_label_files   = label_files[: num_train]
    val_label_files     = label_files[num_train : num_train_val]
    test_label_files    = label_files[num_train_val :]

    #------------------------------------#
    #   设定输出json文件的名称
    #------------------------------------#
    train_out_ann_file  = osp.join(Json_output_dir, 'train_annotations.json')
    val_out_ann_file    = osp.join(Json_output_dir, 'val_annotations.json')
    test_out_ann_file   = osp.join(Json_output_dir, 'test_annotations.json')

    #------------------------------------#
    #   获得列表
    #------------------------------------#
    label_files_list    = [train_label_files, val_label_files, test_label_files]
    out_ann_files_list  = [train_out_ann_file, val_out_ann_file, test_out_ann_file]
    data_list           = [
        dict(
            #------------------------------------#
            #   基础信息
            #------------------------------------#
            info = dict(
                description     = None,
                url             = None,
                version         = None,
                year            = now.year,
                contributor     = None,
                date_created    = now.strftime('%Y-%m-%d %H:%M:%S.%f'),
            ),
            #------------------------------------#
            #   许可证信息
            #------------------------------------#
            licenses=[
                dict(
                    url         = None,
                    id          = 0,
                    name        = None,
                )
            ],
            #------------------------------------#
            #   images是图片信息
            #------------------------------------#
            images=[
                # license, url, file_name, height, width, date_captured, id
            ],
            #------------------------------------#
            #   instances是实例
            #------------------------------------#
            type='instances',
            #------------------------------------#
            #   标签信息
            #------------------------------------#
            annotations=[
                # segmentation, area, iscrowd, image_id, bbox, category_id, id
            ],
            #------------------------------------#
            #   放的是需要区分的种类
            #------------------------------------#
            categories=[
                # supercategory, id, name
            ],
        ) for _ in range(3)
    ]

    #------------------------------------#
    #   该部分增加categories信息
    #------------------------------------#
    class_names, _      = get_classes(classes_path)
    class_names         = ["__ignore__", "_background_"] + class_names
    class_name_to_id    = {}
    for i, line in enumerate(class_names):
        class_id    = i - 1
        class_name  = line.strip()
        if class_id == -1:
            assert class_name == '__ignore__'
            continue
        class_name_to_id[class_name] = class_id
        for data in data_list:
            data['categories'].append(
                dict(
                    supercategory   = None,
                    id              = class_id,
                    name            = class_name,
                )
            )

    for label_files_index, label_files in enumerate(label_files_list):
        #------------------------------------#
        #   读取before文件夹里面的json文件
        #------------------------------------#
        for image_id, label_file in enumerate(label_files):
            print('Generating dataset from:', label_file)
            with open(label_file) as f:
                label_data = json.load(f)

            #------------------------------------#
            #   该部分增加images信息
            #   首先获取其对应的JPG图片
            #   然后保存到指定文件夹
            #   之后写入json数据
            #------------------------------------#
            base            = osp.splitext(osp.basename(label_file))[0]
            out_img_file    = osp.join(Img_output_dir, base + '.jpg')

            img_file = osp.join(osp.dirname(label_file), base + '.jpg')
            img = PIL.Image.open(img_file)
            img.save(out_img_file)
            img = np.asarray(img)
            data_list[label_files_index]['images'].append(
                dict(
                    license         = 0,
                    url             = None,
                    file_name       = base + '.jpg',
                    height          = img.shape[0],
                    width           = img.shape[1],
                    date_captured   = None,
                    id              = image_id,
                )
            )

            masks = {}
            segmentations = collections.defaultdict(list)
            for shape in label_data['shapes']:
                points      = shape['points']
                label       = shape['label']
                shape_type  = shape.get('shape_type', None)
                mask        = labelme.utils.shape_to_mask(img.shape[:2], points, shape_type)

                if label in masks:
                    masks[label] = masks[label] | mask
                else:
                    masks[label] = mask

                points = np.asarray(points).flatten().tolist()
                segmentations[label].append(points)

            for label, mask in masks.items():
                #------------------------------------#
                #   利用-进行分割
                #------------------------------------#
                cls_name = label.split('_')[0]
                if cls_name not in class_name_to_id:
                    continue
                cls_id = class_name_to_id[cls_name]

                #------------------------------------#
                #   获得mask，area和bbox坐标
                #------------------------------------#
                mask = np.asfortranarray(mask.astype(np.uint8))
                mask = pycocotools.mask.encode(mask)
                area = float(pycocotools.mask.area(mask))
                bbox = pycocotools.mask.toBbox(mask).flatten().tolist()

                #------------------------------------#
                #   该部分增加annotations信息
                #------------------------------------#
                data_list[label_files_index]['annotations'].append(dict(
                    id              = len(data_list[label_files_index]['annotations']),
                    image_id        = image_id,
                    category_id     = cls_id,
                    segmentation    = segmentations[label],
                    area            = area,
                    bbox            = bbox,
                    iscrowd         = 0,
                ))


        with open(out_ann_files_list[label_files_index], 'w') as f:
            json.dump(
                data_list[label_files_index], 
                f, 
                indent          = 4, 
                ensure_ascii    = False 
            )
