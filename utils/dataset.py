import numpy as np
import skimage
import logging
import skimage.color
import skimage.io
import skimage.transform
#----------------------------------------------------------#
#  Dataset
#----------------------------------------------------------#

class Dataset(object):
    # 数据集训练的基本格式
    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        # 背景作为第一分类
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # 用于增加新的类
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                return
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, source, image_id, path, **kwargs):
        # 用于增加用于训练的图片
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def image_reference(self, image_id):
        return ""

    def prepare(self, class_map=None):
        # 准备数据
        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])
        # 分多少类
        self.num_classes = len(self.class_info)
        # 种类的id
        self.class_ids = np.arange(self.num_classes)
        # 搞个简称出来，用于显示
        self.class_names = [clean_name(c["name"]) for c in self.class_info]

        # 计算一共有多少个图片
        self.num_images = len(self.image_info)

        # 图片的id
        self._image_ids = np.arange(self.num_images)

        # 从源类和图像id到内部id的映射
        self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.class_info, self.class_ids)}
        self.image_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.image_info, self.image_ids)}

        # 建立sources
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

    def map_source_class_id(self, source_class_id):
        """Takes a source class ID and returns the int class ID assigned to it.

        For example:
        dataset.map_source_class_id("coco.12") -> 23
        """
        return self.class_from_source_map[source_class_id]

    def get_source_class_id(self, class_id, source):
        """Map an internal class ID to the corresponding class ID in the source dataset."""
        info = self.class_info[class_id]
        assert info['source'] == source
        return info['id']

    @property
    def image_ids(self):
        return self._image_ids

    def source_image_link(self, image_id):
        return self.image_info[image_id]["path"]

    def load_image(self, image_id):
        """
            载入图片
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def load_mask(self, image_id):
        '''
            载入语义分割内容
        '''
        logging.warning("You are using the default load_mask(), maybe you need to define your own one.")
        mask = np.empty([0, 0, 0])
        class_ids = np.empty([0], np.int32)
        return mask, class_ids