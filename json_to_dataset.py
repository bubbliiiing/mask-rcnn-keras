import argparse
import json
import os
import os.path as osp
import warnings
 
import PIL.Image
import yaml
 
from labelme import utils
import base64
 
def main():
    count = os.listdir("./before/") 
    index = 0
    for i in range(0, len(count)):
        path = os.path.join("./before", count[i])

        if os.path.isfile(path) and path.endswith('json'):
            data = json.load(open(path))
            
            if data['imageData']:
                imageData = data['imageData']
            else:
                imagePath = os.path.join(os.path.dirname(path), data['imagePath'])
                with open(imagePath, 'rb') as f:
                    imageData = f.read()
                    imageData = base64.b64encode(imageData).decode('utf-8')
            img = utils.img_b64_to_arr(imageData)
            label_name_to_value = {'_background_': 0}
            for shape in data['shapes']:
                label_name = shape['label']
                if label_name in label_name_to_value:
                    label_value = label_name_to_value[label_name]
                else:
                    label_value = len(label_name_to_value)
                    label_name_to_value[label_name] = label_value
            
            # label_values must be dense
            label_values, label_names = [], []
            for ln, lv in sorted(label_name_to_value.items(), key=lambda x: x[1]):
                label_values.append(lv)
                label_names.append(ln)
            
            assert label_values == list(range(len(label_values)))
            
            lbl = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)
            
            captions = ['{}: {}'.format(lv, ln)
                for ln, lv in label_name_to_value.items()]
            lbl_viz = utils.draw_label(lbl, img, captions)

            if not os.path.exists("train_dataset"):
                os.mkdir("train_dataset")
            label_path = "train_dataset/mask"
            if not os.path.exists(label_path):
                os.mkdir(label_path)
            img_path = "train_dataset/imgs"
            if not os.path.exists(img_path):
                os.mkdir(img_path)
            yaml_path = "train_dataset/yaml"
            if not os.path.exists(yaml_path):
                os.mkdir(yaml_path)
            label_viz_path = "train_dataset/label_viz"
            if not os.path.exists(label_viz_path):
                os.mkdir(label_viz_path)

            PIL.Image.fromarray(img).save(osp.join(img_path, str(index)+'.jpg'))

            utils.lblsave(osp.join(label_path, str(index)+'.png'), lbl)
            PIL.Image.fromarray(lbl_viz).save(osp.join(label_viz_path, str(index)+'.png'))
 
            warnings.warn('info.yaml is being replaced by label_names.txt')
            info = dict(label_names=label_names)
            with open(osp.join(yaml_path, str(index)+'.yaml'), 'w') as f:
                yaml.safe_dump(info, f, default_flow_style=False)
            index = index+1
            print('Saved : %s' % str(index))
if __name__ == '__main__':
    main()