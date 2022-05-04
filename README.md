## Mask-Rcnn-实例分割模型在Keras当中的实现
---

### 目录
1. [注意事项 Attention](#注意事项)
2. [仓库更新 Top News](#仓库更新)
3. [相关仓库 Related code](#相关仓库)
4. [所需环境 Environment](#所需环境)
5. [文件下载 Download](#文件下载)
6. [训练步骤 How2train](#训练步骤)
7. [预测步骤 How2predict](#预测步骤)
8. [评估步骤 miou](#评估步骤)
9. [参考资料 Reference](#Reference)

## 注意事项！
为了满足同学们计算mAP的需求，对代码进行了大改，现在使用COCO数据集格式，各位同学请仔细看README，视频中的步骤已经只能用于旧库，旧库地址参考Top News。

## Top News
**`2022-05`**:**进行大幅度更新、支持step、cos学习率下降法、支持adam、sgd优化器选择、支持学习率根据batch_size自适应调整、支持map评估。**  
BiliBili视频中的原仓库地址为：https://github.com/bubbliiiing/mask-rcnn-keras/tree/bilibili

**`2020-10`**:**创建仓库、支持训练与预测。**  

## 相关仓库
| 模型 | 路径 |
| :----- | :----- |
yolact-keras | https://github.com/bubbliiiing/yolact-keras  
yolact-pytorch | https://github.com/bubbliiiing/yolact-pytorch
yolact-tf2 | https://github.com/bubbliiiing/yolact-tf2
mask-rcnn-keras | https://github.com/bubbliiiing/mask-rcnn-keras

## 所需环境
tensorflow-gpu==1.13.1  
keras==2.1.5  

## 文件下载
这个训练好的权重是基于coco数据集的，可以直接运行用于coco数据集的实例分割。  
链接: https://pan.baidu.com/s/1JXdNZ_dTCjtxjLmrxFWrag     
提取码: mpzp     

shapes数据集下载地址如下，该数据集是使用labelme标注的结果，尚未经过其它处理，用于区分三角形和正方形：  
链接: https://pan.baidu.com/s/1hrCaEYbnSGBOhjoiOKQmig   
提取码: jk44    

## 训练步骤
### a、训练shapes形状数据集
1. 数据集的准备   
在**文件下载**部分，通过百度网盘下载数据集，下载完成后解压，将图片和对应的json文件放入根目录下的datasets/before文件夹。

2. 数据集的处理   
打开coco_annotation.py，里面的参数默认用于处理shapes形状数据集，直接运行可以在datasets/coco文件夹里生成图片文件和标签文件，并且完成了训练集和测试集的划分。

3. 开始网络训练   
train.py的默认参数用于训练shapes数据集，默认指向了根目录下的数据集文件夹，直接运行train.py即可开始训练。   

4. 训练结果预测   
训练结果预测需要用到两个文件，分别是mask_rcnn.py和predict.py。
首先需要去mask_rcnn.py里面修改model_path以及classes_path，这两个参数必须要修改。    
**model_path指向训练好的权值文件，在logs文件夹里。   
classes_path指向检测类别所对应的txt。**    
完成修改后就可以运行predict.py进行检测了。运行后输入图片路径即可检测。   

### b、训练自己的数据集
1. 数据集的准备  
**本文使用labelme工具进行标注，标注好的文件有图片文件和json文件，二者均放在before文件夹里，具体格式可参考shapes数据集。**    
在标注目标时需要注意，同一种类的不同目标需要使用 _ 来隔开。   
比如想要训练网络检测**三角形和正方形**，当一幅图片存在两个三角形时，分别标记为：   
```python
triangle_1
triangle_2
```
2. 数据集的处理  
修改coco_annotation.py里面的参数。第一次训练可以仅修改classes_path，classes_path用于指向检测类别所对应的txt。    
训练自己的数据集时，可以自己建立一个cls_classes.txt，里面写自己所需要区分的类别。    
model_data/cls_classes.txt文件内容为：      
```python
cat
dog
...
```  
修改coco_annotation.py中的classes_path，使其对应cls_classes.txt，并运行coco_annotation.py。    

3. 开始网络训练  
**训练的参数较多，均在train.py中，大家可以在下载库后仔细看注释，其中最重要的部分依然是train.py里的classes_path。**   
**classes_path用于指向检测类别所对应的txt，这个txt和coco_annotation.py里面的txt一样！训练自己的数据集必须要修改！**    
修改完classes_path后就可以运行train.py开始训练了，在训练多个epoch后，权值会生成在logs文件夹中。   

4. 训练结果预测  
训练结果预测需要用到两个文件，分别是mask_rcnn.py和predict.py。
首先需要去mask_rcnn.py里面修改model_path以及classes_path，这两个参数必须要修改。    
**model_path指向训练好的权值文件，在logs文件夹里。   
classes_path指向检测类别所对应的txt。**     
完成修改后就可以运行predict.py进行检测了。运行后输入图片路径即可检测。     

### c、训练coco数据集
1. 数据集的准备  
coco训练集 http://images.cocodataset.org/zips/train2017.zip   
coco验证集 http://images.cocodataset.org/zips/val2017.zip   
coco训练集和验证集的标签 http://images.cocodataset.org/annotations/annotations_trainval2017.zip   

2. 开始网络训练  
解压训练集、验证集及其标签后。打开train.py文件，修改其中的classes_path指向model_data/coco_classes.txt。   
修改train_image_path为训练图片的路径，train_annotation_path为训练图片的标签文件，val_image_path为验证图片的路径，val_annotation_path为验证图片的标签文件。   

3. 训练结果预测  
训练结果预测需要用到两个文件，分别是mask_rcnn.py和predict.py。
首先需要去mask_rcnn.py里面修改model_path以及classes_path，这两个参数必须要修改。    
**model_path指向训练好的权值文件，在logs文件夹里。   
classes_path指向检测类别所对应的txt。**    
完成修改后就可以运行predict.py进行检测了。运行后输入图片路径即可检测。   

## 预测步骤
### a、使用预训练权重
1. 下载完库后解压，在百度网盘下载权值，放入model_data，运行predict.py，输入   
```python
img/street.jpg
```
2. 在predict.py里面进行设置可以进行fps测试和video视频检测。   
### b、使用自己训练的权重
1. 按照训练步骤训练。    
2. 在mask_rcnn.py文件里面，在如下部分修改model_path和classes_path使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件，classes_path是model_path对应分的类**。   
```python
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
```
3. 运行predict.py，输入    
```python
img/street.jpg
```
4. 在predict.py里面进行设置可以进行fps测试和video视频检测。    

## 评估步骤 
### a、评估自己的数据集
1. 本文使用coco格式进行评估。    
2. 如果在训练前已经运行过coco_annotation.py文件，代码会自动将数据集划分成训练集、验证集和测试集。
3. 如果想要修改测试集的比例，可以修改coco_annotation.py文件下的trainval_percent。trainval_percent用于指定(训练集+验证集)与测试集的比例，默认情况下 (训练集+验证集):测试集 = 9:1。train_percent用于指定(训练集+验证集)中训练集与验证集的比例，默认情况下 训练集:验证集 = 9:1。
4. 在mask_rcnn.py里面修改model_path以及classes_path。**model_path指向训练好的权值文件，在logs文件夹里。classes_path指向检测类别所对应的txt。**    
5. 前往eval.py文件修改classes_path，classes_path用于指向检测类别所对应的txt，这个txt和训练时的txt一样。评估自己的数据集必须要修改。运行eval.py即可获得评估结果。  

### b、评估coco的数据集
1. 下载好coco数据集。  
2. 在mask_rcnn.py里面修改model_path以及classes_path。**model_path指向coco数据集的权重，在logs文件夹里。classes_path指向model_data/coco_classes.txt。**    
3. 前往eval.py设置classes_path，指向model_data/coco_classes.txt。修改Image_dir为评估图片的路径，Json_path为评估图片的标签文件。 运行eval.py即可获得评估结果。  
  
## Reference
https://github.com/matterport/Mask_RCNN     
https://github.com/feiyuhuahuo/Yolact_minimal     
