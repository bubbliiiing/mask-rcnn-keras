# mask-rcnn-keras
这是一个mask-rcnn的库，可以用于训练自己的实例分割模型。

# 使用方法
## 1、准备数据集
a、利用labelme标注数据集，注意标注的时候同一个类要用不同的序号，比如画面中存在**两个苹果那么一个苹果的label就是apple1另一个是apple2。**    
b、标注完成后将jpg文件和json文件放在根目录下的before里面。  
c、之后运行json_to_dataset.py就可以生成train_dataset文件夹了。  
## 2、修改训练参数
a、dataset.py内修改自己要分的类，分别是load_shapes函数和load_mask函数内和类有关的内容，即将原有的circle、square等修改成自己要分的类。    
b、在train文件夹下面修改ShapesConfig(Config)的内容，NUM_CLASS等于自己要分的类的数量+1。  
c、IMAGE_MAX_DIM、IMAGE_MIN_DIM、BATCH_SIZE和IMAGES_PER_GPU根据自己的显存情况修改。RPN_ANCHOR_SCALES根据IMAGE_MAX_DIM和IMAGE_MIN_DIM进行修改。  
d、STEPS_PER_EPOCH代表每个世代训练多少次。   
## 3、预测
a、测试时下载好coco的h5文件运行即可，img内存在测试文件street.jpg。  
b、测试自身代码时将_defaults里面的参数修改成训练时用的参数。  

# 参考
https://github.com/matterport/Mask_RCNN
