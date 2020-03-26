## Mask-Rcnn-实例分割模型在Keras当中的实现
---

### 目录
1. [所需环境 Environment](#所需环境)
2. [文件下载 Download](#文件下载)
3. [训练步骤 How2train](#训练步骤)
4. [参考资料 Reference](#Reference)

### 所需环境
tensorflow-gpu==1.13.1  
keras==2.1.5  

### 文件下载
这个训练好的权重是基于coco数据集的，可以直接运行用于coco数据集的实例分割。  
链接: https://pan.baidu.com/s/1tR7D2oqsa9O-9K6SA4YLJA 提取码: 15cj  


这个数据集是用于分辨图片中的圆形、正方形、三角形的，格式已经经过了处理，可以让大家明白训练集的格式。  
链接: https://pan.baidu.com/s/14dBd1Lbjw0FCnwKryf9taQ 提取码: 9457  

### 训练步骤
#### 1、准备数据集
a、利用labelme标注数据集，注意标注的时候同一个类要用不同的序号，比如画面中存在**两个苹果那么一个苹果的label就是apple1另一个是apple2。**    
b、标注完成后将jpg文件和json文件放在根目录下的before里面。  
c、之后运行json_to_dataset.py就可以生成train_dataset文件夹了。  
#### 2、修改训练参数
a、dataset.py内修改自己要分的类，分别是load_shapes函数和load_mask函数内和类有关的内容，即将原有的circle、square等修改成自己要分的类。    
b、在train文件夹下面修改ShapesConfig(Config)的内容，NUM_CLASS等于自己要分的类的数量+1。  
c、IMAGE_MAX_DIM、IMAGE_MIN_DIM、BATCH_SIZE和IMAGES_PER_GPU根据自己的显存情况修改。RPN_ANCHOR_SCALES根据IMAGE_MAX_DIM和IMAGE_MIN_DIM进行修改。  
d、STEPS_PER_EPOCH代表每个世代训练多少次。   
#### 3、预测
a、测试时下载好coco的h5文件运行即可，img内存在测试文件street.jpg。  
b、测试自身代码时将_defaults里面的参数修改成训练时用的参数。  

### Reference
https://github.com/matterport/Mask_RCNN
