from keras.layers import (Activation, Add, BatchNormalization, Conv2D,
                          MaxPooling2D, ZeroPadding2D)


#----------------------------------------------#
#   conv_block和identity_block的区别主要就是：
#   conv_block会压缩输入进来的特征层的宽高
#   identity_block用于加深网络
#----------------------------------------------#
def identity_block(input_tensor, kernel_size, filters, stage, block, use_bias=True, train_bn=True):
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x, training=train_bn)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x, training=train_bn)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=use_bias)(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x, training=train_bn)

    x = Add()([x, input_tensor])
    x = Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), use_bias=True, train_bn=True):

    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), strides=strides, name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x, training=train_bn)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x, training=train_bn)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=use_bias)(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x, training=train_bn)

    shortcut = Conv2D(nb_filter3, (1, 1), strides=strides, name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut, training=train_bn)

    x = Add()([x, shortcut])
    x = Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x

#----------------------------------------------#
#   获得resnet的主干部分
#----------------------------------------------#
def get_resnet(input_image, train_bn=True):
    #----------------------------------------------#
    #   假设输入进来的图片为1024,1024,3
    #----------------------------------------------#

    # 1024,1024,3 -> 512,512,64
    x = ZeroPadding2D((3, 3))(input_image)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    x = BatchNormalization(name='bn_conv1')(x, training=train_bn)
    x = Activation('relu')(x)

    # 512,512,64 -> 256,256,64
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    C1 = x

    # 256,256,64 -> 256,256,256
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), train_bn=train_bn)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', train_bn=train_bn)
    C2 = x

    # 256,256,256 -> 128,128,512
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_bn=train_bn)
    C3 = x
    
    # 128,128,512 -> 64,64,1024
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', train_bn=train_bn)
    block_count = 22
    for i in range(block_count):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn)
    C4 = x
    
    # 64,64,1024 -> 32,32,2048
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', train_bn=train_bn)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_bn=train_bn)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', train_bn=train_bn)
    C5 = x 
    return [C1, C2, C3, C4, C5]
