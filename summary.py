#--------------------------------------------#
#   该部分代码只用于看网络结构，并非测试代码
#--------------------------------------------#
from nets.mrcnn import get_predict_model
from utils.config import Config

if __name__ == "__main__":
    model = get_predict_model(Config())
    model.summary()

    # for i,layer in enumerate(model.layers):
    #     print(i,layer.name)
