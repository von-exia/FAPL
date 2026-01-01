from __future__ import print_function
import MNN.numpy as np
import MNN
import MNN.cv as cv2
import MNN.nn as nn
from time import time

def inference(net, imgpath):
    """ inference ViT using a specific picture """
    # 预处理
    image = cv2.imread(imgpath)
    #cv2 read as bgr format
    image = image[..., ::-1]
    #change to rgb format
    image = cv2.resize(image, (224, 224)) / 255.
    #resize to mobile_net tensor size
    image = image - (0.48145466, 0.4578275, 0.40821073)
    image = image / (0.26862954, 0.26130258, 0.27577711)
    #change numpy data type as np.float32 to match tensor's format
    # image = image.astype(np.float32)
    #Make var to save numpy; [h, w, c] -> [n, h, w, c]
    input_var = np.expand_dims(image, [0])
    #cv2 read shape is NHWC, Module's need is NC4HW4, convert it
    input_var = MNN.expr.convert(input_var, MNN.expr.NC4HW4)
    #inference
    st = time()
    output_var = net.forward([input_var])
    ed = time()
    print("inference time: {:.3f} s".format((ed - st)))
    # 后处理及使用
    predict = output_var[0][0][0]
    print("Fake probability: {:.4f}".format(predict))

# 模型加载
config = {}
config['precision'] = 'low' # 当硬件支持（armv8.2）时使用fp16推理
config['backend'] = 0       # CPU
config['numThread'] = 4     # 线程数
rt = nn.create_runtime_manager((config,))
net = MNN.nn.load_module_from_file("./mnn_model./FAPL_detector.mnn", 
                                   ["img"], 
                                   ["prob"],
                                   runtime_manager=rt
                                   )
inference(net, "imgs/fake.png")
