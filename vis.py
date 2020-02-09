from lib.dataset.dataietr import DataIter
from train_config import config
from lib.core.model.ShuffleNet_Series.ShuffleNetV2.network import ShuffleNetV2

import torch
import time
import argparse


import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import cv2
from train_config import config as cfg
cfg.TRAIN.batch_size=1

ds = DataIter(cfg.DATA.root_path,cfg.DATA.val_txt_path,False)



def vis(model):

    ###build model
    face =torch.load(model,map_location=torch.device('cpu'))
    face.eval()
    for step in range(ds.size):

        images, labels = ds()
        img_show = np.array(images)
        print(img_show.shape)
        img_show=np.transpose(img_show[0],axes=[1,2,0])

        images=torch.from_numpy(images)

        start=time.time()
        res=face(images)
        res=res.detach().numpy()
        print(res)
        print('xxxx',time.time()-start)
        #print(res)

        img_show=img_show.astype(np.uint8)

        img_show=cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)

        landmark = np.array(res[0][0:136]).reshape([-1, 2])

        for _index in range(landmark.shape[0]):
            x_y = landmark[_index]
            #print(x_y)
            cv2.circle(img_show, center=(int(x_y[0] * 128),
                                         int(x_y[1] * 128)),
                       color=(255, 122, 122), radius=1, thickness=2)

        cv2.imshow('tmp',img_show)
        cv2.waitKey(0)

def vis_tflite(model):

    ###build model
    # 加载 TFLite 模型并分配张量（tensor）。
    interpreter = tf.lite.Interpreter(model_path=model)
    interpreter.allocate_tensors()

    # 获取输入和输出张量。
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for images, labels in train_dataset:
        img_show = np.array(images)

        images=np.expand_dims(images,axis=0)
        start=time.time()

        interpreter.set_tensor(input_details[0]['index'], images)

        interpreter.invoke()

        tflite_res = interpreter.get_tensor(output_details[2]['index'])


        print('xxxx',time.time()-start)
        #print(res)

        img_show=img_show.astype(np.uint8)

        img_show=cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)

        landmark = np.array(tflite_res).reshape([-1, 2])

        for _index in range(landmark.shape[0]):
            x_y = landmark[_index]
            #print(x_y)
            cv2.circle(img_show, center=(int(x_y[0] * 128),
                                         int(x_y[1] * 128)),
                       color=(255, 122, 122), radius=1, thickness=2)

        cv2.imshow('tmp',img_show)
        cv2.waitKey(0)
def load_checkpoint(net, checkpoint):
    # from collections import OrderedDict
    #
    # temp = OrderedDict()
    # if 'state_dict' in checkpoint:
    #     checkpoint = dict(checkpoint['state_dict'])
    # for k in checkpoint:
    #     k2 = 'module.'+k if not k.startswith('module.') else k
    #     temp[k2] = checkpoint[k]

    net.load_state_dict(torch.load(checkpoint,map_location=torch.device('cpu')), strict=True)
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Start train.')

    parser.add_argument('--model', dest='model', type=str, default=None, \
                        help='the model to use')

    args = parser.parse_args()


    if 'lite' in args.model:
        vis_tflite(args.model)
    else:
        vis(args.model)




