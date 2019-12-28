# !/usr/bin/python
# -*- coding:utf-8 -*-
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import sys

sys.path.append('../../')
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tool.yolo_utils as yolo_utils
from tool.config import Config
from tool.yolo_predict import predict
from datetime import datetime

base_anchors = [[10, 14], [23, 27], [37, 58], [81, 82], [135, 169], [344, 319]]
base_anchors = tf.constant(base_anchors, dtype=tf.float32)
base_anchors = tf.reshape(base_anchors, (2, 3, 2))
base_anchors = base_anchors[::-1]


def draw_gt(im, gt, wh=(1920, 1080)):
    w, h = wh
    im = im.astype(np.uint8)
    boxes = gt.astype(np.int32)
    print('***', boxes.shape)
    for box in boxes:
        # print(box)
        x1, y1, x2, y2 = box[:4]
        # y1, x1, y2, x2 = box[:4]
        im = cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 255))

    # for i in range(26):
    #     a = w / 26 * i
    #     a = int(a)
    #     cv2.line(im, (a, 0), (a, h), (255, 0, 0))
    # for i in range(26):
    #     a = h / 26 * i
    #     a = int(a)
    #     cv2.line(im, (0, a), (w, a), (255, 0, 0))

    im = im.astype(np.uint8)
    # im = im[..., ::-1]
    print(im.shape)
    cv2.imshow('a', im)
    cv2.waitKey(100)
    return im


def get_coord(N, stride):
    print(N, stride)
    t = tf.range(int(N / stride))
    x, y = tf.meshgrid(t, t)

    x = x[..., None]
    y = y[..., None]
    coord = tf.concat((x, y, x, y), axis=-1)
    coord = coord[:, :, None, :]
    coord = tf.to_float(coord) * stride
    return coord


def loc2bboxes(net, anchors, coord, stride):
    xy = tf.nn.sigmoid(net[..., :2]) * stride
    wh = tf.exp(net[..., 2:4]) * anchors
    xy1 = xy - wh / 2
    xy2 = xy + wh / 2
    bboxes = tf.concat((xy1, xy2), axis=-1) + coord
    return bboxes


def decode_net(net, anchors, coord, stride):
    xy = tf.nn.sigmoid(net[..., :2]) * stride
    wh = tf.exp(net[..., 2:4]) * anchors
    xy1 = xy - wh / 2
    xy2 = xy + wh / 2
    bboxes = tf.concat((xy1, xy2), axis=-1) + coord
    net = tf.nn.sigmoid(net[..., 4:])
    return tf.concat([bboxes, net], axis=-1)


def loadNumpyAnnotations(data):
    """
    Convert result data from a numpy array [Nx7] where each row contains {imageID,x1,y1,w,h,score,class}
    :param  data (numpy.ndarray)
    :return: annotations (python nested list)
    """
    print('Converting ndarray to lists...')
    assert (type(data) == np.ndarray)
    print(data.shape)
    assert (data.shape[1] == 7)
    N = data.shape[0]
    ann = []
    for i in range(N):
        if i % 1000000 == 0:
            print('{}/{}'.format(i, N))

        ann += [{
            'image_id': int(data[i, 0]),
            'bbox': [data[i, 1], data[i, 2], data[i, 3], data[i, 4]],
            'score': data[i, 5],
            'category_id': int(data[i, 6]),
        }]

    return ann


def py_resize(img, nh, nw):
    img = cv2.resize(img, (nw, nh))
    # cv2.imshow('img',img)
    # cv2.waitKey(100)
    # cv2.destroyAllWindows()

    return img.astype(np.float32)


def handle_im_same_wh(im_input, size):
    print(size)
    im = tf.image.decode_jpeg(im_input, channels=3)
    h = tf.shape(im)[0]
    w = tf.shape(im)[1]
    h = tf.to_float(h)
    w = tf.to_float(w)
    ma = tf.reduce_max([h, w])

    s = size / ma
    s = tf.to_float(s)

    nh = h * s
    nw = w * s
    nh = tf.to_int32(nh)
    nw = tf.to_int32(nw)
    # im = tf.image.resize_images(im, [nh, nw])
    im = tf.py_func(py_resize, [im, nh, nw], tf.float32)
    im.set_shape(tf.TensorShape([None, None, 3]))
    im = tf.expand_dims(im, axis=0)
    im = tf.pad(im, [[0, 0], [0, size - nh], [0, size - nw], [0, 0]], constant_values=127)
    im = im / 255.0
    im.set_shape(tf.TensorShape([None, None, None, 3]))

    return im, 1 / s, h, w


class YOLOv3():

    def __init__(self, config):
        self.config = config
        self.stride = config.stride
        self.blocks = yolo_utils.parse_cfg(config.cfg_file)
        self.coords = []
        for i in range(len(self.stride)):
            self.coords.append(get_coord(config.img_size, self.stride[i]))

    def build_net(self):
        size = self.config.img_size
        self.im_input = tf.placeholder(dtype=tf.string)
        im, self.b_scale, self.im_h, self.im_w = handle_im_same_wh(self.im_input, size=size)
        # im, self.b_scale, self.im_h, self.im_w = yolo_utils.handle_im_same_wh(self.im_input, size=size)
        self.n_im = im
        m = tf.shape(im)[0]

        with slim.arg_scope([slim.batch_norm], is_training=False, scale=True, decay=0.9, epsilon=1e-5):
            with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(self.config.weight_decay)):
                Name, end_points = yolo_utils.create_net(im, self.blocks)
        self.Name = Name

        for i in range(len(Name)):
            print(i, '\t', Name[i], '\t', end_points[i])

        net13 = end_points[16]
        net26 = end_points[-1]
        map_H = tf.shape(net13)[1]
        self.map_H = map_H
        map_H = 13
        C = [net13, net26]
        self.net13 = net13
        self.net26 = net26
        decode = []
        for i in range(2):
            tH = map_H * 2 ** i
            decode.append(tf.reshape(decode_net(tf.reshape(C[i], (-1, tH, tH, 3, self.config.num_cls + 5)),
                                                base_anchors[i], self.coords[i][:tH, :tH], self.stride[i]),
                                     (m, -1, self.config.num_cls + 5)))
        decode = tf.concat(decode, axis=1)
        self.result = predict(decode[0], size, self.config.num_cls, c_thresh=0.005, iou_thresh_=0.45)

    def test_coco(self):
        self.build_net()
        for v in tf.global_variables():
            print(v)
        catId2cls, cls2catId, catId2name = joblib.load(
            '/home/zhai/PycharmProjects/Demo35/Demo/(catId2cls,cls2catId,catId2name).pkl')
        test_dir = r'/home/zhai/PycharmProjects/Demo35/dataset/coco/val2017/'
        names = os.listdir(test_dir)
        names = [name.split('.')[0] for name in names]
        names = sorted(names)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        W = yolo_utils.load_weights(self.config.weight_file, self.blocks, self.Name, tf.global_variables(),
                                    (80 + 5) * 3, count=5)
        joblib.dump(W, 'yolov3_coco.pkl')
        print(len(W), len(tf.global_variables()))
        with tf.Session(config=config) as sess:
            self.init(sess, self.Name, tf.global_variables(), W)
            Res = []
            i = 0
            mm = 10
            time_start = datetime.now()
            for name in names[:mm]:
                i += 1
                print(datetime.now(), i)
                im_file = test_dir + name + '.jpg'
                img = tf.gfile.FastGFile(im_file, 'rb').read()
                # img=cv2.imread(im_file)[...,::-1]
                res, box_scale, h, w = sess.run([self.result, self.b_scale, self.im_h, self.im_w],
                                                feed_dict={self.im_input: img})
                res = res[..., :200]
                res[:, :4] = res[:, :4] * box_scale
                x1 = res[:, [0]]
                y1 = res[:, [1]]
                x2 = res[:, [2]]
                y2 = res[:, [3]]
                x1 = np.clip(x1, 0, w - 1)
                y1 = np.clip(y1, 0, h - 1)
                x2 = np.clip(x2, 0, w - 1)
                y2 = np.clip(y2, 0, h - 1)
                res[:, :4] = np.concatenate((x1, y1, x2, y2), axis=-1)
                wh = res[:, 2:4] - res[:, :2] + 1

                imgId = int(name)
                m = res.shape[0]

                imgIds = np.zeros((m, 1)) + imgId

                cls = res[:, 5]
                cid = map(lambda x: cls2catId[x], cls)
                cid = list(cid)
                cid = np.array(cid)
                cid = cid.reshape(-1, 1)

                res = np.concatenate((imgIds, res[:, :2], wh, res[:, 4:5], cid), axis=1)
                # Res=np.concatenate([Res,res])
                res = np.round(res, 4)
                Res.append(res)

            Res = np.concatenate(Res, axis=0)
            Ann = loadNumpyAnnotations(Res)
            print('==================================', mm, datetime.now() - time_start)

            with codecs.open('yolov3_bbox.json', 'w', 'ascii') as f:
                json.dump(Ann, f)
            eval_coco_box.eval('yolov3_bbox.json', mm)

    def test(self):
        self.build_net()

        file = r'D:\PycharmProjects\Demo36\YOLOv3_tiny\train_voc\models\yolov3_voc_3.ckpt-60000'

        saver = tf.train.Saver()

        test_dir = r'D:\dataset\VOCtest_06-Nov-2007\VOCdevkit\VOC2007\JPEGImages/'
        names = os.listdir(test_dir)
        names = [name.split('.')[0] for name in names if 'jpg' in name]
        names = sorted(names)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        Res = {}
        with tf.Session() as sess:
            saver.restore(sess, file)
            Res = {}
            i = 0
            m = 10000
            time_start = datetime.now()
            for name in names[:m]:
                i += 1
                print(datetime.now(), i)
                im_file = test_dir + name + '.jpg'
                img = tf.gfile.FastGFile(im_file, 'rb').read()
                res, box_scale, h, w, n_im = sess.run([self.result, self.b_scale, self.im_h, self.im_w, self.n_im],
                                                      feed_dict={self.im_input: img})

                inds = res[:, 4] > 0.005
                res = res[inds]

                # draw_gt(n_im[0] * 255, res, wh=(416, 416))
                res = res[:100]
                res[:, :4] = res[:, :4] * box_scale
                res[:, slice(0, 4, 2)] = np.clip(res[:, slice(0, 4, 2)], 0, w - 1)
                res[:, slice(1, 4, 2)] = np.clip(res[:, slice(1, 4, 2)], 0, h - 1)
                Res[name] = res

                # draw_gt(cv2.imread(im_file), res)

                # x1 = res[:, [0]]
                # y1 = res[:, [1]]
                # x2 = res[:, [2]]
                # y2 = res[:, [3]]
                # x1 = np.clip(x1, 0, w - 1)
                # y1 = np.clip(y1, 0, h - 1)
                # x2 = np.clip(x2, 0, w - 1)
                # y2 = np.clip(y2, 0, h - 1)
                # res[:, :4] = np.concatenate((x1, y1, x2, y2), axis=-1)
            #     Res[name] = res
            #
            print(datetime.now() - time_start)
            joblib.dump(Res, 'yolov3_1.pkl')

        pass

    pass


from sklearn.externals import joblib
import codecs
import json
import eval_coco_box
import cv2

if __name__ == "__main__":
    Mean = np.array([123.68, 116.78, 103.94], dtype=np.float32)
    path = '/home/zhai/PycharmProjects/Demo35/data_set_yxyx/'
    files = [path + 'voc_07.tf', path + 'voc_12.tf']
    cfgfile = r'yolov3-tiny.cfg'
    weightsfile = None

    config = Config(files, cfgfile, weightsfile, num_cls=20, img_size=416, ignore_thresh=0.5, stride=[32, 16])

    yolov3 = YOLOv3(config)
    yolov3.test()
    # yolov3.test_coco()
    pass
