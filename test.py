import configparser
import argparse
import random

import numpy as np
import os
import SimpleITK as sitk
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

from config import UConfig


def ImportImage(filename, data_type):
    image = sitk.ReadImage(filename)
    imagearry = sitk.GetArrayFromImage(image)
    if image.GetNumberOfComponentsPerPixel() == 1:
        imagearry = imagearry[..., np.newaxis]
    return imagearry


def GenerateBatchData(datalist, paddingsize=[44, 44, 44]):
    ps = paddingsize
    # j = 0

    while True:
        indices = list(range(len(datalist)))
        random.shuffle(indices)

        for idx in range(0, len(indices)):
            image = ImportImage(datalist[idx][0], 'ct')
            onehotlabel = ImportImage(datalist[idx][1], 'label')
            onehotlabel = onehotlabel[ps[0]:-ps[0], ps[1]:-ps[1], ps[2]:-ps[2]]

            yield (np.array(image), np.array(onehotlabel))


def ReadSliceDataList(c: UConfig, train_val):
    data_list_path = os.path.join('./lists', c.train_data)
    label_list_path = os.path.join('./lists', c.train_label)
    datalist = []
    if train_val == 'train':
        num_data_lists = c.train_list
    else:
        num_data_lists = c.val_list

    for i in num_data_lists:
        dfname = os.path.join(data_list_path, i)
        lfname = os.path.join(label_list_path, i)
        with open(dfname) as f:
            datas = f.readlines()
        with open(lfname) as f:
            labels = f.readlines()
        for imagefile, labelfile in zip(datas, labels):
            imagefile = imagefile.replace("\\\\", "\\")
            imagefile = imagefile.replace("\n", "")
            labelfile = labelfile.replace("\\\\", "\\")
            labelfile = labelfile.replace("\n", "")
            datalist.append((imagefile, labelfile))

    return datalist


conf = configparser.ConfigParser()
conf.read("test.conf")

config_path = "test.conf"
c = UConfig(config_path)
train_list = ReadSliceDataList(c, 'train')
'''for i in GenerateBatchData(train_list[:10]):
    print(i[1].shape)'''
# , output_shapes=((132, 132, 116, 1),(44, 44, 28, 9))
ds_counter = tf.data.Dataset.from_generator(lambda: GenerateBatchData(train_list), output_types=(tf.int32,tf.int32))
ds_counter = ds_counter.batch(5)
for count_batch in ds_counter.take(10):
    print(count_batch[0].shape)
