import configparser
import argparse
import numpy as np
import os
import SimpleITK as sitk
from tensorflow.keras.utils import to_categorical

conf = configparser.ConfigParser()
conf.read("test.conf")
print(conf.get("all", "is_label"))
is_label = conf.get("all", "is_label") is 'True'
print(is_label)
path1 = r'E:\3D_Unet\patchs\kits19_okd_label_s1\img0001\image1.mha'
path2 = r'E:\hist_01\comp30\img0001\c_label_8.mha'
image = sitk.ReadImage(path2)

image_array = sitk.GetArrayFromImage(image)
cat_array = to_categorical(image_array)
print(image_array.shape)
print(cat_array.shape)

image = sitk.ReadImage(path1)
image_array = sitk.GetArrayFromImage(image)
print(image_array.shape)