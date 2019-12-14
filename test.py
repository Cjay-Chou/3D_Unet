import configparser
import argparse
import numpy as np
import os

conf = configparser.ConfigParser()
conf.read("test.conf")
#print(conf.get("win32", "log_dir2"))
try:
    print(conf.get("win32", "log_dir2"))
except configparser.NoOptionError:
    pass
else:
    print('other error')

input = conf.get("all", "input_shape")
inputs = input[1:-1].split(',')
for i in range(len(inputs)):
    inputs[i] = int(inputs[i])
input_arr = np.array(inputs)
print(inputs)
ct_name = 'label.mha'

is_label = ct_name == 'label.mha'
print(is_label)