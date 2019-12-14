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

for i in range(0,100,8):
    print(i)
