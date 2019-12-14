import os
import sys
import configparser


class UConfig:
    def __init__(self, config_path):
        self.conf = configparser.ConfigParser()
        self.conf.read(config_path)

        self.input_shape = self.__get_shape("input_shape")
        self.output_shape = self.__get_shape("output_shape")
        self.data_name = self.conf.get("all", "data_name")
        self.step_scale = float(self.conf.get("all", "step_scale"))
        if self.conf.get("all", "mask_name") is 'None':
            self.mask_name = None
        else:
            self.mask_name = self.conf.get("all", "mask_name")

        if sys.platform == "win32":
            self.log_dir = self.conf.get("win32", "log_dir")
            self.org_data_path = self.conf.get("win32", "org_data_path")
            self.patch_path = self.conf.get("win32", "patch_path")
            self.list_path = self.conf.get("win32", "list_path")
        else:
            self.log_dir = "ubuntu"
            self.org_data_path = "ubuntu"
            self.patch_path = "ubuntu"

        self.train_list = self.__try_get("train_list")
        if self.train_list is None:
            self.train_list = os.listdir(self.org_data_path)
        self.val_list = self.__try_get("val_list")
        self.test_list = self.__try_get("test_list")

    def __get_shape(self, dtype):
        temp = self.conf.get("all", dtype)
        temps = temp[1:-1].split(',')
        for i in range(len(temps)):
            temps[i] = int(temps[i])
        return temps

    def __try_get(self, dtype):
        try:
            temp = self.conf.get("all", dtype)
        except configparser.NoOptionError:
            temp = None
        return temp
