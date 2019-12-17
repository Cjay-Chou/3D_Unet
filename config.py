import os
import sys
import configparser


class UConfig:
    def __init__(self, config_path):
        self.conf = configparser.ConfigParser()
        self.conf.read(config_path)

        self.input_shape = self.__get_shape("input_shape")
        self.output_shape = self.__get_shape("output_shape")
        self.padding_type = self.__try_get("all","padding_type")
        self.data_name = self.conf.get("all", "data_name")
        self.is_label = self.__try_get("all", "is_label")
        self.step_scale = float(self.conf.get("all", "step_scale"))
        self.batch_size = int(self.__try_get("all", "batch_size"))

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
            self.log_dir = self.conf.get("Linux", "log_dir")
            self.org_data_path = self.conf.get("Linux", "org_data_path")
            self.patch_path = self.conf.get("Linux", "patch_path")
            self.list_path = self.conf.get("Linux", "list_path")

        self.train_list = self.__try_list("all", "train_list")
        if self.train_list is None:
            self.train_list = os.listdir(self.org_data_path)
        self.val_list = self.__try_list("all", "val_list")
        self.test_list = self.__try_list("all", "test_list")
        self.train_data = self.__try_get("all", "train_data")
        self.train_label = self.__try_get("all", "train_label")

    def __get_shape(self, dtype):
        temp = self.conf.get("all", dtype)
        temps = temp[1:-1].split(',')
        for i in range(len(temps)):
            temps[i] = int(temps[i])
        return temps

    def __try_get(self, where, dtype):
        try:
            temp = self.conf.get(where, dtype)
        except configparser.NoOptionError:
            temp = None
        return temp

    def __try_list(self, where, dtype):
        try:
            temp = self.conf.get(where, dtype)
            temp = temp.split(',')
            for i in range(len(temp)):
                temp[i] = temp[i].replace(' ', '')
                temp[i] = temp[i].replace("'", '')
                temp[i] = temp[i].replace('[', '')
                temp[i] = temp[i].replace(']', '')
        except configparser.NoOptionError:
            temp = None
        return temp
