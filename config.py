import sys
import configparser


class UConfig:
    def __init__(self, config_path):
        conf = configparser.ConfigParser()
        conf.read(config_path)

        if sys.platform == "win32":
            self.log_dir = conf.get("win32", "log_dir")
            self.org_data_path = "E:/hist_01/comp30/"
            self.hist_data_path = "win321"
            self.patch_path = ""
        else:
            self.org_data_path = "ubuntu"
            self.hist_data_path = "ubuntu"

        self.inputShape = (128, 128, 32)
        self.outputShape = (128, 128, 32)



