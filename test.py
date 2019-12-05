import configparser


conf = configparser.ConfigParser()
conf.read("test.conf")
print(conf.get("win32", "log_dir"))
