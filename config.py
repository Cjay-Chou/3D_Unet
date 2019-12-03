import sys

print(sys.platform)
if sys.platform == "win32":
    org_data_path = "E:/hist_01/comp30/"
    hist_data_path = "win321"
else:
    org_data_path = "ubuntu"
    hist_data_path = "ubuntu"