import os
import sys
sys.path.append(os.getcwd())
import json
import re

def get_config_new(recipe):
    # 获取当前脚本文件的绝对路径
    file_path = os.path.abspath(__file__)
    # 获取文件所在的目录
    current_directory = os.path.dirname(file_path)
    with open(os.path.join(current_directory, "config.json"), "r") as f:
        args = json.load(f)[recipe]
    binary_info, image_info = args.values()
    return image_info, binary_info

if __name__ == '__main__':
    recipe = "OC2D58B-SANXING"
    # result = get_config(recipe)
    result = get_config_new(recipe)
    print(result)