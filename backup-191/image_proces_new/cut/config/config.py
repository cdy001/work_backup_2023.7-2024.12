import os
import sys
sys.path.append(os.getcwd())
import json
import re
from cut.config.binary_config import binary_config
from cut.config.image_config import image_config
from cut.config.model_config import model_config


def get_config(recipe):
    image_info = None
    binary_info = None
    model_info = None
    for match_str in image_config:
        if re.match(match_str, recipe):
            image_info = image_config[match_str]

    for match_str in model_config:
        if re.match(match_str, recipe):
            model_info = model_config[match_str]

    for match_str in binary_config:
        if re.match(match_str, recipe):
            binary_info = binary_config[match_str]

    if image_info is None or model_info is None:
        print("not match config")
        exit()

    return image_info, binary_info, model_info

def get_config_new(recipe):
    # 获取当前脚本文件的绝对路径
    file_path = os.path.abspath(__file__)
    # 获取文件所在的目录
    current_directory = os.path.dirname(file_path)
    with open(os.path.join(current_directory, "config.json"), "r") as f:
        args = json.load(f)[recipe]
    binary_info, image_info, model_info = args.values()
    return image_info, binary_info, model_info

if __name__ == '__main__':
    recipe = "3535"
    # result = get_config(recipe)
    result = get_config_new(recipe)
    print(result)