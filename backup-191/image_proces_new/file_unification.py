import os
import sys
import glob
import shutil

# 复制文件
def copy(path_list, save_path):
    """
    复制文件(夹)列表下的所有数据到指定路径
    args:
        path_list: 文件(夹)路径列表
        save_path: 保存路径
    """
    for i, path in enumerate(path_list):
        print(f"{(i+1)}/{len(path_list)}")
        if os.path.isfile(path):
            try:
                shutil.move(path, save_path)
            except:
                continue
        elif os.path.isdir(path):
            files = os.listdir(path)
            _, path_name = os.path.split(path)
            save_file_path = os.path.join(save_path, path_name)
            if not os.path.exists(save_file_path):
                os.makedirs(save_file_path)
            for file_name in files:
                file_path = os.path.join(path, file_name)
                # shutil.copy(file_path, save_file_path)
                try:
                    shutil.move(file_path, save_file_path)
                except:
                    continue
        else:
            continue

        # 删除文件夹及文件夹下所有文件
        shutil.rmtree(path)

# print(os.getcwd())
# sys.path.append(os.getcwd())

if __name__ == '__main__':
    root_path = "/var/cdy_data/aoyang/data/BWB1O42A"
    # 创建文件汇总保存路径
    save_path = os.path.join(root_path, "0621-0702")
    print(f"save_path: {save_path}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # 待复制（移动）的图片路径
    img_dir_path = glob.glob(os.path.join(root_path, "*/*"))
    # # 排除某些不需要汇总的路径
    img_dir_path = [path for path in img_dir_path if "0945W" not in path]
    # img_dir_path = [path for path in img_dir_path if "0945U" in path]
    # 排除文件汇总保存路径
    img_dir_path = [path for path in img_dir_path if save_path+"/" not in path]
    for path in img_dir_path:
        print(path)
    # copy(img_dir_path, save_path)