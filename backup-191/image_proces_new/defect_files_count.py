import os

filename = 0
dir_name = 0
file_count = 0
path = r'C:\jucan_aoi\0945W\0829\缺陷数据样本'
for file in os.listdir(path):
    name_path = os.path.join(path, file)
    if os.path.isfile(name_path):
        filename += 1
    elif os.path.isdir(name_path):
        dir_name += 1
        file_count += len(os.listdir(name_path))
        if len(os.listdir(name_path)) > 0:
            print(f'{file}: {len(os.listdir(name_path))}')
print(f"总数量：{file_count}")