def partition_path(path):
    l4_num = len(path)
    l4_num_patch = list(range(0, l4_num, 40))

    total_paths = []
    l4_num_patch_n = len(l4_num_patch)

    for i in range(l4_num_patch_n):
        if i == 0:
            total_paths.append(path[:l4_num_patch[i + 1]])
        elif i == (l4_num_patch_n - 1):
            total_paths.append(path[l4_num_patch[i]:])
        else:
            total_paths.append(path[l4_num_patch[i]:l4_num_patch[i + 1]])

    return total_paths


# 划分数据
def partition_path_list(refer_paths):
    if len(refer_paths) > 40:
        total_paths = partition_path(refer_paths)
    else:
        total_paths = [refer_paths]

    return total_paths
