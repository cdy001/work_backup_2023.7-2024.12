import numpy as np


def normalize_distances(points_list):
    """
    计算每个点与其他点之间的距离，并返回归一化的距离值列表

    参数：
    points_list: 包含[x, y]形式的点坐标的列表

    返回：
    distances: 距离值列表，与输入列表长度相同
    """
    # 将输入列表转换为numpy数组以便计算
    points_array = np.array(points_list)
    
    # 计算每对点之间的欧氏距离
    distances = []
    for i in range(len(points_array)):
        dist = np.sqrt(np.sum((points_array - points_array[i])**2, axis=1))
        distances.append(dist)
    # 计算每个点到其他点的平均距离
    distances = np.mean(distances, axis=1)
    # 归一化距离衡量值
    max_distance = np.max(distances)
    distances = (distances / max_distance)
    
    return distances

if __name__ == '__main__':
    # 示例用法
    points_list = [[1, 2], [4, 6], [2, 8], [5, 3], [100, 20]]
    normalized_distances = normalize_distances(points_list)

    print("归一化的距离值列表：", normalized_distances)
