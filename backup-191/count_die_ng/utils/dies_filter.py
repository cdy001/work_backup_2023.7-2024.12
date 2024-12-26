from utils.point_distance import normalize_distances
from utils.z_score import z_score_outliers


def die_filter(points):
    # 计算点之间的归一化距离
    distances = normalize_distances(points)
    # 计算离群值
    outliner = z_score_outliers(distances, 3)
    indexs = set(outliner)  # 转换为索引，加速查找
    # filtered_points = [point for i, point in enumerate(points) if i not in indexs]
    indexs_save = [i for i, point in enumerate(points) if i not in indexs]
    return indexs_save

if __name__ == '__main__':
    # 示例用法
    points = [[1, 2], [2, 3], [3, 4], [20, 30], [21, 31], [22, 32]]
