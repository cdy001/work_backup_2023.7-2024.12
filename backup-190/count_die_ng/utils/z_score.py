import numpy as np

def z_score_outliers(data, threshold=3):
    """
    使用Z-score方法检测离群值

    参数：
    data: numpy数组或类似数组的数据集
    threshold: Z-score阈值，超过此阈值的数据点将被认为是离群值，默认为3

    返回：
    outliers: 包含所有离群值索引的列表
    """
    mean = np.mean(data)
    std_dev = np.std(data)
    z_scores = [(x - mean) / std_dev for x in data]
    outliers = np.where(np.abs(z_scores) > threshold)[0]
    return outliers

if __name__ == '__main__':
    # 示例用法
    data = np.array([1, 2, 3, 100, 5, 6, 7, 8, 9, 10])
    outliers = z_score_outliers(data, 1)
    print("离群值索引：", outliers)
    print("离群值：", data[outliers])
