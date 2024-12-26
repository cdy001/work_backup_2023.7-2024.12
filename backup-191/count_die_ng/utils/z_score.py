import numpy as np

def delete_indices(array, indices_to_delete):
    """
    从数组中删除指定索引的元素
    
    参数:
    array (numpy.ndarray): 原始数组
    indices_to_delete (numpy.ndarray): 需要删除的索引数组
    
    返回:
    numpy.ndarray: 删除指定索引后的数组
    """
    # 创建布尔掩码
    mask = np.ones(len(array), dtype=bool)
    mask[indices_to_delete] = False
    
    # 使用掩码筛选数组
    result_array = array[mask]
    
    return result_array

def z_score_outliers(data, threshold=3):
    """
    使用Z-score方法检测离群值

    参数：
    data: numpy数组或类似数组的数据集
    threshold: Z-score阈值, 超过此阈值的数据点将被认为是离群值, 默认为3

    返回：
    outliers: 包含所有离群值索引的列表
    """
    mean = np.mean(data)
    std_dev = np.std(data)
    # z_scores = [(x - mean) / std_dev for x in data]
    z_scores = [((1 + (x - mean) / std_dev) * ((x - mean) / std_dev)) for x in data]  # 加权z-score
    # outliers = np.where(np.abs(z_scores) > 7)[0]
    outliers_candinate = np.where(np.abs(z_scores) > 3)[0]
    # 剔除离群值之后重新计算mean和std
    data_new = delete_indices(np.array(data), outliers_candinate)
    mean_new = np.mean(data_new)
    std_new = np.std(data_new) + 1e-5  # 避免"devide by zero"错误
    z_scores_new = [(x - mean_new) / std_new for x in data]
    # z_scores_new = [((1 + (x - mean_new) / std_new) * ((x - mean_new) / std_new)) for x in data]
    outliers = np.where(np.abs(z_scores_new) > threshold)[0]
    return outliers

if __name__ == '__main__':
    # 示例用法
    data = np.array([1, 2, 3, 100, 5, 6, 7, 8, 9, 10])
    outliers = z_score_outliers(data, 1)
    print("离群值索引：", outliers)
    print("离群值：", data[outliers])
