from utils.z_score import z_score_outliers

def ngSkew(dies):
    '''
    args:
        dies: 识别到的所有die
    return:
        indexs: 判断为歪斜的die的索引
    '''
    area_ratio = []
    for i, die in enumerate(dies):
        x_min, y_min, x_max, y_max, contour_area, rect_area = die
        area_ratio.append(contour_area / rect_area)
    # 晶粒歪斜判断
    indexs = z_score_outliers(area_ratio, 5)
    return indexs