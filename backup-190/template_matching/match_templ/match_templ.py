# coding=gbk
import os
import sys
sys.path.append(os.getcwd())
import cv2
import time
import numpy as np

from match_templ.utils import nms

# 模板匹配定位
def match_template_det(img_src, img_templ, thresh=0.5):
    """
    args:
        img_src: 需要匹配的原始输入图像
        img_templ: 模板图像
        thresh: 检测结果筛选的初始阈值
    return:
        dets: [x1, y1, x2, y2]  # 检测框坐标
        scores: [score1, score2, ...]  # 与检测框对应的归一化匹配分数(0-1)
    """
    w, h = img_templ.shape[:2][::-1]
    w_src, h_src = img_src.shape[:2][::-1]
    
    # 缩放原图和模板
    scale_factor=0.1
    img_src = cv2.resize(img_src, None, fx=scale_factor, fy=scale_factor)
    img_templ = cv2.resize(img_templ, None, fx=scale_factor, fy=scale_factor)
    # for i in range(2):
    #     img_src = cv2.pyrDown(img_src)
    #     img_templ = cv2.pyrDown(img_templ)

    # 给img_src边缘补0元素矩阵，以避免在边缘处的目标匹配不到
    img_zeros_h = np.zeros_like((img_src[:img_templ.shape[0]]))
    img_src = np.vstack([img_zeros_h, img_src, img_zeros_h])
    # img_zeros_w = np.zeros_like((img_src[:, :img_templ.shape[1]]))
    # img_src = np.hstack([img_zeros_w, img_src, img_zeros_w])

    result = cv2.matchTemplate(img_src, img_templ, cv2.TM_CCOEFF_NORMED)
    # 取得分值高于一定阈值的结果
    loc = np.where(result >= thresh)
    # 生成检测框及对应得分值
    dets = []
    scores = []
    for i in range(len(loc[0])):
        # 将匹配的位置映射回原图的位置
        row, col = int(loc[0][i] / scale_factor), int(loc[1][i] / scale_factor)
        # row, col = int(loc[0][i]), int(loc[1][i])
        xmin = col
        ymin = row - h if row - h > 0 else 0
        xmax = col + w
        ymax = row if row < h_src else h_src
        dets.append([xmin, ymin, xmax, ymax])
        scores.append(result[loc[0][i], loc[1][i]])
    return dets, scores

def match_template_MutilScaleDet(img_src, img_templ, thresh=0.5):
    """
    args:
        img_src: 需要匹配的原始输入图像
        img_templ: 模板图像
        thresh: 检测结果筛选的初始阈值
    return:
        dets: [x1, y1, x2, y2]  # 检测框坐标
        scores: [score1, score2, ...]  # 与检测框对应的归一化匹配分数(0-1)
    """
    w, h = img_templ.shape[:2][::-1]
    w_src, h_src = img_src.shape[:2][::-1]

    # 定义缩放尺度因子
    scale_factors = [0.1, ]
    # 初始化一个空的匹配结果矩阵
    final_result = np.zeros(shape=img_src.shape[:2])
    # 在多个尺度上进行模板匹配
    for scale_factor in scale_factors:
        # 缩放图像和模板
        img_scaled = cv2.resize(img_src, None, fx=scale_factor, fy=scale_factor)
        template_scaled = cv2.resize(img_templ, None, fx=scale_factor, fy=scale_factor)

        # # 给img_src边缘补0元素矩阵，以避免在边缘处的目标匹配不到
        # img_zeros_h = np.zeros_like((img_src[:img_templ.shape[0]]))
        # img_src = np.vstack([img_zeros_h, img_src, img_zeros_h])
        # # img_zeros_w = np.zeros_like((img_src[:, :img_templ.shape[1]]))
        # # img_src = np.hstack([img_zeros_w, img_src, img_zeros_w])

        # 在缩放后的图像中进行模板匹配
        result = cv2.matchTemplate(img_scaled, template_scaled, cv2.TM_CCOEFF_NORMED)

        # 取得分值高于一定阈值的结果
        indices = np.argwhere(result > thresh)
        # 把result插入原图尺度的相应位置
        for row, col in indices:
            row_1 = int(row / scale_factor)
            col_1 = int(col / scale_factor)
            final_result[row_1, col_1] += result[row, col]

        # # 累加匹配结果
        # final_result += result_resized
    loc = np.where(final_result)
    # 生成检测框及对应得分值
    dets = []
    scores = []
    # for row, col in zip(loc[0], loc[1]):
    for i in range(len(loc[0])):
        # 将匹配的位置映射回原图的位置
        # row, col = int(loc[0][i] / scale_factor), int(loc[1][i] / scale_factor)
        row, col = int(loc[0][i]), int(loc[1][i])
        xmin = col
        ymin = row
        xmax = xmin + w if xmin + w < w_src else w_src
        ymax = ymin + h if ymin + h < h_src else h_src
        dets.append([xmin, ymin, xmax, ymax])
        # scores.append(result[row, col])
        scores.append(final_result[loc[0][i], loc[1][i]])
    return dets, scores

def match_templates_det(img_src, img_templs, thresh=0.5, target_number=0):
    """
    args:
        img_src: 需要匹配的原始输入图像
        img_templs: [模板图像1, 模板图像2, ...]
        scale_factor: 模板匹配时缩放比例
        thresh: 检测结果筛选的初始阈值
        max_target_number: 最大匹配目标数
    return:
        nms_dets: [xmin, ymin, xmax, ymax], 经过nms处理后的检测结果,(xmin, ymin)和(xmax, ymax)分别表示检测框的左上角和右下角
        nms_scores: nms_dets对应的模板匹配得分
    """
    dets_all, scores_all = [], []
    time_match_start = time.time()
    for img_templ in img_templs:
        dets, scores = match_template_det(
            img_src=img_src,
            img_templ=img_templ,
            thresh=thresh)
        # dets, scores = match_template_MutilScaleDet(
        #     img_src=img_src,
        #     img_templ=img_templ,
        #     thresh=thresh)
        dets_all.extend(dets)
        scores_all.extend(scores)
    time_match_end = time.time()
    print('match time: {:.4f}s'.format(time_match_end-time_match_start))
    dets_all = np.array(dets_all)
    scores_all = np.array(scores_all)

    # 按匹配分数降序排序
    # dets_scores = sorted(zip(dets_all, scores_all), key=lambda x:x[-1], reverse=True)
    # dets_all = np.array([det for det, _ in dets_scores][:100])
    # scores_all = np.array([score for _, score in dets_scores][:100])

    time_nms_start = time.time()
    nms_dets, nms_scores = nms(dets_all, scores_all, threshold=0.05)
    nms_dets = nms_dets[:target_number]
    nms_scores = nms_scores[:target_number]
    time_nms_end = time.time()
    print('nms time: {:.4f}s'.format(time_nms_end-time_nms_start))
    if not dets_all.any():
        print('detection numbers: 0')
        return []
    print('detection numbers: {}'.format(nms_dets.shape[0]))
    return nms_dets, nms_scores