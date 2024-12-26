# coding=gbk
import os
import sys
sys.path.append(os.getcwd())
import cv2
import time
import numpy as np

from match_templ.utils import nms

# ģ��ƥ�䶨λ
def match_template_det(img_src, img_templ, thresh=0.5):
    """
    args:
        img_src: ��Ҫƥ���ԭʼ����ͼ��
        img_templ: ģ��ͼ��
        thresh: �����ɸѡ�ĳ�ʼ��ֵ
    return:
        dets: [x1, y1, x2, y2]  # ��������
        scores: [score1, score2, ...]  # ������Ӧ�Ĺ�һ��ƥ�����(0-1)
    """
    w, h = img_templ.shape[:2][::-1]
    w_src, h_src = img_src.shape[:2][::-1]
    
    # ����ԭͼ��ģ��
    scale_factor=0.1
    img_src = cv2.resize(img_src, None, fx=scale_factor, fy=scale_factor)
    img_templ = cv2.resize(img_templ, None, fx=scale_factor, fy=scale_factor)
    # for i in range(2):
    #     img_src = cv2.pyrDown(img_src)
    #     img_templ = cv2.pyrDown(img_templ)

    # ��img_src��Ե��0Ԫ�ؾ����Ա����ڱ�Ե����Ŀ��ƥ�䲻��
    img_zeros_h = np.zeros_like((img_src[:img_templ.shape[0]]))
    img_src = np.vstack([img_zeros_h, img_src, img_zeros_h])
    # img_zeros_w = np.zeros_like((img_src[:, :img_templ.shape[1]]))
    # img_src = np.hstack([img_zeros_w, img_src, img_zeros_w])

    result = cv2.matchTemplate(img_src, img_templ, cv2.TM_CCOEFF_NORMED)
    # ȡ�÷�ֵ����һ����ֵ�Ľ��
    loc = np.where(result >= thresh)
    # ���ɼ��򼰶�Ӧ�÷�ֵ
    dets = []
    scores = []
    for i in range(len(loc[0])):
        # ��ƥ���λ��ӳ���ԭͼ��λ��
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
        img_src: ��Ҫƥ���ԭʼ����ͼ��
        img_templ: ģ��ͼ��
        thresh: �����ɸѡ�ĳ�ʼ��ֵ
    return:
        dets: [x1, y1, x2, y2]  # ��������
        scores: [score1, score2, ...]  # ������Ӧ�Ĺ�һ��ƥ�����(0-1)
    """
    w, h = img_templ.shape[:2][::-1]
    w_src, h_src = img_src.shape[:2][::-1]

    # �������ų߶�����
    scale_factors = [0.1, ]
    # ��ʼ��һ���յ�ƥ��������
    final_result = np.zeros(shape=img_src.shape[:2])
    # �ڶ���߶��Ͻ���ģ��ƥ��
    for scale_factor in scale_factors:
        # ����ͼ���ģ��
        img_scaled = cv2.resize(img_src, None, fx=scale_factor, fy=scale_factor)
        template_scaled = cv2.resize(img_templ, None, fx=scale_factor, fy=scale_factor)

        # # ��img_src��Ե��0Ԫ�ؾ����Ա����ڱ�Ե����Ŀ��ƥ�䲻��
        # img_zeros_h = np.zeros_like((img_src[:img_templ.shape[0]]))
        # img_src = np.vstack([img_zeros_h, img_src, img_zeros_h])
        # # img_zeros_w = np.zeros_like((img_src[:, :img_templ.shape[1]]))
        # # img_src = np.hstack([img_zeros_w, img_src, img_zeros_w])

        # �����ź��ͼ���н���ģ��ƥ��
        result = cv2.matchTemplate(img_scaled, template_scaled, cv2.TM_CCOEFF_NORMED)

        # ȡ�÷�ֵ����һ����ֵ�Ľ��
        indices = np.argwhere(result > thresh)
        # ��result����ԭͼ�߶ȵ���Ӧλ��
        for row, col in indices:
            row_1 = int(row / scale_factor)
            col_1 = int(col / scale_factor)
            final_result[row_1, col_1] += result[row, col]

        # # �ۼ�ƥ����
        # final_result += result_resized
    loc = np.where(final_result)
    # ���ɼ��򼰶�Ӧ�÷�ֵ
    dets = []
    scores = []
    # for row, col in zip(loc[0], loc[1]):
    for i in range(len(loc[0])):
        # ��ƥ���λ��ӳ���ԭͼ��λ��
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
        img_src: ��Ҫƥ���ԭʼ����ͼ��
        img_templs: [ģ��ͼ��1, ģ��ͼ��2, ...]
        scale_factor: ģ��ƥ��ʱ���ű���
        thresh: �����ɸѡ�ĳ�ʼ��ֵ
        max_target_number: ���ƥ��Ŀ����
    return:
        nms_dets: [xmin, ymin, xmax, ymax], ����nms�����ļ����,(xmin, ymin)��(xmax, ymax)�ֱ��ʾ��������ϽǺ����½�
        nms_scores: nms_dets��Ӧ��ģ��ƥ��÷�
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

    # ��ƥ�������������
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