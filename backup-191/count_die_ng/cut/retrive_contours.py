import cv2
import time

def __thresholdMinMax(gray_img, lower_threshold, upper_threshold, inverted=True):
    '''
    args:
        gray_img: 单通道的灰度图
        lower_threshold(int): 阈值下界
        upper_threshold(int): 阈值上界
        inverted(bool): 是否取反
    '''
    thr = cv2.inRange(gray_img, lower_threshold, upper_threshold)
    if inverted:
        thr = cv2.bitwise_not(thr)
    return thr

def retrive_contours_binary(img, lower_thr, upper_thr, struct_element_tuple, inverted=True):
    
    # # 滤波去噪
    # img = cv2.GaussianBlur(img, (5, 5), 0)

    # time_start = time.time()
    thresh = __thresholdMinMax(img, lower_thr, upper_thr, inverted)
    # binary_type = cv2.THRESH_BINARY if inverted else cv2.THRESH_BINARY_INV
    # _, thresh = cv2.threshold(img, upper_thr, 255, binary_type)

    # time_1 = time.time()
    # print(f"time threshold: {time_1 - time_start}")

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, struct_element_tuple)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # time_2 = time.time()
    # print(f"time morphology: {time_2 - time_1}")

    # contours, _ = cv2.findContours(image=thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    contours, _ = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    # time_4 = time.time()
    # print(f"time contours: {time_4 - time_2}")
    return contours

def retrive_contours_edge(img, Canny_thrs=(100, 300), close_struct=(3, 3)):
    time_start = time.time()
    # 滤波去噪(中值滤波保留边缘信息)
    # img = cv2.GaussianBlur(img, (5, 5), 0)
    # img = cv2.medianBlur(img, 5)
    # 边缘检测
    edges = cv2.Canny(
        image=img,
        threshold1=Canny_thrs[0],
        threshold2=Canny_thrs[1]
    )
    time_edge_end = time.time()
    # print(f"time edges detection: {time_edge_end - time_start}")
    # 形态学运算扩展边缘
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, close_struct)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    time_morph_end = time.time()
    # print(f"time morphology: {time_morph_end - time_edge_end}")
    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    time_contour_end = time.time()
    # print(f"time contour find: {time_contour_end - time_morph_end}")
    # # 填充轮廓内部
    # edges = cv2.drawContours(edges, contours, -1, (255), thickness=cv2.FILLED)
    # edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return contours, edges