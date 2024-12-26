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

def retrive_contours_binary(img, lower_thr, upper_thr, struct_element_tuple):
    
    # # 滤波去噪
    # img = cv2.GaussianBlur(img, (5, 5), 0)

    time_start = time.time()
    thresh = __thresholdMinMax(img, lower_thr, upper_thr)

    time_1 = time.time()
    print(f"time threshold: {time_1 - time_start}")

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, struct_element_tuple)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    time_2 = time.time()
    print(f"time morphology: {time_2 - time_1}")

    contours, _ = cv2.findContours(image=thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    time_4 = time.time()
    print(f"time contours: {time_4 - time_2}")
    return contours