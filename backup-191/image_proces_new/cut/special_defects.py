import cv2 as cv
import numpy as np


def specific_defect_uv(img, template):
    img = img.squeeze()
    template = template.squeeze()

    img_new = np.multiply(img, template)
    # cv.imwrite("img_new_1.bmp", img_new)

    # 面积比卡控逻辑
    area = np.sum(img_new != 0)
    img_new = np.logical_and(img_new > 1, img_new < 105)
    defect_area = np.sum(img_new)
    rate = defect_area / area
    if rate > 0.55:
        return False
    else:
        return True


def specific_defect_sy(img, template):
    img = img.squeeze()
    template = template.squeeze()

    img_new = np.multiply(img, template)
    # cv.imwrite("img_new_2.bmp", img_new)

    area = np.sum(img_new != 0)
    img_new = np.logical_and(img_new > 1, img_new < 165)
    defect_area = np.sum(img_new)
    rate = defect_area / area
    if rate > 0.012:
        return False
    else:
        return True


def process_image2(img):
    is_ok = True
    img = img.squeeze()
    height, width = img.shape

    img_one = np.ones((height - 50, width - 50))
    img_one_zero = np.pad(img_one, ((20, 30), (30, 20)), mode="constant")
    img = np.multiply(img, img_one_zero)
    # cv.imwrite("test1.bmp", img)

    img = np.where(np.logical_and(img >= 70, img <= 110), 255, 0)
    img = img.astype(np.uint8)
    # cv.imwrite("test2.bmp", img)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
    # cv.imwrite("test3.bmp", img)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 10))
    img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    img_area = np.sum(img != 0)
    # cv.imwrite("test4.bmp", img)
    if img_area > 57892 * 1.4 or img_area < 57892 * 0.6:
        is_ok = False

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    img = cv.erode(img, kernel)
    # cv.imwrite("test5.bmp", img)

    img = img / 255
    return img, is_ok

# img = cv.imread(r"test\2.bmp", 0)
# template, is_ok = process_image2(img)

# img = cv.imread(r"test\1.bmp", 0)
# specific_defect_uv(img, template)

# img = cv.imread(r"test\3.bmp", 0)
# specific_defect_sy(img, template)
