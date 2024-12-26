import cv2 as cv


def retrieve_contours(img, binary_para):
    threshold = binary_para.get('threshold')
    binary_type = binary_para.get('binary_type')
    open_close_type = binary_para.get('open_close_type')
    struct_element_tuple = binary_para.get('struct_element_tuple')

    # 二值化
    binary_type_dict = {
        "THRESH_BINARY": cv.THRESH_BINARY,
        "THRESH_BINARY_INV": cv.THRESH_BINARY_INV
    }
    open_close_type_dict = {
        "MORPH_OPEN": cv.MORPH_OPEN,
        "MORPH_CLOSE": cv.MORPH_CLOSE
    }
    binary_type = binary_type_dict.get(binary_type)
    open_close_type = open_close_type_dict.get(open_close_type)

    _, thresh = cv.threshold(img, threshold, 255, binary_type)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, struct_element_tuple)
    thresh = cv.morphologyEx(thresh, open_close_type, kernel)

    contours, _ = cv.findContours(image=thresh, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)

    return contours, []
