# coding=gbk
import os
import sys
sys.path.append(os.getcwd())
import cv2
import glob

from match_templ.utils import load_src_templ
from match_templ.match_templ import match_templates_det



def main():
    src_root_path = '/data/cdy/adc/data/img_src/9'
    templ_path = glob.glob(os.path.join(src_root_path, 'temp*.jpg'))[0]
    for src_path in glob.glob(os.path.join(src_root_path, '*.jpg')):
        img_name = os.path.split(src_path)[-1]
        if 'temp' in img_name:
            continue
        # save_path = os.path.splitext(src_path)[0]
        save_path = os.path.join('/data/cdy/adc/data/img_die', img_name.replace('.jpg', ''))
        # print(save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # 1、读取大图和模板图
        img_src, img_templ = load_src_templ(
            src_path=src_path,
            templ_path=templ_path)
        # 模板图水平垂直翻转
        img_templ_flip = cv2.flip(img_templ, -1)
        

        # 2、使用模板图对大图进行模板匹配
        dets, scores = match_templates_det(
            img_src=img_src,
            img_templs=[img_templ, img_templ_flip],
            thresh=0.2,
            target_number=32
        )
        # print(dets)
        for det in dets:
            img_die = img_src[det[1]:det[3], det[0]:det[2]]
            save_die_path = os.path.join(save_path, img_name.replace(
                '.jpg', f'#{det[0]}_{det[1]}_{det[2]}_{det[3]}.jpg'
                ))
            # print(save_die_path)
            cv2.imwrite(save_die_path, img_die)

def test():
    src_path = '/data/cdy/adc/data/img_src/2/67BOSW00024-O.jpg'
    templ_path = '/data/cdy/adc/data/img_src/2/temp2.jpg'

    # 1、读取大图和模板图
    img_src, img_templ = load_src_templ(
        src_path=src_path,
        templ_path=templ_path)
    # 模板图水平垂直翻转
    img_templ_flip = cv2.flip(img_templ, -1)
    

    # 2、使用模板图对大图进行模板匹配
    dets, scores = match_templates_det(
        img_src=img_src,
        img_templs=[img_templ, img_templ_flip],
        thresh=0.2,
        target_number=40
    )
    # print(dets)
    for result in zip(dets, scores):
        print(result)

    # 检测结果显示
    img_disp = img_src.copy()
    # 绘制检测框
    for i, det in enumerate(dets):
        x1, y1, x2, y2 = det
        x1 = x1 - 10 if x1 - 10 > 0 else 0
        y1 = y1 - 10 if y1 - 10 > 0 else 0
        x2 = x2 + 10 if x2 + 10 < img_disp.shape[1] else img_disp.shape[1]
        y2 = y2 + 10 if y2 + 10 < img_disp.shape[0] else img_disp.shape[0]
        cv2.rectangle(img_disp,
                      (x1, y1),
                      (x2, y2),
                      color=(0, 255, 0),
                      thickness=3)
        cv2.putText(img_disp,
                    text='{} conf:{:.4f}'.format(i, scores[i]),
                    org=(x1, y1-10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=3,
                    color=(0, 0, 255),
                    thickness=3)
    # 保存图像
    img_name = os.path.split(src_path)[-1]
    # print(img_name)
    cv2.imwrite(os.path.join('/data/cdy/adc', img_name.replace('.jpg', '_det.jpg')), img_disp)
    
if __name__ == '__main__':
    # main()
    test()

