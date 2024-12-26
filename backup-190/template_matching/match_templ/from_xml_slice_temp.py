import os
import cv2
import xml.etree.ElementTree as ET

def main():
    xml_file = '/data/cdy/adc/data/img_src/9/123.xml'
    root_path = os.path.split(xml_file)[0]
    img_src = cv2.imread(os.path.splitext(xml_file)[0] + '.jpg', flags=-1)
    print(root_path)
    root = ET.parse(xml_file).getroot()
    object_ = root.find('object')
    name = object_.find('name').text
    bndbox = object_.find('bndbox')
    xmin = int(bndbox.find('xmin').text)
    ymin = int(bndbox.find('ymin').text)
    xmax = int(bndbox.find('xmax').text)
    ymax = int(bndbox.find('ymax').text)
    img_temp = img_src[ymin:ymax, xmin:xmax]
    print(name, [xmin, ymin, xmax, ymax])
    cv2.imwrite(os.path.join(root_path, name+'.jpg'), img_temp)


if __name__ == '__main__':
    main()